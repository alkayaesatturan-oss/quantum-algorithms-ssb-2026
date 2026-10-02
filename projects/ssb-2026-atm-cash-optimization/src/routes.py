"""
Classical sub-layer: generates a pool of feasible route "columns" that
start and end at VEZNE (the depot) and do not exceed 300 minutes. This
pool is subsequently transformed by qubo.py into a "which set of routes
should be selected" combinatorial problem (a set-packing QUBO).

There is NO capacity constraint (Section 1.1: "the transportable amount
is assumed to be unlimited") -- this module never applies a cash/capacity
constraint; it applies only the duration constraint (300 min).

The pool size is FIXED: `generate_route_pool` returns at most `pool_size`
columns. Since the qubit count = pool size + number of vehicles, this cap
keeps the qubit budget of the quantum step identical across runs (because
OR-Tools runs under a time limit, the raw pool may vary from run to run;
the selection step does not propagate this variability to the NUMBER of
columns).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .distance import travel_minutes_matrix

try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    _HAS_ORTOOLS = True
except Exception:  # pragma: no cover
    _HAS_ORTOOLS = False

TIME_SCALE = 10  #

@dataclass
class RouteContext:
    locationnos: list          # candidate location IDs (excluding the depot); index order matters
    time_matrix: np.ndarray    # (n+1) x (n+1), index 0 = VEZNE
    service_minutes: float
    max_minutes: float
    mandatory_mask: np.ndarray  # len == n, True = mandatory
    utility: np.ndarray         # len == n

    def route_minutes(self, seq: list[int]) -> float:
        """seq: candidate indices (0..n-1) in visiting order, excluding the depot."""
        if not seq:
            return 0.0
        total = self.time_matrix[0, seq[0] + 1]
        for a, b in zip(seq, seq[1:]):
            total += self.time_matrix[a + 1, b + 1]
        total += self.time_matrix[seq[-1] + 1, 0]
        total += self.service_minutes * len(seq)
        return float(total)

    def solo_minutes(self, i: int) -> float:
        return self.route_minutes([i])


def build_context(candidates: pd.DataFrame, depot_lat: float, depot_lon: float, cfg: Config) -> RouteContext:
    lats = np.concatenate([[depot_lat], candidates["latitude"].to_numpy(dtype=float)])
    lons = np.concatenate([[depot_lon], candidates["longitude"].to_numpy(dtype=float)])
    tmat = travel_minutes_matrix(lats, lons, cfg.avg_speed_kmh)
    return RouteContext(
        locationnos=candidates["locationno"].tolist(),
        time_matrix=tmat,
        service_minutes=cfg.service_minutes_per_stop,
        max_minutes=cfg.max_route_minutes,
        mandatory_mask=candidates["mandatory"].to_numpy(dtype=bool),
        utility=candidates["utility_score"].to_numpy(dtype=float),
    )


def structurally_unreachable(ctx: RouteContext) -> list[int]:
    """Locations whose round trip alone already exceeds 300 minutes -- they
    cannot fit into any route in any way (the 'otherwise no feasible
    solution can be produced' exception of Section 1.1, item 12)."""
    n = len(ctx.locationnos)
    return [i for i in range(n) if ctx.solo_minutes(i) > ctx.max_minutes + 1e-6]


def _two_opt(ctx: RouteContext, seq: list[int]) -> list[int]:
    improved = True
    while improved and len(seq) > 3:
        improved = False
        for i in range(len(seq) - 1):
            for j in range(i + 2, len(seq)):
                new_seq = seq[:i + 1] + seq[i + 1:j + 1][::-1] + seq[j + 1:]
                if ctx.route_minutes(new_seq) < ctx.route_minutes(seq) - 1e-9:
                    seq = new_seq
                    improved = True
    return seq


def _greedy_insertion_route(ctx: RouteContext, pool: list[int], rng: random.Random,
                             utility_power: float = 1.0, must_include: list[int] | None = None) -> list[int]:
    """Randomized greedy route constructor (GRASP-style) based on cheapest
    insertion + a utility/added-time score. If `must_include` is given,
    these nodes are inserted first (as space becomes available)."""
    seq: list[int] = []
    remaining = set(pool)

    def try_insert(node: int) -> bool:
        best_delta, best_pos = None, None
        for pos in range(len(seq) + 1):
            trial = seq[:pos] + [node] + seq[pos:]
            dt = ctx.route_minutes(trial) - ctx.route_minutes(seq)
            if best_delta is None or dt < best_delta:
                best_delta, best_pos = dt, pos
        if best_delta is not None and ctx.route_minutes(seq) + best_delta <= ctx.max_minutes + 1e-6:
            seq.insert(best_pos, node)
            return True
        return False

    if must_include:
        for node in must_include:
            if node in remaining and try_insert(node):
                remaining.discard(node)

    while remaining:
        scored = []
        for node in remaining:
            pos_best = None
            for pos in range(len(seq) + 1):
                trial = seq[:pos] + [node] + seq[pos:]
                dt = ctx.route_minutes(trial) - ctx.route_minutes(seq)
                if ctx.route_minutes(seq) + dt > ctx.max_minutes + 1e-6:
                    continue
                if pos_best is None or dt < pos_best[0]:
                    pos_best = (dt, pos)
            if pos_best is not None:
                dt, pos = pos_best
                score = (ctx.utility[node] ** utility_power) / max(dt, 1.0)
                scored.append((score, node, pos))
        if not scored:
            break
        scored.sort(reverse=True)
        top_k = scored[: max(1, len(scored) // 3)]
        _, node, pos = rng.choice(top_k)
        seq.insert(pos, node)
        remaining.discard(node)

    return _two_opt(ctx, seq)


def _column(ctx: RouteContext, seq: list[int], source: str, run: int | None = None) -> dict:
    return {
        "nodes": list(seq),
        "minutes": ctx.route_minutes(seq),
        "utility": float(ctx.utility[seq].sum()) if seq else 0.0,
        "source": source,
        "run": run,   # OR-Tools run index (0 = classical reference solution); None for GRASP
    }


def generate_route_pool(ctx: RouteContext, num_vehicles: int, pool_size: int,
                         seed: int = 0) -> tuple[list[dict], list[int]]:

    if pool_size < num_vehicles:
        raise ValueError(f"pool_size={pool_size} cannot be smaller than the number of vehicles ({num_vehicles}).")

    rng = random.Random(seed)
    n = len(ctx.locationnos)
    unreachable = set(structurally_unreachable(ctx))
    reachable = [i for i in range(n) if i not in unreachable]
    mandatory_reachable = [i for i in reachable if ctx.mandatory_mask[i]]

    raw: list[dict] = []
    seen: set[tuple] = set()

    def _add(seq: list[int], source: str, run: int | None = None) -> bool:
        if not seq or ctx.route_minutes(seq) > ctx.max_minutes + 1e-6:
            return False
        key = tuple(sorted(seq))
        if key in seen:
            return False
        seen.add(key)
        raw.append(_column(ctx, seq, source, run))
        return True

    run_key_sets: list[list[tuple]] = []
    if _HAS_ORTOOLS and reachable:
        try:
            runs = _ortools_routes(ctx, reachable, mandatory_reachable, num_vehicles, rng)
        except Exception:
            runs = []
        for run_i, run_routes in enumerate(runs):
            keys = []
            for r in run_routes:
                _add(r, "ortools", run=run_i)
                keys.append(tuple(sorted(r)))
            run_key_sets.append(keys)

    target_raw = 2 * pool_size
    attempts = 0
    while len(raw) < target_raw and attempts < pool_size * 12:
        attempts += 1
        power = rng.choice([0.6, 1.0, 1.5, 2.2])
        must = rng.sample(mandatory_reachable, k=min(len(mandatory_reachable), rng.randint(0, 3))) \
            if mandatory_reachable else []
        seq = _greedy_insertion_route(ctx, reachable, rng, utility_power=power, must_include=must)
        _add(seq, "grasp")

    raw.sort(key=lambda c: -c["utility"])

    selected: list[dict] = []
    sel_keys: set[tuple] = set()

    def _pick(c: dict) -> None:
        k = tuple(sorted(c["nodes"]))
        if k not in sel_keys:
            sel_keys.add(k)
            selected.append(c)

    by_key = {tuple(sorted(c["nodes"])): c for c in raw}
    for keys in run_key_sets:                      
        if any(k not in by_key for k in keys):
            continue                              
        if len(selected) + sum(k not in sel_keys for k in keys) > pool_size and selected:
            break                                 
        for k in keys:
            _pick(by_key[k])
    for m in mandatory_reachable:                  
        if any(m in c["nodes"] for c in selected):
            continue
        best = next((c for c in raw if m in c["nodes"]), None)
        if best is None:
            dedicated = _greedy_insertion_route(ctx, reachable, rng, utility_power=1.0, must_include=[m])
            if m not in dedicated:
                dedicated = [m]
            best = _column(ctx, dedicated, "dedicated")
        _pick(best)
    for c in raw:                                  
        if len(selected) >= pool_size:
            break
        _pick(c)

    selected.sort(key=lambda c: -c["utility"])
    return selected, sorted(unreachable)


def _ortools_routes(ctx: RouteContext, reachable: list[int], mandatory_reachable: list[int],
                     num_vehicles: int, rng: random.Random,
                     runs: int = 4, time_limit_s: int = 3) -> list[list[list[int]]]:
    n = len(reachable)
    node_ids = [0] + [j + 1 for j in reachable]  # 0 = depot, ctx-index+1 = ortools index
    sub_time = ctx.time_matrix[np.ix_(node_ids, node_ids)]
    service = np.array([0.0] + [ctx.service_minutes] * n)
    utilities = ctx.utility[reachable]

    def _solve(jitter: np.ndarray, hard_mandatory: bool, limit_s: int):
        manager = pywrapcp.RoutingIndexManager(n + 1, num_vehicles, 0)
        routing = pywrapcp.RoutingModel(manager)

        def time_cb(from_index, to_index):
            f = manager.IndexToNode(from_index)
            t = manager.IndexToNode(to_index)
            return int(math.ceil((sub_time[f, t] + service[f]) * TIME_SCALE - 1e-9))

        transit_idx = routing.RegisterTransitCallback(time_cb)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)
        routing.AddDimension(transit_idx, 0, int(ctx.max_minutes * TIME_SCALE), True, "Time")

        profit = utilities * jitter
        big_m = int(utilities.sum() * 1000) + 10_000
        for local_i, global_i in enumerate(reachable):
            node_index = manager.NodeToIndex(local_i + 1)
            if ctx.mandatory_mask[global_i]:
                if not hard_mandatory:
                    routing.AddDisjunction([node_index], big_m)
                # hard_mandatory: NO disjunction -> the node must be visited (hard constraint)
            else:
                routing.AddDisjunction([node_index], int(profit[local_i]))

        search = pywrapcp.DefaultRoutingSearchParameters()
        search.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
        search.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search.time_limit.FromSeconds(limit_s)

        solution = routing.SolveWithParameters(search)
        if solution is None:
            return None
        run_routes = []
        for v in range(num_vehicles):
            idx = routing.Start(v)
            seq = []
            while not routing.IsEnd(idx):
                node = manager.IndexToNode(idx)
                if node != 0:
                    seq.append(reachable[node - 1])
                idx = solution.Value(routing.NextVar(idx))
            if seq:
                run_routes.append(seq)
        return run_routes

    runs_out: list[list[list[int]]] = []

    ones = np.ones(n)
    first = _solve(ones, hard_mandatory=True, limit_s=max(time_limit_s, 3) * 3)
    if first is None:
        first = _solve(ones, hard_mandatory=False, limit_s=time_limit_s)
    if first is not None:
        runs_out.append(first)
    for run in range(1, runs):
        jitter = np.array([rng.uniform(0.85, 1.15) for _ in range(n)])
        rr = _solve(jitter, hard_mandatory=True, limit_s=time_limit_s)
        if rr is None:
            rr = _solve(jitter, hard_mandatory=False, limit_s=time_limit_s)
        if rr is not None:
            runs_out.append(rr)
    return runs_out
