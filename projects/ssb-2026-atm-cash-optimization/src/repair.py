"""
Converts the raw bitstring returned by the quantum/classical solver into a
STRICTLY feasible set of routes. Although the solvers already tend to
return feasible bitstrings (because the penalty coefficients are chosen to
be dominant, see qubo.py), this layer serves as a safety net -- for cases
where convergence is incomplete or the pool grows -- and is ALWAYS
executed; it never silently accepts the result as is.

The repair preserves the existing quantum/classical PROPOSAL as far as
possible (it does not rebuild the routes from scratch); it only (1)
deduplicates overlapping nodes, (2) pads/trims the selection to the number
of vehicles (K), and (3) attempts to fit in missing mandatory nodes. If a
mandatory node genuinely cannot be accommodated, it is NOT SILENTLY
SKIPPED -- it is explicitly reported in the `blocked_mandatory` list (the
"otherwise no feasible solution can be produced" exception of Section 1.1,
item 12).
"""
from __future__ import annotations

import numpy as np

from .qubo import QuboProblem
from .routes import RouteContext


def _try_insert(ctx: RouteContext, route: list[int], node: int) -> list[int] | None:
    best = None
    for pos in range(len(route) + 1):
        trial = route[:pos] + [node] + route[pos:]
        if ctx.route_minutes(trial) <= ctx.max_minutes + 1e-6:
            if best is None or ctx.route_minutes(trial) < ctx.route_minutes(best):
                best = trial
    return best


def decode_and_repair(qp: QuboProblem, ctx: RouteContext, bits: np.ndarray) -> dict:
    sel = [r for r, b in enumerate(bits) if int(b) == 1]
    routes = [list(qp.columns[r]["nodes"]) for r in sel]

    seen = set()
    for i, route in enumerate(routes):
        cleaned = []
        for n in route:
            if n in seen:
                continue
            seen.add(n)
            cleaned.append(n)
        routes[i] = cleaned

    freed_nodes_unplaced: list[int] = []
    while len(routes) < qp.K:
        routes.append([])
    if len(routes) > qp.K:
        order = sorted(range(len(routes)),
                        key=lambda i: sum(ctx.utility[n] for n in routes[i]))
        drop = order[: len(routes) - qp.K]
        freed_nodes = [n for i in drop for n in routes[i]]
        routes = [r for i, r in enumerate(routes) if i not in drop]
        for n in freed_nodes:
            seen.discard(n)
            for i, route in enumerate(routes):
                trial = _try_insert(ctx, route, n)
                if trial is not None:
                    routes[i] = trial
                    seen.add(n)
                    break
            else:
                freed_nodes_unplaced.append(n)

    covered = set(n for r in routes for n in r)
    blocked_mandatory = []
    dropped_by_repair: list[int] = list(freed_nodes_unplaced)
    for m in qp.mandatory_nodes:
        if m in covered:
            continue
        placed = False

        candidates = []
        for i, route in enumerate(routes):
            trial = _try_insert(ctx, route, m)
            if trial is not None:
                candidates.append((ctx.route_minutes(trial) - ctx.route_minutes(route), i, trial))
        if candidates:
            candidates.sort()
            _, i, trial = candidates[0]
            routes[i] = trial
            covered.add(m)
            placed = True
        if not placed:
            for i, route in enumerate(routes):
                optional_in_route = sorted(
                    [n for n in route if n not in qp.mandatory_nodes],
                    key=lambda n: ctx.utility[n],
                )
                reduced = list(route)
                dropped = []
                for drop_node in optional_in_route:
                    reduced = [n for n in reduced if n != drop_node]
                    dropped.append(drop_node)
                    trial = _try_insert(ctx, reduced, m)
                    if trial is not None:
                        lost = sum(ctx.utility[n] for n in dropped)
                        if best_option is None or lost < best_option[0]:
                            best_option = (lost, i, trial, list(dropped))
                        break
            if best_option is not None:
                _, i, trial, dropped = best_option
                routes[i] = trial
                covered.add(m)
                for d in dropped:
                    seen.discard(d)
                    dropped_by_repair.append(d)
                placed = True
        if not placed:
            blocked_mandatory.append(m)

    covered = set(n for r in routes for n in r)
    leftovers = sorted((n for n in dropped_by_repair if n not in covered),
                       key=lambda n: -ctx.utility[n])
    for n in leftovers:
        best_i, best_trial, best_dt = None, None, None
        for i, route in enumerate(routes):
            trial = _try_insert(ctx, route, n)
            if trial is not None:
                dt = ctx.route_minutes(trial) - ctx.route_minutes(route)
                if best_dt is None or dt < best_dt:
                    best_i, best_trial, best_dt = i, trial, dt
        if best_trial is not None:
            routes[best_i] = best_trial

    total_minutes = [ctx.route_minutes(r) for r in routes]
    total_utility = float(sum(ctx.utility[n] for r in routes for n in r))
    all_visited = [n for r in routes for n in r]
    within_time = all(m <= ctx.max_minutes + 1e-6 for m in total_minutes)
    no_duplicates = len(all_visited) == len(set(all_visited))
    return {
        "routes": routes,
        "route_minutes": total_minutes,
        "total_utility": total_utility,
        "blocked_mandatory": blocked_mandatory,
        "feasible": len(blocked_mandatory) == 0 and within_time and no_duplicates and len(routes) == qp.K,
    }
