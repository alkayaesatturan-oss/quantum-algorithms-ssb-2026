"""
End-to-end orchestration: data loading -> preprocessing -> utility_score ->
route pool -> QUBO -> quantum/classical solution -> repair -> order amount ->
output files.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import pandas as pd

from .config import Config
from .io import load_raw_data
from .demand import DemandModel
from .preprocess import build_candidate_pool, build_prio_lookup
from .utility import compute_utility_scores, assert_priority_band_separation
from .routes import build_context, structurally_unreachable, generate_route_pool
from .qubo import build_route_qubo_full
from .solvers import solve_exact_bruteforce, simulated_annealing, numpy_qaoa, cudaq_qaoa, qiskit_qaoa


def _qiskit_aer_available() -> bool:
    try:
        import qiskit_aer  # noqa: F401
        return True
    except Exception:
        return False
from .repair import decode_and_repair
from .outputs import build_todays_orders, build_enriched_inventory, build_route_summary


@dataclass
class RunResult:
    todays_orders: pd.DataFrame
    current_inventory: pd.DataFrame
    route_summary: pd.DataFrame
    solver_report: dict
    missed_prio1: list
    missed_prio2: list
    task_date: pd.Timestamp
    num_vehicles: int
    diagnostics: dict = field(default_factory=dict)


def run_pipeline(data_dir: str, num_vehicles: int, output_dir: str | None = None,
                  cfg: Config | None = None, pool_size: int = 16, max_qubits: int = 19,
                  quantum_backend: str = "auto", qaoa_layers: int = 2,
                  qaoa_maxiter: int = 120, qaoa_restarts: int = 3, cvar_alpha: float = 0.02,
                  shots: int = 4000, aer_refine_iters: int = 40,
                  seed: int = 0, task_date=None, verbose: bool = True) -> RunResult:
    cfg = cfg or Config()
    t_start = time.time()

    raw = load_raw_data(data_dir)
    demand = DemandModel(raw.demand)
    if task_date is None:
        task_date = demand.min_date
    task_date = pd.Timestamp(task_date)

    prio_lookup = build_prio_lookup(raw.prio)
    madde_by_code = prio_lookup["madde"].to_dict()

    cand = build_candidate_pool(raw.inventory, raw.prio, demand, task_date, cfg)
    scored = compute_utility_scores(cand)
    assert_priority_band_separation(scored)

    depot_rows = raw.inventory[raw.inventory["type"].astype(str) == "VEZNE"]
    if depot_rows.empty:
        raise ValueError("No depot row with type == 'VEZNE' was found in CURRENT_INVENTORY.")
    depot = depot_rows.iloc[0]

    candidates_only = scored[scored["is_candidate"]].reset_index(drop=True)
    locationno_by_idx = {i: loc for i, loc in enumerate(candidates_only["locationno"])}

    ctx = build_context(candidates_only, float(depot["latitude"]), float(depot["longitude"]), cfg)
    unreachable_idx = structurally_unreachable(ctx)
    mandatory_idx_all = [i for i, m in enumerate(ctx.mandatory_mask) if m]
    mandatory_idx_reachable = [i for i in mandatory_idx_all if i not in unreachable_idx]

    effective_pool = min(pool_size, max_qubits - num_vehicles)
    if effective_pool < num_vehicles:
        raise ValueError(
            f"With max_qubits={max_qubits} and {num_vehicles} vehicles, the pool shrinks to {effective_pool} columns; "
            f"at least {num_vehicles} columns are required. Increase max_qubits."
        )
    pool, _ = generate_route_pool(ctx, num_vehicles, pool_size=effective_pool, seed=seed)
    node_utility = {i: float(ctx.utility[i]) for i in range(len(ctx.utility))}

    qp = build_route_qubo_full(pool, num_vehicles, mandatory_idx_reachable, node_utility)
    R = qp.Q.shape[0]
    if R > max_qubits:
        raise RuntimeError(f"The QUBO produced {R} qubits, but the budget is {max_qubits}. (Pool capping failed -- please report this.)")

    solver_report = {"qubit_count": R, "pool_size_used": len(pool), "max_qubits": max_qubits}

    exact = solve_exact_bruteforce(qp, max_R=max_qubits)
    solver_report["exact"] = exact

    sa = simulated_annealing(qp, iters=10_000, restarts=8, seed=seed)
    solver_report["simulated_annealing"] = sa

    if quantum_backend == "auto":
        quantum_backend = "qiskit_aer" if _qiskit_aer_available() else "numpy"
    reference = numpy_qaoa(qp, layer_count=qaoa_layers, max_R=max_qubits,
                           maxiter=qaoa_maxiter, n_restarts=qaoa_restarts,
                           cvar_alpha=cvar_alpha, init="grid", seed=seed)
    if reference.get("skipped"):
        raise RuntimeError(f"The quantum reference step did not run: {reference.get('reason')}")
    solver_report["quantum_reference_numpy"] = reference

    if quantum_backend == "qiskit_aer":
        quantum_result = qiskit_qaoa(qp, layer_count=qaoa_layers, theta0=reference["theta"],
                                     shots=shots, maxiter=aer_refine_iters, refine=aer_refine_iters > 0,
                                     cvar_alpha=cvar_alpha, seed=seed)
        solver_report["quantum_backend_used"] = "qiskit_aer"
    elif quantum_backend == "cudaq":
        quantum_result = cudaq_qaoa(qp, layer_count=qaoa_layers)
        if quantum_result is None:
            raise RuntimeError("quantum_backend='cudaq' was requested, but the 'cudaq' package is not installed.")
        solver_report["quantum_backend_used"] = "cudaq"
    elif quantum_backend == "numpy":
        quantum_result = reference
        solver_report["quantum_backend_used"] = "numpy_statevector_qaoa"
    else:
        raise ValueError(f"Bilinmeyen quantum_backend: {quantum_backend}")
    if quantum_result.get("skipped"):
        raise RuntimeError(f"The quantum step did not run: {quantum_result.get('reason')}")
    solver_report["quantum"] = quantum_result

    q_src = quantum_result.get("feasible_source", "none")
    q_label = {"numpy_statevector_qaoa": "numpy_qaoa", "qiskit_aer": "qiskit_aer_qaoa",
               "cudaq": "cudaq_qaoa"}[solver_report["quantum_backend_used"]]
    if q_src == "exhaustive_fallback":
        q_label += "_exhaustive_fallback"
    ordered = []
    if quantum_result.get("bits") is not None and q_src == "qaoa_samples":
        ordered.append((q_label, quantum_result))
    if not exact.get("skipped") and exact.get("bits") is not None:
        ordered.append(("exact_bruteforce", exact))
    if sa.get("bits") is not None:
        ordered.append(("simulated_annealing", sa))
    if quantum_result.get("bits") is not None and q_src != "qaoa_samples":
        ordered.append((q_label, quantum_result))
    feasible_ordered = [(lbl, r) for lbl, r in ordered if r.get("feasible")]
    if feasible_ordered:
        label, best = min(feasible_ordered, key=lambda lr: lr[1]["energy"])   # min is stable: on ties, the first one wins
        chosen_bits = best["bits"]
    elif ordered:
        label, best = min(ordered, key=lambda lr: lr[1]["energy"])
        chosen_bits = best["bits"]
        label += "_infeasible_repaired"
    else:
        chosen_bits = [0] * R
        label = "none_feasible_fallback_empty"
    solver_report["chosen_solution_source"] = label
    solver_report["quantum_feasible_source"] = q_src
    solver_report["quantum_energy_gap_vs_exact"] = (
        None if exact.get("skipped") or quantum_result.get("energy") is None
        else float(quantum_result["energy"] - exact["energy"])
    )

    repaired = decode_and_repair(qp, ctx, chosen_bits)
    routes = repaired["routes"]
    route_minutes = repaired["route_minutes"]

    covered_idx = set(n for r in routes for n in r)
    missed_idx = [i for i in mandatory_idx_all if i not in covered_idx]
    missed_locs = [locationno_by_idx[i] for i in missed_idx]
    missed_prio1 = [locationno_by_idx[i] for i in missed_idx if scored_priority(candidates_only, i) == 1]
    missed_prio2 = [locationno_by_idx[i] for i in missed_idx if scored_priority(candidates_only, i) == 2]

    todays_orders = build_todays_orders(candidates_only, routes, locationno_by_idx, cfg, madde_by_code)
    current_inventory = build_enriched_inventory(
        raw.inventory, scored, routes, locationno_by_idx, route_minutes,
        missed_locs, missed_prio1, missed_prio2,
    )
    route_summary = build_route_summary(routes, route_minutes, candidates_only, locationno_by_idx)

    diagnostics = {
        "runtime_s": time.time() - t_start,
        "num_candidates": int(scored["is_candidate"].sum()),
        "num_mandatory_total": len(mandatory_idx_all),
        "num_structurally_unreachable_mandatory": len(set(mandatory_idx_all) & set(unreachable_idx)),
        "num_visited": len(covered_idx),
        "repair": repaired,
        "unreachable_locationnos": [locationno_by_idx[i] for i in unreachable_idx],
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        todays_orders.to_csv(os.path.join(output_dir, "TODAYS_ORDERS.csv"), sep="|", index=False)
        current_inventory.to_csv(os.path.join(output_dir, "CURRENT_INVENTORY.csv"), sep="|", index=False)
        route_summary.to_csv(os.path.join(output_dir, "ROUTE_SUMMARY.csv"), sep="|", index=False)

    if verbose:
        q = quantum_result
        print(f"[bnao] task date: {task_date.date()}  |  vehicles: {num_vehicles}  |  candidates: {diagnostics['num_candidates']}  |  "
              f"visited: {diagnostics['num_visited']}  |  missed prio1/2: {len(missed_prio1)}/{len(missed_prio2)}")
        print(f"[bnao] qubits: {R}  |  feasible states: {q.get('num_feasible_states')}/{q.get('num_states')}  |  "
              f"QAOA feasible mass: {q.get('feasible_mass', 0):.3%} (uniform {q.get('uniform_feasible_mass', 0):.3%})  |  "
              f"optimum rank: {q.get('optimum_rank')}  |  source: {q_src}")
        if solver_report["quantum_backend_used"] == "qiskit_aer":
            ref = reference
            print(f"[bnao] Aer ({q['final_shots']} shots): feasible shot ratio {q['feasible_mass']:.3%}  |  "
                  f"reference (statevector) feasible mass {ref['feasible_mass']:.3%}  |  "
                  f"transpiled depth {q['transpiled']['depth']}, 2-qubit gates {q['transpiled']['two_qubit_gates']}")
        print(f"[bnao] selected solver: {label}  |  QAOA-exact gap: {solver_report['quantum_energy_gap_vs_exact']}  |  "
              f"quantum time: {q.get('runtime_s', 0):.1f}s (+reference {reference['runtime_s']:.1f}s)  |  total: {diagnostics['runtime_s']:.1f}s")

    return RunResult(
        todays_orders=todays_orders, current_inventory=current_inventory,
        route_summary=route_summary, solver_report=solver_report,
        missed_prio1=missed_prio1, missed_prio2=missed_prio2,
        task_date=task_date, num_vehicles=num_vehicles, diagnostics=diagnostics,
    )


def scored_priority(candidates_only: pd.DataFrame, idx: int) -> float:
    return float(candidates_only.iloc[idx]["priority"])
