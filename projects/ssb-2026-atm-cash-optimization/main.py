#!/usr/bin/env python3
"""
BNAO -- ATM Cash Flow Optimization: command-line entry point.

Submission requirement (Section 2): the input files are read directly, and
the candidate ATMs, utility scores, routes and order amounts are all
computed in code. The file paths, the number of vehicles and the task date
are not hard-coded; all of them are supplied on the command line.

Usage:
    python main.py --data-dir dataset --vehicles 3
    python main.py --data-dir /path/to/new_data --vehicles 4 --output-dir results
    python main.py --data-dir dataset --vehicles 3 --task-date 2026-09-06 --backend numpy

Generated files (Section 3, '|'-delimited UTF-8):
    <output-dir>/TODAYS_ORDERS.csv      orders per vehicle, in visiting order
    <output-dir>/CURRENT_INVENTORY.csv  input + utility_score + route durations + priority coverage
    <output-dir>/ROUTE_SUMMARY.csv      stops / duration / total utility per vehicle (supplementary)
    <output-dir>/solver_report.json     qubit count, solver comparison, diagnostics
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config          # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402


def _json_safe(obj):
    """Converts numpy/pandas types into JSON-serializable equivalents."""
    import numpy as np
    import pandas as pd
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BNAO: candidate selection + quantum (QAOA) route optimization + order amount computation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", required=True,
                   help="Directory containing the CURRENT_INVENTORY / BANKAMATIK_DEMAND / PRIO_LOGIC csv files")
    p.add_argument("--vehicles", type=int, required=True,
                   help="Number of vehicles (announced in Appendix A)")
    p.add_argument("--output-dir", default="output",
                   help="Directory to which the output files are written")
    p.add_argument("--task-date", default=None,
                   help="Task date (YYYY-MM-DD). Defaults to the first date in BANKAMATIK_DEMAND")
    p.add_argument("--backend", default="auto",
                   choices=["auto", "numpy", "qiskit_aer", "cudaq"],
                   help="Backend for the quantum step ('auto': qiskit_aer if available, otherwise numpy statevector)")
    p.add_argument("--max-qubits", type=int, default=19,
                   help="Qubit budget; the route pool is capped according to this budget")
    p.add_argument("--pool-size", type=int, default=16,
                   help="Upper bound on the number of routes (columns) in the pool")
    p.add_argument("--qaoa-layers", type=int, default=2, help="Number of QAOA layers (p)")
    p.add_argument("--qaoa-maxiter", type=int, default=120, help="Classical optimizer iterations")
    p.add_argument("--qaoa-restarts", type=int, default=3, help="Number of restarts from different initial points")
    p.add_argument("--cvar-alpha", type=float, default=0.02, help="Alpha value of the CVaR objective")
    p.add_argument("--shots", type=int, default=4000, help="Number of measurement shots")
    p.add_argument("--seed", type=int, default=0, help="Random seed (reproducibility)")
    p.add_argument("--quiet", action="store_true", help="Suppress the console summary")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not os.path.isdir(args.data_dir):
        print(f"[error] data directory not found: {args.data_dir}", file=sys.stderr)
        return 2
    if args.vehicles < 1:
        print("[error] --vehicles must be at least 1", file=sys.stderr)
        return 2

    cfg = Config()
    result = run_pipeline(
        data_dir=args.data_dir,
        num_vehicles=args.vehicles,
        output_dir=args.output_dir,
        cfg=cfg,
        pool_size=args.pool_size,
        max_qubits=args.max_qubits,
        quantum_backend=args.backend,
        qaoa_layers=args.qaoa_layers,
        qaoa_maxiter=args.qaoa_maxiter,
        qaoa_restarts=args.qaoa_restarts,
        cvar_alpha=args.cvar_alpha,
        shots=args.shots,
        seed=args.seed,
        task_date=args.task_date,
        verbose=not args.quiet,
    )

    report = {
        "task_date": result.task_date.date().isoformat(),
        "num_vehicles": result.num_vehicles,
        "data_dir": os.path.abspath(args.data_dir),
        "missed_prio1": result.missed_prio1,
        "missed_prio2": result.missed_prio2,
        "solver_report": result.solver_report,
        "diagnostics": result.diagnostics,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "solver_report.json"), "w", encoding="utf-8") as f:
        json.dump(_json_safe(report), f, ensure_ascii=False, indent=1)

    if not args.quiet:
        print()
        print(result.route_summary.to_string(index=False))
        print()
        print(f"[bnao] outputs written -> {os.path.abspath(args.output_dir)}/"
              f"{{TODAYS_ORDERS.csv, CURRENT_INVENTORY.csv, ROUTE_SUMMARY.csv, solver_report.json}}")
        if result.missed_prio1 or result.missed_prio2:
            print(f"[bnao] WARNING: missed mandatory cases -- prio1: {result.missed_prio1} | prio2: {result.missed_prio2}")

    # Return 1 if a mandatory (priority 1-2) case was missed, otherwise 0.
    return 1 if (result.missed_prio1 or result.missed_prio2) else 0


if __name__ == "__main__":
    sys.exit(main())
