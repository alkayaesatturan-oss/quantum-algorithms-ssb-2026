# BNAO — ATM Cash Flow Optimization

*A hybrid quantum–classical approach to daily ATM cash replenishment planning.*
Developed for the SSB Quantum Algorithms Competition (2026).

The system determines, on a daily basis, **which ATMs are to be visited**, **which
vehicle visits them and in which order**, and **how much cash is to be loaded into
each ATM**.

Route selection is formulated as a **QUBO** problem and solved with **QAOA**;
route generation, repair and order-amount computation are carried out in the
classical layer.

---

## Method

```
Data loading ──► Preprocessing ──► Utility score ──► Route pool (classical)
                                                           │
                                                           ▼
Output files ◄── Order amounts ◄── Repair ◄── QUBO + QAOA (quantum)
```

1. **Preprocessing** — A case code is determined for each ATM, priorities and
   weights are assigned, and the candidate pool is filtered. ATMs with priority
   1–2 are treated as *mandatory* visits.
2. **Utility score** — Each candidate receives a utility score based on its
   priority and the associated metric. Priority bands are constructed so that
   they provably do not overlap.
3. **Route pool** — Feasible routes that depart from and return to the depot
   (VEZNE) and do not exceed 300 minutes are generated (Haversine distance,
   35 km/h average speed, OR-Tools + local search).
4. **QUBO / QAOA** — The question *"which routes should be selected?"* is
   encoded as a set-packing QUBO with one qubit per route. The qubit count
   therefore depends on the size of the route pool rather than on the number
   of ATMs, which allows the problem to fit on current quantum hardware.
5. **Repair** — The bitstring returned by the solver is converted into a
   strictly feasible set of routes. Mandatory ATMs that cannot be accommodated
   are reported explicitly rather than silently skipped.
6. **Order amounts** — For each visited ATM, the loading amount and the
   cassette action (new cassette / idle-cash removal) are computed.

### QUBO formulation

With $y_r \in \{0,1\}$ indicating whether route column $r$ is selected and $K$
denoting the number of vehicles:

$$
H = -\sum_r u_r\, y_r
  + A\Big(\sum_r y_r - K\Big)^2
  + B \sum_{(r,s)\ \text{overlapping}} y_r\, y_s
  + C \sum_{m \in \mathcal{M}} \Big(\sum_{r \ni m} y_r - 1\Big)^2
$$

where $u_r$ is the normalized utility of the optional ATMs on route $r$ and
$\mathcal{M}$ is the set of mandatory ATMs. The terms enforce, respectively,
that exactly $K$ routes are selected, that each ATM appears in at most one
route, and that each mandatory ATM appears in exactly one route. The penalty
coefficients are set to $A = B = C = 4\left(\sum_r |u_r| + 1\right) + R$, an
upper bound that makes every constraint violation strictly more costly than
any attainable objective gain; they are therefore provably dominant rather
than merely "large".

---

## Solvers

| Solver | Purpose |
|---|---|
| `solve_exact_bruteforce` | Exact optimum for small instances (reference for the optimality gap) |
| `simulated_annealing` | Classical heuristic baseline |
| `numpy_qaoa` | Statevector QAOA simulation; requires no quantum SDK |
| `qiskit_qaoa` | Shot-based QAOA on Qiskit Aer with parameter transfer |
| `qiskit_qaoa_xy` | Constraint-preserving QAOA with an XY mixer (hardware-oriented) |
| IBM Runtime | Live execution on an IBM quantum device (optional) |

---

## Results

Representative results from a single run on the competition practice
instance (3 vehicles, qubit budget 19):

| Metric | Value |
|---|---|
| Qubits (16 route columns + 3 empty-route variables) | 19 |
| Feasible states / total states | 8 / 524,288 |
| QAOA (p = 2, statevector) — gap to exact optimum | **0.0** (optimum recovered) |
| Simulated annealing — gap to exact optimum | 0.0082 |
| QAOA feasible probability mass | 1.09 % (uniform: 0.0015 %, ≈ ×711) |

**IBM hardware** (`ibm_pittsburgh`, XY-mixer QAOA with p = 1, 4,000 shots,
dynamical decoupling + gate twirling):

| Metric | Value |
|---|---|
| Feasible shot ratio | 15.2 % |
| Shots in the correct Hamming-weight sector | 46.1 % |
| Shots improving on the warm-start state | 14.9 % |
| P(optimum) | 4.58 % |
| Transpiled depth / CZ gates | 233 / 143 |

Because OR-Tools runs under a time limit, the route pool — and hence the
numerical values above — may vary slightly from run to run.

---

## Project Structure

```
ssb-2026-atm-cash-optimization/
├── main.py            # Command-line entry point
├── run.ipynb          # End-to-end walkthrough and analysis notebook
├── requirements.txt
├── .env.example       # Template for IBM Quantum credentials
└── src/
    ├── pipeline.py    # Orchestration (run_pipeline)
    ├── config.py      # All parameters in one place
    ├── io.py          # Input loading and schema validation
    ├── preprocess.py  # Case codes, candidate pool, idle-cash removal
    ├── utility.py     # utility_score computation
    ├── demand.py      # Demand model (net need)
    ├── distance.py    # Haversine travel-time matrix
    ├── routes.py      # Route pool generation (classical layer)
    ├── qubo.py        # Set-packing QUBO formulation
    ├── solvers.py     # Exact, SA, QAOA (NumPy / Qiskit / CUDA-Q)
    ├── repair.py      # Repair of bitstrings into feasible routes
    ├── orders.py      # Order amounts and cassette actions
    ├── outputs.py     # Generation of output files
    └── ibm_account.py # IBM Quantum credentials (.env)
```

References in the source code to "Section x.y" and "Appendix A/B" refer to the
competition specification.

---

## Data Availability

The competition dataset and specification documents were provided by the
organizers and **are not redistributed in this repository**. To run the code,
place the following pipe-delimited (`|`) UTF-8 CSV files in a directory
(e.g. `dataset/`):

- `CURRENT_INVENTORY.csv` — ATM inventory, balances, malfunction codes, coordinates
- `BANKAMATIK_DEMAND.csv` — daily cash withdrawal / deposit demand
- `PRIO_LOGIC.csv` — case codes, priorities and ranking rules

The required columns are validated in `src/io.py`.

**Outputs** (written to `output/`):

- `TODAYS_ORDERS.csv` — visiting order, cassette action and loading amount per vehicle
- `CURRENT_INVENTORY.csv` — enriched inventory (utility_score, route information)
- `ROUTE_SUMMARY.csv` — stops, duration and total utility per vehicle
- `solver_report.json` — qubit count, solver comparison and diagnostics

No file path, date or location name is hard-coded; everything is read from the
supplied directory.

---

## Installation and Usage

```bash
pip install -r requirements.txt
```

`cudaq` is optional; if it is not installed, the corresponding solver is skipped.

**Command line (recommended):**

```bash
python main.py --data-dir dataset --vehicles 3
python main.py --data-dir /path/to/new_data --vehicles 4 --output-dir results
python main.py --data-dir dataset --vehicles 3 --task-date 2026-09-06 --backend numpy
```

| Argument | Default | Description |
|---|---|---|
| `--data-dir` | required | Directory containing the input CSV files |
| `--vehicles` | required | Number of vehicles |
| `--output-dir` | `output` | Output directory |
| `--task-date` | first demand date | Task date (YYYY-MM-DD) |
| `--backend` | `auto` | `auto`, `numpy`, `qiskit_aer`, `cudaq` |
| `--max-qubits` | `19` | Qubit budget |
| `--pool-size` | `16` | Upper bound on the route pool size |
| `--qaoa-layers` | `2` | Number of QAOA layers |
| `--shots` | `4000` | Number of measurement shots |
| `--seed` | `0` | Random seed |
| `--quiet` | off | Suppress the console summary |

Run `python main.py --help` for the full list of arguments. Exit code: `1` if a
mandatory case was missed, `0` otherwise.

**Notebook:**

```bash
jupyter notebook run.ipynb
```

The values of `DATA_DIR`, `NUM_VEHICLES` and `MAX_QUBITS` at the top of the
notebook can be adjusted.

**Python API:**

```python
from src.pipeline import run_pipeline

result = run_pipeline("dataset", num_vehicles=3, output_dir="output")
```

The `quantum_backend` parameter accepts `"auto"`, `"qiskit_aer"`, `"numpy"` or `"cudaq"`.

---

## IBM Quantum (Optional)

To run on real hardware, copy `.env.example` to `.env` in the project root and
fill in your credentials:

```
IBM_QUANTUM_TOKEN=...
IBM_QUANTUM_CHANNEL=ibm_quantum_platform
IBM_QUANTUM_INSTANCE=...
```

Setting `RUN_ON_IBM = True` in the notebook starts a live run on the selected
device. This step consumes quota and is therefore disabled by default. The
`.env` file is excluded from version control.

---

## Design Notes

- **Route-column encoding:** qubit count = pool size + number of vehicles. The
  pool is capped at a fixed size so that the qubit budget remains identical
  across runs.
- **Provably dominant penalties:** the QUBO penalty coefficients are chosen
  such that every constraint violation is strictly more costly than every
  possible objective gain.
- **Safety net:** the repair layer is always executed; the quantum proposal is
  preserved and modified only as much as necessary.
- **No capacity constraint:** the transportable amount is assumed to be
  unlimited; only the 300-minute duration constraint is enforced.
