"""
Engines that solve the QUBO:

  1. `solve_exact_bruteforce` -- exact (brute-force) optimum for small R.
     Provides the "ground truth" used to measure how far the quantum and
     classical-heuristic solutions are from the optimum (optimality gap).
  2. `simulated_annealing`    -- classical heuristic baseline for comparison.
  3. `numpy_qaoa`             -- full statevector QAOA simulation
     (NumPy/SciPy). It requires no quantum SDK and serves as the local
     reference/validation engine. It EXPLICITLY reports the feasible mass,
     the probability rank of the optimum, and whether "the answer came from
     the QAOA samples or from an exhaustive scan".
  4. `cudaq_qaoa`             -- gate-level QAOA with NVIDIA CUDA-Q
     (optional; returns None if `cudaq` is not installed). Since the team's
     target hardware is IBM, this is NOT the primary path; the Qiskit Aer /
     IBM Runtime path is provided as `qiskit_qaoa` (phase 2).

In addition, `to_ising` and `qiskit_resource_report` convert the QUBO into
an Ising Hamiltonian (ZZ + Z terms) and report resource metrics such as
circuit depth, gate count and qubit count using the Qiskit transpiler
(Evaluation Rubric: "Quantum Resource Efficiency").
"""
from __future__ import annotations

import time

import numpy as np

from .qubo import QuboProblem, energy as qubo_energy, decode_value


def _all_bits(R: int) -> np.ndarray:
    idx = np.arange(1 << R, dtype=np.int64)
    return ((idx[:, None] >> np.arange(R)[None, :]) & 1).astype(np.int8)


def _all_energies(qp: QuboProblem) -> np.ndarray:
    bits = _all_bits(qp.Q.shape[0]).astype(np.float64)
    return np.einsum('ij,jk,ik->i', bits, qp.Q, bits) + qp.offset


def feasible_mask_all(qp: QuboProblem) -> np.ndarray:
    R = qp.Q.shape[0]
    bits = _all_bits(R)
    nodes = sorted({x for c in qp.columns for x in c["nodes"]})
    node_idx = {v: i for i, v in enumerate(nodes)}
    if any(m not in node_idx for m in qp.mandatory_nodes):
        return np.zeros(1 << R, dtype=bool)  # a mandatory node absent from every column -> no selection is feasible
    M = np.zeros((R, len(nodes)), dtype=np.int16)
    for r, c in enumerate(qp.columns):
        for x in c["nodes"]:
            M[r, node_idx[x]] = 1
    counts = bits.astype(np.int16) @ M                     # n x |nodes|
    card_ok = bits.sum(axis=1, dtype=np.int32) == qp.K
    no_conflict = (counts <= 1).all(axis=1)
    mand_cols = [node_idx[m] for m in qp.mandatory_nodes]
    mand_ok = (counts[:, mand_cols] == 1).all(axis=1) if mand_cols else np.ones(1 << R, dtype=bool)
    return card_ok & no_conflict & mand_ok


def solve_exact_bruteforce(qp: QuboProblem, max_R: int = 20) -> dict:
    R = qp.Q.shape[0]
    if R > max_R:
        return {"skipped": True, "reason": f"R={R} > max_R={max_R}, brute force skipped"}

    t0 = time.time()
    energies = _all_energies(qp)
    best = int(np.argmin(energies))
    best_bits = _all_bits(R)[best].astype(int)
    info = decode_value(qp, best_bits)
    return {
        "skipped": False, "bits": best_bits, "energy": float(energies[best]),
        "runtime_s": time.time() - t0, **info,
        "energy_distribution_min_max": (float(energies.min()), float(energies.max())),
    }


def simulated_annealing(qp: QuboProblem, iters: int = 20_000, restarts: int = 8,
                         seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    R = qp.Q.shape[0]

    best_overall_bits, best_overall_e = None, np.inf
    t0 = time.time()
    for _ in range(restarts):
        bits = rng.integers(0, 2, size=R).astype(float)
        e = qubo_energy(qp, bits)
        T0, T1 = 3.0, 0.01
        for it in range(iters):
            T = T0 * (T1 / T0) ** (it / max(1, iters - 1))
            k = rng.integers(0, R)
            new = 1.0 - bits[k]
            trial = bits.copy(); trial[k] = new
            e_trial = qubo_energy(qp, trial)
            d = e_trial - e
            if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
                bits, e = trial, e_trial
        if e < best_overall_e:
            best_overall_bits, best_overall_e = bits.copy(), e
    info = decode_value(qp, best_overall_bits.astype(int))
    return {"bits": best_overall_bits.astype(int), "energy": float(best_overall_e),
            "runtime_s": time.time() - t0, **info}


def _rx_tensor(beta: float, k: int) -> np.ndarray:
    c, s = np.cos(beta), np.sin(beta)
    g = np.array([[c, -1j * s], [-1j * s, c]])
    G = np.array([[1.0 + 0j]])
    for _ in range(k):
        G = np.kron(G, g)
    return G


def _apply_rx_all_qubits(psi: np.ndarray, beta: float, R: int, chunk: int = 5) -> np.ndarray:
    q0 = 0
    while q0 < R:
        k = min(chunk, R - q0)
        G = _rx_tensor(beta, k)
        hi, mid, lo = 1 << (R - q0 - k), 1 << k, 1 << q0
        v = psi.reshape(hi, mid, lo)
        if lo == 1:
            psi = (v.reshape(hi, mid) @ G.T).reshape(-1)
        elif hi == 1:
            psi = (G @ v.reshape(mid, lo)).reshape(-1)
        else:
            psi = np.einsum('ab,hbl->hal', G, v, optimize=True).reshape(-1)
        q0 += k
    return psi


def numpy_qaoa(qp: QuboProblem, layer_count: int = 2, max_R: int = 20,
               cvar_alpha: float = 0.02, maxiter: int = 120, seed: int = 0,
               n_restarts: int = 3, top_k_samples: int = 4096,
               normalize_energy: bool = True, init: str = "grid",
               grid_gamma: int = 24, grid_beta: int = 12) -> dict:
    from scipy.optimize import minimize

    R = qp.Q.shape[0]
    if R > max_R:
        return {"skipped": True, "reason": f"R={R} > max_R={max_R}, statevector simulation skipped"}

    t0 = time.time()
    energies = _all_energies(qp)
    n_states = 1 << R
    order_e = np.argsort(energies, kind="stable")
    e_sorted = energies[order_e]
    feas = feasible_mask_all(qp)
    num_feasible = int(feas.sum())
    rng = np.random.default_rng(seed)
    energy_scale = float(max(qp.A, 1e-9)) if normalize_energy else 1.0
    phase_energies = energies / energy_scale

    def run_circuit(theta):
        gammas, betas = theta[:layer_count], theta[layer_count:]
        psi = np.full(n_states, 1.0 / np.sqrt(n_states), dtype=complex)
        for l in range(layer_count):
            psi = psi * np.exp(-1j * gammas[l] * phase_energies)
            psi = _apply_rx_all_qubits(psi, betas[l], R)
        return psi

    def cvar_cost(theta):
        psi = run_circuit(theta)
        probs = np.abs(psi) ** 2
        ps = probs[order_e]
        cum = np.cumsum(ps)
        cutoff = int(np.searchsorted(cum, cvar_alpha * cum[-1])) + 1
        w = ps[:cutoff]
        return float(np.sum(w * e_sorted[:cutoff]) / max(w.sum(), 1e-12))

    best_theta, best_cost, n_evals = None, np.inf, 0
    grid_info = None
    starts: list[np.ndarray] = []
    if init == "grid":
        gammas = np.linspace(0.0, 2 * np.pi, grid_gamma, endpoint=False)
        betas = np.linspace(0.0, np.pi, grid_beta, endpoint=False)
        psi0 = np.full(n_states, 1.0 / np.sqrt(n_states), dtype=complex)
        grid_costs = np.full((grid_gamma, grid_beta), np.inf)
        for gi, g in enumerate(gammas):
            phased = psi0 * np.exp(-1j * g * phase_energies)
            for bi, b in enumerate(betas):
                psi_gb = _apply_rx_all_qubits(phased.copy(), b, R)
                probs = np.abs(psi_gb) ** 2
                ps = probs[order_e]
                cum = np.cumsum(ps)
                cutoff = int(np.searchsorted(cum, cvar_alpha * cum[-1])) + 1
                w = ps[:cutoff]
                grid_costs[gi, bi] = float(np.sum(w * e_sorted[:cutoff]) / max(w.sum(), 1e-12))
                n_evals += 1
        flat = np.argsort(grid_costs, axis=None, kind="stable")
        picks: list[tuple[float, float, float]] = []
        for f in flat:
            gi, bi = divmod(int(f), grid_beta)
            g, b = float(gammas[gi]), float(betas[bi])
            far = all(min(abs(g - pg), 2 * np.pi - abs(g - pg)) > 0.5 or abs(b - pb) > 0.4
                      for pg, pb, _ in picks)
            if far:
                picks.append((g, b, float(grid_costs[gi, bi])))
            if len(picks) >= max(1, n_restarts):
                break
        g_best, b_best, c_best = picks[0]
        grid_info = {"gamma1": g_best, "beta1": b_best, "cvar_p1": c_best,
                     "grid_points": int(grid_gamma * grid_beta),
                     "starts": [(g, b) for g, b, _ in picks]}
        L = layer_count
        for g, b, _ in picks:
            starts.append(np.concatenate([np.full(L, g), np.full(L, b)]))
        while len(starts) < n_restarts:
            starts.append(starts[0] + rng.normal(0.0, 0.15, size=2 * L))
    else:
        for _ in range(n_restarts):
            starts.append(np.concatenate([rng.uniform(0.0, 2 * np.pi, size=layer_count),
                                          rng.uniform(0.0, np.pi, size=layer_count)]))

    for theta0 in starts[: max(n_restarts, 1)]:
        res = minimize(cvar_cost, theta0, method="COBYLA",
                        options={"maxiter": maxiter, "rhobeg": 0.3})
        n_evals += int(res.nfev)
        if res.fun < best_cost:
            best_cost, best_theta = res.fun, res.x

    psi = run_circuit(best_theta)
    probs = np.abs(psi) ** 2
    probs = probs / probs.sum()
    order_p = np.argsort(-probs, kind="stable")
    top = order_p[: min(top_k_samples, n_states)]
    bits_all = _all_bits(R)

    opt_idx = int(order_e[0])
    feasible_mass = float(probs[feas].sum()) if num_feasible else 0.0
    result = {
        "skipped": False, "theta": best_theta.tolist(), "cvar_cost": best_cost,
        "runtime_s": time.time() - t0, "qubit_count": R, "layer_count": layer_count,
        "optimizer_evals": n_evals, "energy_scale": energy_scale, "cvar_alpha": cvar_alpha,
        "init": init, "grid": grid_info,
        "num_feasible_states": num_feasible, "num_states": n_states,
        "feasible_mass": feasible_mass,
        "uniform_feasible_mass": num_feasible / n_states,
        "p_optimum": float(probs[opt_idx]),
        "optimum_rank": int(np.where(order_p == opt_idx)[0][0]),
        "energy_most_probable": float(energies[order_p[0]]),
        "samples": [(tuple(int(b) for b in bits_all[i]), float(energies[i]), float(probs[i])) for i in top],
    }

    feas_top = top[feas[top]]
    if len(feas_top):
        best = int(feas_top[np.argmin(energies[feas_top])])
        result["feasible_source"] = "qaoa_samples"
    elif num_feasible:
        fe = np.where(feas, energies, np.inf)
        best = int(np.argmin(fe))
        result["feasible_source"] = "exhaustive_fallback"
    else:
        best = None
        result["feasible_source"] = "none"

    if best is not None:
        b = bits_all[best].astype(int)
        result.update({"bits": b, "energy": float(energies[best]), **decode_value(qp, b)})
    else:
        result.update({"bits": None, "energy": None, "feasible": False})
    return result


def to_ising(Q: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, float]], float]:
    R = Q.shape[0]
    diag = np.diag(Q).copy()
    off = Q.copy()
    np.fill_diagonal(off, 0.0)
    off_sym = off + off.T  # let Q_ij (i<j) contribute to both the row and the column sums

    h = -diag / 2.0 - off_sym.sum(axis=1) / 4.0
    edges = []
    for i in range(R):
        for j in range(i + 1, R):
            if off[i, j] != 0.0:
                edges.append((i, j, off[i, j] / 4.0))
    const = diag.sum() / 2.0 + off[np.triu_indices(R, k=1)].sum() / 4.0
    return h, edges, const


def cudaq_qaoa(qp: QuboProblem, layer_count: int = 2, maxiter: int = 150,
               target: str = "qpp-cpu") -> dict | None:
    try:
        import cudaq
        from cudaq import spin
    except ImportError:
        return None

    field_w_arr, edges, const = to_ising(qp.Q)
    R = len(field_w_arr)
    edges_src = [e[0] for e in edges]
    edges_tgt = [e[1] for e in edges]
    edge_w = [e[2] for e in edges]
    field_w = field_w_arr.tolist()

    cudaq.set_target(target)

    @cudaq.kernel
    def qaoa_kernel(qubit_count: int, layer_count: int,
                     edges_src: list[int], edges_tgt: list[int], edge_w: list[float],
                     field_w: list[float], thetas: list[float]):
        q = cudaq.qvector(qubit_count)
        h(q)
        for layer in range(layer_count):
            gamma = thetas[layer]
            for e in range(len(edges_src)):
                u = edges_src[e]
                v = edges_tgt[e]
                x.ctrl(q[u], q[v])
                rz(2.0 * gamma * edge_w[e], q[v])
                x.ctrl(q[u], q[v])
            for i in range(qubit_count):
                rz(2.0 * gamma * field_w[i], q[i])
            beta = thetas[layer_count + layer]
            for i in range(qubit_count):
                rx(2.0 * beta, q[i])

    hamiltonian = 0.0 * spin.i(0)
    for (u, v, w) in edges:
        hamiltonian += w * spin.z(u) * spin.z(v)
    for i, w in enumerate(field_w_arr):
        hamiltonian += float(w) * spin.z(i)

    param_count = 2 * layer_count
    optimizer = cudaq.optimizers.COBYLA()
    optimizer.max_iterations = maxiter
    optimizer.initial_parameters = np.random.uniform(-np.pi / 8, np.pi / 8, param_count).tolist()

    def objective(params):
        return cudaq.observe(qaoa_kernel, hamiltonian, R, layer_count,
                              edges_src, edges_tgt, edge_w, field_w, params).expectation()

    t0 = time.time()
    optimal_expectation, optimal_params = optimizer.optimize(dimensions=param_count, function=objective)
    counts = cudaq.sample(qaoa_kernel, R, layer_count, edges_src, edges_tgt, edge_w, field_w,
                           optimal_params, shots_count=4096)
    runtime = time.time() - t0

    samples = []
    best_bits, best_e = None, np.inf
    for bitstring, count in counts.items():
        bits = np.array([int(c) for c in bitstring])
        e = qubo_energy(qp, bits)
        samples.append((tuple(bits.tolist()), float(e), int(count)))
        info = decode_value(qp, bits)
        if info["feasible"] and e < best_e:
            best_e, best_bits = e, bits

    result = {
        "skipped": False, "backend": "cudaq", "target": target,
        "qubit_count": R, "layer_count": layer_count, "runtime_s": runtime,
        "optimal_expectation": float(optimal_expectation + const),
        "samples": samples,
    }
    if best_bits is not None:
        info = decode_value(qp, best_bits)
        result.update({"bits": best_bits, "energy": float(best_e), "feasible_source": "qaoa_samples", **info})
    else:
        result.update({"bits": None, "energy": None, "feasible": False, "feasible_source": "none"})
    return result


def _runtime_sampler(backend):
    try:
        from qiskit_ibm_runtime.executor_sampler import Sampler as RuntimeSampler
    except ImportError:  # pragma: no cover
        from qiskit_ibm_runtime import SamplerV2 as RuntimeSampler
    return RuntimeSampler(mode=backend)


def build_qaoa_circuit(qp: QuboProblem, layer_count: int, normalize_energy: bool = True):
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    scale = float(max(qp.A, 1e-9)) if normalize_energy else 1.0
    h, edges, const = to_ising(qp.Q / scale)
    R = len(h)
    gammas = ParameterVector("gamma", layer_count)
    betas = ParameterVector("beta", layer_count)
    qc = QuantumCircuit(R)
    qc.h(range(R))
    for l in range(layer_count):
        for (u, v, w) in edges:
            qc.rzz(2.0 * gammas[l] * w, u, v)
        for i in range(R):
            if h[i] != 0.0:
                qc.rz(2.0 * gammas[l] * h[i], i)
        for i in range(R):
            qc.rx(2.0 * betas[l], i)
    qc.measure_all()
    return qc, list(gammas), list(betas), scale, const * scale + qp.offset


def qiskit_qaoa(qp: QuboProblem, layer_count: int = 2, theta0=None, shots: int = 4000,
                final_shots: int = 8192, maxiter: int = 40, refine: bool = True,
                cvar_alpha: float = 0.02, seed: int = 0, backend=None, sampler=None,
                optimization_level: int = 1, normalize_energy: bool = True,
                top_k_samples: int = 4096, max_R_exact: int = 22) -> dict:

    from scipy.optimize import minimize
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    t0 = time.time()
    R = qp.Q.shape[0]
    qc, gam, bet, energy_scale, _ = build_qaoa_circuit(qp, layer_count, normalize_energy)

    if backend is None:
        from qiskit_aer import AerSimulator
        backend = AerSimulator(seed_simulator=seed)
        backend_name = "aer_simulator"
    else:
        backend_name = getattr(backend, "name", str(backend))
    if sampler is None:
        if backend_name == "aer_simulator":
            from qiskit_aer.primitives import SamplerV2
            sampler = SamplerV2(seed=seed, default_shots=shots)
        else:
            sampler = _runtime_sampler(backend)

    pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend)
    isa = pm.run(qc)
    ops = dict(isa.count_ops())
    two_q = sum(v for k, v in ops.items() if k in ("cx", "cz", "ecr", "rzz", "swap", "iswap"))
    transpiled = {"depth": isa.depth(), "gate_counts": ops, "two_qubit_gates": int(two_q),
                  "num_qubits": isa.num_qubits}

  
    exact_tables = R <= max_R_exact
    if exact_tables:
        energies_all = _all_energies(qp)
        feas_all = feasible_mask_all(qp)
        opt_idx = int(np.argmin(energies_all))
        num_feasible = int(feas_all.sum())
    else:
        energies_all = feas_all = None
        opt_idx, num_feasible = None, None

    def energy_of(idx: int) -> float:
        if exact_tables:
            return float(energies_all[idx])
        bits = np.array([(idx >> q) & 1 for q in range(R)])
        return qubo_energy(qp, bits)

    def feasible_of(idx: int) -> bool:
        if exact_tables:
            return bool(feas_all[idx])
        bits = np.array([(idx >> q) & 1 for q in range(R)])
        return decode_value(qp, bits)["feasible"]

    param_index = {}
    for l in range(layer_count):
        param_index[gam[l]] = l
        param_index[bet[l]] = layer_count + l
    bind_order = [param_index[prm] for prm in isa.parameters]

    def sample(theta, n_shots):
        theta = np.asarray(theta, dtype=float)
        pub = (isa, [float(theta[i]) for i in bind_order])
        res = sampler.run([pub], shots=n_shots).result()
        return res[0].data.meas.get_int_counts()

    def cvar_from_counts(counts):
        idxs = np.array(list(counts.keys()), dtype=np.int64)
        cnts = np.array(list(counts.values()), dtype=float)
        e = np.array([energy_of(int(i)) for i in idxs])
        order = np.argsort(e, kind="stable")
        e, cnts = e[order], cnts[order]
        cum = np.cumsum(cnts)
        cutoff = int(np.searchsorted(cum, cvar_alpha * cum[-1])) + 1
        w = cnts[:cutoff]
        return float(np.sum(w * e[:cutoff]) / max(w.sum(), 1e-12))

    if theta0 is None:
        if R <= 20:
            ref = numpy_qaoa(qp, layer_count=layer_count, cvar_alpha=cvar_alpha, seed=seed,
                             normalize_energy=normalize_energy, max_R=20)
            theta0 = ref["theta"]
        else:
            rng = np.random.default_rng(seed)
            theta0 = np.concatenate([rng.uniform(0, 2 * np.pi, layer_count),
                                     rng.uniform(0, np.pi, layer_count)])
    theta0 = np.asarray(theta0, dtype=float)
    assert theta0.shape == (2 * layer_count,), "theta0 must have length 2*layer_count"

    n_evals = 0
    if refine and maxiter > 0:
        def objective(th):
            nonlocal n_evals
            n_evals += 1
            return cvar_from_counts(sample(th, shots))
        res = minimize(objective, theta0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.15})
        theta = np.asarray(res.x); cvar_cost = float(res.fun)
    else:
        theta = theta0; cvar_cost = cvar_from_counts(sample(theta, shots)); n_evals = 1

    counts = sample(theta, final_shots)
    total = float(sum(counts.values()))
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    feasible_shots = 0.0
    best_idx, best_e = None, np.inf
    samples = []
    for idx, c in items:
        e = energy_of(int(idx)); f = feasible_of(int(idx))
        if f:
            feasible_shots += c
            if e < best_e:
                best_e, best_idx = e, int(idx)
        if len(samples) < top_k_samples:
            samples.append((tuple(int((idx >> q) & 1) for q in range(R)), float(e), int(c)))
    most_prob_idx = int(items[0][0])

    result = {
        "skipped": False, "backend": backend_name, "shots": shots, "final_shots": int(total),
        "theta": theta.tolist(), "theta0": theta0.tolist(), "cvar_cost": cvar_cost,
        "optimizer_evals": n_evals, "refined": bool(refine and maxiter > 0),
        "runtime_s": time.time() - t0, "qubit_count": R, "layer_count": layer_count,
        "energy_scale": energy_scale, "cvar_alpha": cvar_alpha, "transpiled": transpiled,
        "feasible_shot_fraction": feasible_shots / max(total, 1.0),
        "feasible_mass": feasible_shots / max(total, 1.0),
        "num_feasible_states": num_feasible, "num_states": 1 << R,
        "uniform_feasible_mass": (num_feasible / (1 << R)) if num_feasible is not None else None,
        "p_optimum": (counts.get(opt_idx, 0) / max(total, 1.0)) if opt_idx is not None else None,
        "energy_most_probable": energy_of(most_prob_idx),
        "samples": samples,
    }
    if best_idx is not None:
        b = np.array([(best_idx >> q) & 1 for q in range(R)], dtype=int)
        result.update({"bits": b, "energy": float(best_e), "feasible_source": "qaoa_samples",
                       **decode_value(qp, b)})
    else:
        result.update({"bits": None, "energy": None, "feasible": False, "feasible_source": "none"})
    return result

def _xy_chain_pairs(R: int) -> list[tuple[int, int]]:
    """Gate order of the chain mixer: first the even pairs (0,1),(2,3),...,
    then the odd pairs (1,2),(3,4),... Each XY term preserves the Hamming
    weight."""
    return [(i, i + 1) for start in (0, 1) for i in range(start, R - 1, 2)]


def xy_layer_plan(R: int, init_bits, layer_count: int, mixer_first: bool = True) -> dict:
    active = {i for i, b in enumerate(init_bits) if int(b) == 1}
    xy_gates: list[list[tuple[int, int]]] = []
    phase_active: list[set[int]] = []

    def xy_layer():
        gates = []
        for (a, b) in _xy_chain_pairs(R):
            if a in active or b in active:
                gates.append((a, b)); active.update((a, b))
        xy_gates.append(gates)

    if mixer_first:
        xy_layer()
    for _ in range(layer_count):
        phase_active.append(set(active))
        xy_layer()
    return {"xy_gates": xy_gates, "phase_active": phase_active, "active": sorted(active)}


def build_qaoa_xy_circuit(qp_xy: QuboProblem, init_bits, layer_count: int = 1,
                          energy_scale: float | None = None, mixer_first: bool = True,
                          prune: bool = True):
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.circuit.library import XXPlusYYGate

    scale = float(energy_scale if energy_scale is not None else max(qp_xy.B, 1e-9))
    h, edges, _ = to_ising(qp_xy.Q / scale)
    R = len(h)
    plan = xy_layer_plan(R, init_bits, layer_count, mixer_first)
    if not prune:
        plan = {"xy_gates": [_xy_chain_pairs(R)] * len(plan["xy_gates"]),
                "phase_active": [set(range(R))] * layer_count, "active": list(range(R))}
    active = plan["active"]
    qmap = {g: i for i, g in enumerate(active)}
    n_beta = layer_count + (1 if mixer_first else 0)
    gammas = ParameterVector("gamma", layer_count)
    betas = ParameterVector("beta", n_beta)
    qc = QuantumCircuit(len(active))
    for g in active:
        if int(init_bits[g]) == 1:
            qc.x(qmap[g])
    kept = folded = dropped = 0
    n_xy = 0

    def xy_layer(gates, beta):
        nonlocal n_xy
        for (a, b) in gates:
            qc.append(XXPlusYYGate(4.0 * beta), [qmap[a], qmap[b]]); n_xy += 1

    xy_idx = 0
    if mixer_first:
        xy_layer(plan["xy_gates"][0], betas[0]); xy_idx = 1
    for l in range(layer_count):
        S = plan["phase_active"][l]
        h_eff = np.array(h, dtype=float)
        for (u, v, w) in edges:
            if u in S and v in S:
                qc.rzz(2.0 * gammas[l] * w, qmap[u], qmap[v]); kept += 1
            elif u in S:      # v is definitely |0> (s_v=+1): J s_u s_v -> J s_u
                h_eff[u] += w; folded += 1
            elif v in S:
                h_eff[v] += w; folded += 1
            else:
                dropped += 1  # global phase
        for i in S:
            if h_eff[i] != 0.0:
                qc.rz(2.0 * gammas[l] * h_eff[i], qmap[i])
        xy_layer(plan["xy_gates"][xy_idx], betas[xy_idx]); xy_idx += 1
    qc.measure_all()
    info = {"edges_total": len(edges), "edges_kept": kept, "edges_folded": folded, "edges_dropped": dropped,
            "xy_gates": n_xy, "active": list(active), "qubit_map": qmap, "mixer_first": mixer_first,
            "pruned": bool(prune)}
    return qc, list(gammas), list(betas), scale, info


class XYSubspaceSimulator:
    def __init__(self, qp_xy: QuboProblem, init_bits, layer_count: int = 1,
                 energy_scale: float | None = None, mixer_first: bool = True):
        from itertools import combinations
        self.R = R = qp_xy.Q.shape[0]
        self.K = K = int(sum(int(b) for b in init_bits))
        self.layer_count = layer_count; self.mixer_first = mixer_first
        self.scale = float(energy_scale if energy_scale is not None else max(qp_xy.B, 1e-9))
        states = np.array([sum(1 << q for q in comb) for comb in combinations(range(R), K)], dtype=np.int64)
        self.states = states
        self.index = {int(s): i for i, s in enumerate(states)}
        bits = ((states[:, None] >> np.arange(R)[None, :]) & 1).astype(float)
        Qs = qp_xy.Q / self.scale
        self.phase_energy = np.einsum("ni,ij,nj->n", bits, np.triu(Qs), bits)  # y^T Q y (upper triangle + diagonal)
        self.init_index = self.index[int(sum(int(b) << i for i, b in enumerate(init_bits)))]
        self.plan = xy_layer_plan(R, init_bits, layer_count, mixer_first)
        # index pairs for each XY gate (A: states with y_a=1,y_b=0; B: their counterparts)
        self._pair_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        for gates in self.plan["xy_gates"]:
            for (a, b) in gates:
                if (a, b) in self._pair_cache:
                    continue
                A = np.where(((states >> a) & 1 == 1) & ((states >> b) & 1 == 0))[0]
                partner = states[A] - (1 << a) + (1 << b)
                B = np.array([self.index[int(s)] for s in partner], dtype=np.int64)
                self._pair_cache[(a, b)] = (A, B)

    def _apply_xy(self, psi, gates, beta):
        c, s = np.cos(2.0 * beta), -1j * np.sin(2.0 * beta)
        for g in gates:
            A, B = self._pair_cache[g]
            pa, pb = psi[A].copy(), psi[B].copy()
            psi[A] = c * pa + s * pb
            psi[B] = s * pa + c * pb
        return psi

    def split_theta(self, theta):
        theta = np.asarray(theta, dtype=float); p = self.layer_count
        return theta[:p], theta[p:]

    def statevector(self, theta):
        gammas, betas = self.split_theta(theta)
        psi = np.zeros(len(self.states), dtype=complex); psi[self.init_index] = 1.0
        xi = 0
        if self.mixer_first:
            psi = self._apply_xy(psi, self.plan["xy_gates"][0], betas[0]); xi = 1
        for l in range(self.layer_count):
            psi = psi * np.exp(-1j * gammas[l] * self.phase_energy)
            psi = self._apply_xy(psi, self.plan["xy_gates"][xi], betas[xi]); xi += 1
        return psi

    def probabilities(self, theta) -> np.ndarray:
        return np.abs(self.statevector(theta)) ** 2

    def full_probabilities(self, theta) -> np.ndarray:
        """Probability vector of length 2^R (states outside weight K are 0)."""
        out = np.zeros(1 << self.R); out[self.states] = self.probabilities(theta); return out


def _cvar_exact(probs: np.ndarray, energies: np.ndarray, alpha: float) -> float:
    order = np.argsort(energies, kind="stable")
    ps, es = probs[order], energies[order]
    cum = np.cumsum(ps)
    cutoff = int(np.searchsorted(cum, alpha * cum[-1])) + 1
    w = ps[:cutoff]
    return float(np.sum(w * es[:cutoff]) / max(w.sum(), 1e-12))


def optimize_xy_parameters(sim: XYSubspaceSimulator, energies_full: np.ndarray, cvar_alpha: float = 0.02,
                           grid_gamma: int = 16, grid_beta: int = 8, maxiter: int = 200,
                           seed: int = 0, n_restarts: int = 3) -> dict:
    from scipy.optimize import minimize
    E = energies_full[sim.states]
    p = sim.layer_count; nb = p + (1 if sim.mixer_first else 0)

    def cost(theta):
        return _cvar_exact(sim.probabilities(theta), E, cvar_alpha)

    gam = np.linspace(0.0, 2 * np.pi, grid_gamma, endpoint=False)
    bet = np.linspace(0.0, np.pi / 2, grid_beta, endpoint=False)[1:]  # β=0 is the identity; skip it
    cands = []
    if sim.mixer_first and p == 1:
        for b0 in bet:
            for g in gam:
                for b1 in bet:
                    th = np.array([g, b0, b1]); cands.append((cost(th), th))
    else:
        for g in gam:
            for b in bet:
                th = np.concatenate([np.full(p, g), np.full(nb, b)]); cands.append((cost(th), th))
    cands.sort(key=lambda t: t[0])
    rng = np.random.default_rng(seed)
    best = (np.inf, None); n_evals = len(cands)
    for k in range(min(n_restarts, len(cands))):
        th0 = cands[k][1]
        if k > 0:
            th0 = th0 + rng.normal(0, 0.05, size=th0.shape)
        res = minimize(cost, th0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.15})
        n_evals += res.nfev
        if res.fun < best[0]:
            best = (float(res.fun), np.asarray(res.x, dtype=float))
    return {"theta": best[1].tolist(), "cvar_cost": best[0], "grid_best": cands[0][1].tolist(),
            "grid_best_cost": float(cands[0][0]), "grid_points": len(cands), "optimizer_evals": int(n_evals)}


def qiskit_qaoa_xy(qp_xy: QuboProblem, qp_full: QuboProblem, init_bits, layer_count: int = 1,
                   theta0=None, optimize: bool = True, grid_gamma: int = 16, grid_beta: int = 8,
                   maxiter: int = 200, shots: int = 4000, final_shots: int = 8192,
                   cvar_alpha: float = 0.1, seed: int = 0, backend=None, sampler=None,
                   optimization_level: int = 3, top_k_samples: int = 4096,
                   mixer_first: bool = True, prune: bool = True, transpile_seeds: int = 4,
                   max_R_subspace: int = 26) -> dict:
 
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    t0 = time.time()
    R = qp_full.Q.shape[0]
    K = int(sum(int(b) for b in init_bits))
    energies_all = _all_energies(qp_full)
    feas_all = feasible_mask_all(qp_full)
    opt_idx = int(np.argmin(energies_all))
    num_feasible = int(feas_all.sum())
    init_idx = int(sum(int(b) << i for i, b in enumerate(init_bits)))
    init_energy = float(energies_all[init_idx])

    sim = XYSubspaceSimulator(qp_xy, init_bits, layer_count, mixer_first=mixer_first) if R <= max_R_subspace else None
    opt_info = None
    if optimize:
        if sim is None:
            raise ValueError(f"R={R} > max_R_subspace={max_R_subspace}: no subspace simulator is available for the parameter search; provide theta0")
        opt_info = optimize_xy_parameters(sim, energies_all, cvar_alpha=cvar_alpha, grid_gamma=grid_gamma,
                                          grid_beta=grid_beta, maxiter=maxiter, seed=seed)
        theta = np.asarray(opt_info["theta"], dtype=float)
    else:
        assert theta0 is not None, "theta0 is required when optimize=False"
        theta = np.asarray(theta0, dtype=float)
    n_params = layer_count + layer_count + (1 if mixer_first else 0)
    if theta.shape[0] != n_params:
        raise ValueError(f"theta has length {theta.shape[0]}, expected {n_params} "
                         f"([gamma_1..gamma_p, beta_0, beta_1..beta_p]; mixer_first={mixer_first})")

    qc, gam, bet, energy_scale, info = build_qaoa_xy_circuit(qp_xy, init_bits, layer_count,
                                                             mixer_first=mixer_first, prune=prune)
    active = info["active"]
    n_edges = info["edges_kept"]

    if backend is None:
        from qiskit_aer import AerSimulator
        backend = AerSimulator(seed_simulator=seed)
        backend_name = "aer_simulator"
    else:
        backend_name = getattr(backend, "name", str(backend))
    if sampler is None:
        if backend_name == "aer_simulator":
            from qiskit_aer.primitives import SamplerV2
            sampler = SamplerV2(seed=seed, default_shots=shots)
        else:
            sampler = _runtime_sampler(backend)

    # --- transpile: multiple SABRE seeds, fewest two-qubit gates ---
    two_q_names = ("cx", "cz", "ecr", "rzz", "swap", "iswap", "xx_plus_yy")
    has_coupling = getattr(backend, "coupling_map", None) is not None
    n_seeds = max(1, transpile_seeds) if has_coupling else 1
    best = None
    for s in range(n_seeds):
        pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend,
                                          seed_transpiler=seed + s)
        cand = pm.run(qc)
        ops = dict(cand.count_ops())
        two_q = int(sum(v for k, v in ops.items() if k in two_q_names))
        if best is None or (two_q, cand.depth()) < (best[1], best[2]):
            best = (cand, two_q, cand.depth(), ops, seed + s)
    isa, two_q, depth, ops, used_seed = best
    layout = None
    if isa.layout is not None:
        fl = isa.layout.final_index_layout()
        layout = [int(q) for q in fl[:len(active)]]
    transpiled = {"depth": depth, "gate_counts": ops, "two_qubit_gates": two_q,
                  "num_qubits": isa.num_qubits, "active_qubits": len(active), "zz_edges": n_edges,
                  "transpile_seed": used_seed, "seeds_tried": n_seeds, "physical_layout": layout}

    param_index = {}
    for l in range(layer_count):
        param_index[gam[l]] = l
    for j in range(len(bet)):
        param_index[bet[j]] = layer_count + j
    bind_order = [param_index[prm] for prm in isa.parameters]

    # local (active-qubit) integer -> global R-bit integer
    def to_global(local_idx: int) -> int:
        g = 0
        for li, q in enumerate(active):
            if (local_idx >> li) & 1:
                g |= (1 << q)
        return g

    def sample(theta_, n_shots):
        theta_ = np.asarray(theta_, dtype=float)
        res = sampler.run([(isa, [float(theta_[i]) for i in bind_order])], shots=n_shots).result()
        raw = res[0].data.meas.get_int_counts()
        out: dict[int, int] = {}
        for k, v in raw.items():
            gk = to_global(int(k)); out[gk] = out.get(gk, 0) + int(v)
        return out

    counts = sample(theta, final_shots)
    total = float(sum(counts.values()))
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    feasible_shots = 0.0; weight_k_shots = 0.0; improved_shots = 0.0
    best_idx, best_e = None, np.inf
    samples = []
    for idx, c in items:
        idx = int(idx)
        e = float(energies_all[idx]); f = bool(feas_all[idx])
        if bin(idx).count("1") == K:
            weight_k_shots += c
        if f:
            feasible_shots += c
            if e < init_energy - 1e-12:
                improved_shots += c
            if e < best_e:
                best_e, best_idx = e, idx
        if len(samples) < top_k_samples:
            samples.append((tuple(int((idx >> q) & 1) for q in range(R)), e, int(c)))

    reference = None
    if sim is not None:
        pf = sim.full_probabilities(theta)
        reference = {"feasible_mass": float(pf[feas_all].sum()), "p_optimum": float(pf[opt_idx]),
                     "improved_fraction": float(pf[feas_all & (energies_all < init_energy - 1e-12)].sum()),
                     "init_state_fraction": float(pf[init_idx]),
                     "cvar_cost": _cvar_exact(pf[sim.states], energies_all[sim.states], cvar_alpha)}

    result = {
        "skipped": False, "backend": backend_name, "mixer": "XY-chain", "shots": shots,
        "final_shots": int(total), "theta": theta.tolist(),
        "theta_layout": "[gamma_1..gamma_p, beta_0, beta_1..beta_p]" if mixer_first else "[gamma_1..gamma_p, beta_1..beta_p]",
        "mixer_first": mixer_first,
        "cvar_cost": None if opt_info is None else opt_info["cvar_cost"],
        "optimizer_evals": 0 if opt_info is None else opt_info["optimizer_evals"],
        "grid": None if opt_info is None else {"theta": opt_info["grid_best"], "cvar": opt_info["grid_best_cost"],
                                                "grid_points": opt_info["grid_points"]},
        "runtime_s": time.time() - t0,
        "qubit_count": R, "active_qubits": list(active), "layer_count": layer_count, "energy_scale": energy_scale,
        "pruning": {k: info[k] for k in ("edges_total", "edges_kept", "edges_folded", "edges_dropped", "xy_gates", "pruned")},
        "transpiled": transpiled,
        "feasible_mass": feasible_shots / max(total, 1.0),
        "weight_k_fraction": weight_k_shots / max(total, 1.0),
        "feasible_given_k": (feasible_shots / weight_k_shots) if weight_k_shots > 0 else 0.0,
        "init_state_fraction": counts.get(init_idx, 0) / max(total, 1.0),
        "improved_fraction": improved_shots / max(total, 1.0),   # fraction of feasible samples BETTER than the initial state
        "num_feasible_states": num_feasible, "num_states": 1 << R,
        "uniform_feasible_mass": num_feasible / (1 << R),
        "p_optimum": counts.get(opt_idx, 0) / max(total, 1.0),
        "energy_most_probable": float(energies_all[int(items[0][0])]),
        "init_energy": init_energy,
        "optimum_energy": float(energies_all[opt_idx]),
        "reference": reference,
        "samples": samples,
    }
    if best_idx is not None:
        b = np.array([(best_idx >> q) & 1 for q in range(R)], dtype=int)
        result.update({"bits": b, "energy": float(best_e), "feasible_source": "qaoa_samples",
                       **decode_value(qp_full, b)})
    else:
        result.update({"bits": None, "energy": None, "feasible": False, "feasible_source": "none"})
    return result


def qiskit_resource_report(qp: QuboProblem, layer_count: int = 2,
                            heavy_hex_distance: int = 5) -> dict:
    from qiskit import QuantumCircuit
    from qiskit.transpiler import CouplingMap
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    h, edges, _ = to_ising(qp.Q)
    R = len(h)

    qc = QuantumCircuit(R)
    qc.h(range(R))
    for layer in range(layer_count):
        gamma, beta = 0.31 + 0.01 * layer, 0.27 + 0.01 * layer
        for (u, v, w) in edges:
            qc.cx(u, v)
            qc.rz(2.0 * gamma * w, v)
            qc.cx(u, v)
        for i in range(R):
            qc.rz(2.0 * gamma * h[i], i)
        for i in range(R):
            qc.rx(2.0 * beta, i)
    qc.measure_all()

    logical = {
        "qubit_count": R,
        "depth": qc.depth(),
        "gate_counts": dict(qc.count_ops()),
        "cx_count": qc.count_ops().get("cx", 0),
    }

    coupling = CouplingMap.from_heavy_hex(heavy_hex_distance)
    pm = generate_preset_pass_manager(optimization_level=2, coupling_map=coupling,
                                       basis_gates=["rz", "sx", "x", "cx"])
    transpiled = pm.run(qc)
    physical = {
        "physical_qubit_count": coupling.size(),
        "depth": transpiled.depth(),
        "gate_counts": dict(transpiled.count_ops()),
        "cx_count": transpiled.count_ops().get("cx", 0),
    }
    return {"logical": logical, "transpiled_heavy_hex": physical,
            "num_edges": len(edges), "layer_count": layer_count}
