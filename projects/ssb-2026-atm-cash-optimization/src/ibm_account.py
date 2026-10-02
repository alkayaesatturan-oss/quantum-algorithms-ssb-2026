"""
Reads IBM Quantum credentials from a `.env` file (or from environment
variables) and returns a QiskitRuntimeService. Resolution order:

  1. The IBM_QUANTUM_TOKEN environment variable (loaded from .env if present)
  2. Otherwise, the account saved by Qiskit (~/.qiskit/qiskit-ibm.json)

.env fields (see .env.example):
  IBM_QUANTUM_TOKEN=...          required (if this path is used)
  IBM_QUANTUM_CHANNEL=ibm_quantum_platform   (default)
  IBM_QUANTUM_INSTANCE=...       optional (CRN); the account default is used if empty
"""
from __future__ import annotations

import os

ENV_TOKEN = "IBM_QUANTUM_TOKEN"
ENV_CHANNEL = "IBM_QUANTUM_CHANNEL"
ENV_INSTANCE = "IBM_QUANTUM_INSTANCE"


def find_dotenv(start: str | None = None) -> str | None:
    """Finds the first `.env` file in `start` (default: the parent directory
    of this package, i.e. the project root) or any of its ancestors."""
    d = os.path.abspath(start or os.path.join(os.path.dirname(__file__), ".."))
    while True:
        cand = os.path.join(d, ".env")
        if os.path.isfile(cand):
            return cand
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def load_dotenv(path: str | None = None, override: bool = False) -> dict[str, str]:
    """Loads KEY=VALUE lines into os.environ (comments and blank lines are
    skipped, surrounding quotes are stripped). Returns the loaded keys (not
    their values)."""
    path = path or find_dotenv()
    loaded: dict[str, str] = {}
    if not path or not os.path.isfile(path):
        return loaded
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            if key.startswith("export "):
                key = key[len("export "):].strip()
            val = val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                val = val[1:-1]
            if override or key not in os.environ:
                os.environ[key] = val
            loaded[key] = "<set>"
    return loaded


def get_runtime_service(dotenv_path: str | None = None, verbose: bool = True):
    """QiskitRuntimeService: uses the .env/environment token if available,
    otherwise the saved account. Returns: (service, source_description)."""
    import logging
    from qiskit_ibm_runtime import QiskitRuntimeService
    logging.getLogger("qiskit_ibm_runtime.qiskit_runtime_service").setLevel(logging.ERROR)

    load_dotenv(dotenv_path)
    token = os.environ.get(ENV_TOKEN, "").strip()
    if token:
        kwargs = {"channel": os.environ.get(ENV_CHANNEL, "ibm_quantum_platform").strip() or "ibm_quantum_platform",
                  "token": token}
        instance = os.environ.get(ENV_INSTANCE, "").strip()
        if instance:
            kwargs["instance"] = instance
        service = QiskitRuntimeService(**kwargs)
        source = f".env / environment variable ({ENV_TOKEN}, channel={kwargs['channel']}"
        source += f", instance provided)" if instance else ")"
    else:
        service = QiskitRuntimeService()
        source = "Qiskit saved account (~/.qiskit/qiskit-ibm.json)"
    if verbose:
        print(f"[IBM account] source: {source}")
    return service, source
