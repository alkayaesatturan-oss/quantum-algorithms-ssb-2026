# Quantum Algorithms

A collection of my work on quantum algorithms: competition projects, algorithm
implementations and experiments on simulators and real quantum hardware.

## Projects

| Project | Topic | Methods | Hardware |
|---|---|---|---|
| [ATM Cash Flow Optimization](projects/ssb-2026-atm-cash-optimization/) — SSB Quantum Algorithms Competition, 2026 | Vehicle routing / combinatorial optimization | QUBO, QAOA, XY-mixer QAOA | IBM `ibm_pittsburgh` |

## Repository Layout

```
quantum-algorithms/
└── projects/    # Self-contained projects, each with its own README and requirements
```

Each project directory is self-contained: install its dependencies with
`pip install -r requirements.txt` from within that directory.

## License

Released under the [MIT License](LICENSE).
