# Prototype Notes

This repository now contains a runnable offline prototype of the DTE skill/backend loop.

The prototype is deliberately deterministic and does not call external LLMs. Its purpose is to validate the machine-facing protocol before wiring Codex/Kimi/OpenClaw executor adapters.

## Implemented

- Pydantic run spec and search node models.
- Deterministic batch Judge heuristic.
- Local hashed text features for offline novelty/uncertainty estimation.
- UCB score:

```text
U_i = V_i + c * tau * uncertainty_i
```

- Boltzmann expansion allocation over UCB by default.
- Deterministic expansion operator that closes expanded parents and appends child SearchNodes.
- Mandatory frontier loop: Judge → novelty/uncertainty → UCB/allocation → expansion → synthesis.
- CLI commands for validate, allocate, and run.
- Tests for schema, math engine, and runner.

## Executor adapter boundary

The prototype now exposes the Executor role as an adapter boundary inside the
Expansion phase. A subprocess adapter may read an ExpansionRequest JSON object
from stdin and return either a SearchNode JSON list or `{"nodes": [...]}` on
stdout. Returned nodes are validated as frontier children and cannot pre-fill
Judge/Evolution metrics or produce synthesis nodes.

Example:

```bash
python -m dte_backend run --spec examples/run_spec.json --nodes examples/frontier_nodes.json --executor-command "python path/to/adapter.py"
```

## Not implemented yet

- Real LLM Judge.
- Concrete Codex/Kimi/OpenClaw command wrappers around the executor adapter boundary.
- Real embedding model / KDE backend.
- Merge operator.
- Cache layer.
- Hook enforcement beyond the validation example.

## Run

```bash
python -m pip install -e .[dev]
pytest
python -m dte_backend validate examples/run_spec.json
python -m dte_backend allocate examples/frontier_nodes.json --budget 4
python -m dte_backend run --spec examples/run_spec.json --nodes examples/frontier_nodes.json --out-dir artifacts/prototype
```

If the package is not installed, use:

```bash
PYTHONPATH=src python -m dte_backend run --spec examples/run_spec.json --nodes examples/frontier_nodes.json --out-dir artifacts/prototype
```

## Design choice: UCB not cost-aware

The prototype does not add cost penalty into UCB. The exploration objective remains value/uncertainty driven. Costs are controlled by hard budgets and model/executor policy.

## Next step

Wire concrete Codex/Kimi/OpenClaw command wrappers around the executor adapter
boundary. The adapter must not produce final answers directly.
