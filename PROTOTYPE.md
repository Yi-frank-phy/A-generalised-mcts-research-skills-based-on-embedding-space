# Prototype Notes

This repository now contains a runnable offline prototype of the DTE skill/backend loop.

The prototype is deliberately deterministic and does not call external LLMs. Its purpose is to validate the machine-facing protocol before wiring Codex/Kimi/OpenClaw executor adapters.

## Implemented

- Pydantic run spec and search node models.
- Deterministic batch Judge heuristic with per-run content-hash cache.
- Local hashed text features for offline novelty/uncertainty estimation with embedding cache.
- UCB score:

```text
U_i = V_i + c * tau * uncertainty_i
```

- Boltzmann expansion allocation over UCB by default.
- Executor subprocess adapter boundary with strict child-node validation.
- Conservative equivalent-claim merge skeleton.
- Cache telemetry written to `cache_stats.json`.
- Deterministic expansion operator that closes expanded parents and appends child SearchNodes.
- Mandatory frontier loop: Judge → novelty/uncertainty → UCB/allocation → expansion → synthesis.
- CLI commands for validate, allocate, and run.
- Tests for schema, math engine, runner, cache, adapter boundary, and merge skeleton.

## Not implemented yet

- Real LLM Judge.
- Concrete Codex/Kimi/OpenClaw command wrappers around the subprocess executor adapter boundary.
- Real embedding model / KDE backend.
- Model-backed complementary/conflict merge operator beyond the current equivalent-claim skeleton.
- Persistent cache layer beyond the current in-memory per-run cache.
- Hook enforcement beyond the validation example.

## Run

```bash
python -m pip install -e .[dev]
pytest
python -m dte_backend validate examples/run_spec.json
python -m dte_backend allocate examples/frontier_nodes.json --budget 4
python -m dte_backend run --spec examples/run_spec.json --nodes examples/frontier_nodes.json --out-dir artifacts/prototype
python -m dte_backend run --spec examples/run_spec.json --nodes examples/frontier_nodes.json --out-dir artifacts/adapter --executor-command "python examples/mock_executor_adapter.py"
```

If the package is not installed, use:

```bash
PYTHONPATH=src python -m dte_backend run --spec examples/run_spec.json --nodes examples/frontier_nodes.json --out-dir artifacts/prototype
python -m dte_backend run --spec examples/run_spec.json --nodes examples/frontier_nodes.json --out-dir artifacts/adapter --executor-command "python examples/mock_executor_adapter.py"
```

## Design choice: UCB not cost-aware

The prototype does not add cost penalty into UCB. The exploration objective remains value/uncertainty driven. Costs are controlled by hard budgets and model/executor policy.

## Next step

Wire one real executor adapter that accepts an expansion request and returns structured `SearchNode` JSON. The adapter must not produce final answers directly.
