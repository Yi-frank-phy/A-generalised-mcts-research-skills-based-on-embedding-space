# Prototype Notes

This repository now contains a runnable offline prototype of the DTE skill/backend loop.

The prototype is deliberately deterministic and does not call external LLMs. Its purpose is to validate the machine-facing protocol before wiring Codex/Kimi/OpenClaw executor adapters.

## Implemented

- Pydantic run spec and search node models.
- Deterministic batch Judge heuristic with per-run content-hash cache.
- Local hashed text features for offline novelty/uncertainty estimation with embedding cache.
- Canonical local UCB score:

```text
U_i = V_i + SD_i
```

  where the current `uncertainty` field is the provisional standard-deviation/standard-error-like estimator.
- Global controller temperature is separate from UCB: for the current provisional diversity proxy `H`, `tau = H / log(N)` when `N > 1` (else `0`) and `T = T_max * tau`.
- Boltzmann expansion allocation over UCB by default: `p_i ∝ exp(U_i / T)`.
- Cross-iteration entropy/proxy delta is plateau telemetry only; it does not define UCB or temperature.
- Executor subprocess adapter boundary with strict child-node validation.
- Conservative equivalent-claim merge skeleton.
- Cache telemetry written to `cache_stats.json`.
- Deterministic expansion operator that closes expanded parents and appends child SearchNodes.
- Mandatory frontier loop: Judge → novelty/uncertainty → UCB/allocation → expansion → synthesis.
- CLI commands for validate, allocate, and run.
- Tests for schema, math engine, runner, cache, adapter boundary, and merge skeleton.

The current KDE-derived `H` is retained only as a compatibility diversity proxy. It is not claimed to be a validated thermodynamic entropy or proof of epistemic convergence.

## Executor adapter boundary

The prototype now exposes the Executor role as an adapter boundary inside the
Expansion phase. A subprocess adapter may read an ExpansionRequest JSON object
from stdin and return either a SearchNode JSON list or `{"nodes": [...]}` on
stdout. Returned nodes are validated as frontier children and cannot pre-fill
Judge/Evolution metrics or produce synthesis nodes.

Example:

```bash
python -m dte_backend validate-executor --request examples/expansion_request.json --executor-command "python examples/echo_executor_adapter.py"
python -m dte_backend run --spec examples/run_spec.json --nodes examples/frontier_nodes.json --executor-command "python path/to/adapter.py"
```

## Not implemented yet

- Real LLM Judge.
- Concrete Codex/Kimi/OpenClaw command wrappers around the subprocess executor adapter boundary.
- Final research-grade estimator for local `SD_i`.
- Final research-state diversity/entropy observable replacing the quarantined batch-relative KDE proxy.
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

The prototype does not add cost penalty into UCB. The local exploration objective remains value plus uncertainty. Costs are controlled by hard budgets and model/executor policy. Global temperature only controls how concentrated Boltzmann resource allocation is across the already-computed UCB values.

## Next step

Replace the provisional local `SD_i` and global diversity proxy only when the method→understanding research provides better estimators; preserve the canonical controller separation unless new mathematics justifies changing it.
