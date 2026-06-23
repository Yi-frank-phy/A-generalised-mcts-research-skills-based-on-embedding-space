# CODEX_NEXT_STEPS.md — read this first

This file is intentionally placed at the repository root so Codex can find the current blockers quickly. Do **not** redesign the architecture. Fix these items while preserving the DTE protocol.

## Current status

The repo has a runnable DTE backend skeleton with:

- fixed DTE protocol and AGENTS/SKILL instructions;
- role-isolated seeding with optional compile hints instead of a mandatory Distiller role;
- embedding/KDE/entropy/temperature/UCB/Boltzmann controller;
- executor adapter validation;
- Judge and Relation oracle task contracts;
- subprocess oracle runners and mock adapters;
- file-backed cache;
- Codex-app-facing artifacts;
- `hooks/dte_guard.py` for boundary checks.

## Highest-priority blockers

### 1. Wire Judge oracle into the main run loop

Current state:

- `src/dte_backend/subprocess_oracles.py` has `build_subprocess_judge_adapter()` and `run_subprocess_judge()`.
- `python -m dte_backend judge-oracle --nodes ... --judge-command ...` can run and validate a Judge oracle.
- `run_frontier_search()` still uses the heuristic Judge internally.

Required change:

- Add an optional `judge_adapter` parameter to `run_frontier_search()`.
- In the Judge phase, call `judge_adapter(frontier)` when supplied.
- Fallback to `batch_judge(frontier, cache=cache)` when no adapter is supplied.
- Add CLI support:

```bash
python -m dte_backend run \
  --spec examples/run_spec.json \
  --out-dir artifacts/run \
  --judge-command "python examples/mock_judge_adapter.py"
```

Rules:

- Judge output must pass `validate_judge_output()`.
- Judge must not allocate budget, write embeddings, write UCB, or synthesize.
- EvolutionController remains deterministic Python.

### 2. Fix embedding default mismatch in `embedding.py`

Current state:

- `DTERunSpec.embedding_dimension` defaults to `3072`.
- `schemas/run_spec.schema.json` defaults to `3072`.
- `examples/run_spec.json` explicitly sets `3072`.
- But `GeminiEmbedding2Provider` and `get_embedding_provider()` still show old internal defaults in `src/dte_backend/embedding.py`.

Required change:

- Make all internal provider defaults use `3072`.
- Keep `hash` provider usable for tests, but treat it as fallback/debug only.
- Add or update tests to assert the default provider dimension is `3072`.

### 3. Make hooks easy for Codex to run

Current state:

- `hooks/dte_guard.py` exists and validates `spec`, `executor`, `judge`, and `relation` outputs.
- It is not yet wired into a documented Codex workflow.

Required change:

- Add `hooks/README.md` with exact commands and when to run them.
- Ensure examples exist for each hook mode.
- Add tests or a script that exercises the guard on sample artifacts.

### 4. Complete Relation oracle workflow

Current state:

- Relation oracle task contract exists.
- `relation-oracle` CLI can run a subprocess and validate the result.
- Deterministic merge only performs conservative exact equivalent-claim merging.

Required change:

- Add a documented workflow for when Codex should call relation oracle:
  - after expansion;
  - when frontier nodes are semantically close;
  - when branches conflict;
  - when entropy plateaus.
- Do not mutate graph directly from subagent output. Validate first, then create a `MergeProposal` or discriminator task.

### 5. Keep Distiller removed

Current state:

- The old mandatory Distiller role has been removed from the seed chain.
- `CompileHint` now describes optional agent-local context compilation.

Required rule:

- Do not restore a mandatory Distiller phase.
- Compile is a prompt-level operation that any subagent may choose when its context is too large.

## Important constraints

- Do not introduce LangChain/LangGraph as a required dependency.
- Do not turn the repo into a web UI or SaaS.
- Do not make UCB cost-aware by default.
- Do not lower Gemini embedding dimension below `3072` for real geometry.
- Do not treat Judge as an embedding model; closed Judge only returns observable judgments.
- Do not let executor or oracle subagents produce the final answer directly.

## Minimum validation after changes

Run:

```bash
python -m pip install -e .[dev]
pytest
python -m dte_backend validate examples/run_spec.json
python hooks/dte_guard.py spec examples/run_spec.json
python -m dte_backend judge-oracle --nodes examples/frontier_nodes.json --judge-command "python examples/mock_judge_adapter.py"
python -m dte_backend relation-oracle --nodes examples/frontier_nodes.json --relation-command "python examples/mock_relation_adapter.py"
python -m dte_backend run --spec examples/run_spec.json --out-dir artifacts/smoke --cache-path .dte_cache/cache.json
```

After Judge is wired into `run`, also run:

```bash
python -m dte_backend run \
  --spec examples/run_spec.json \
  --out-dir artifacts/judge-smoke \
  --cache-path .dte_cache/cache.json \
  --judge-command "python examples/mock_judge_adapter.py"
```

## Expected architecture after completion

```text
main agent / Codex app
  -> DTERunSpec
  -> DTE backend validates spec
  -> optional role-isolated seeding
  -> Judge oracle subagent returns observable scores
  -> EvolutionController computes embedding/KDE/entropy/UCB/Boltzmann
  -> Executor subagent returns SearchNode children
  -> Relation oracle subagent classifies merge/conflict/complementarity
  -> backend validates all outputs
  -> DTE synthesis produces final report
```
