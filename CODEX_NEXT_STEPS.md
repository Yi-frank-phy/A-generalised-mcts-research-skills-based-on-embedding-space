# CODEX_NEXT_STEPS.md — read this first

This file is intentionally placed at the repository root so Codex can find the current state quickly. Do **not** redesign the architecture. Continue by wiring the remaining oracle workflow while preserving the DTE protocol.

## Current status

The repo now has a runnable DTE backend with:

- fixed DTE protocol and AGENTS/SKILL instructions;
- role-isolated seeding with optional compile hints instead of a mandatory Distiller role;
- max-dimensional embedding geometry by default (`embedding_dimension=3072`);
- embedding/KDE/entropy/temperature/UCB/Boltzmann controller;
- executor adapter validation;
- Judge and Relation oracle task contracts;
- subprocess oracle runners and mock adapters;
- `run --judge-command ...` wired into the main loop;
- file-backed cache;
- Codex-app-facing artifacts;
- `hooks/dte_guard.py`, `hooks/README.md`, and hook tests for boundary checks.

## Done — do not redo

These items were previous blockers and are now complete:

1. Judge oracle can be run standalone via `judge-oracle` and can also be passed into the main run with `--judge-command`.
2. Gemini Embedding 2 provider defaults to `3072`, and the run spec/schema/example also default to or explicitly use `3072`.
3. Hook documentation exists in `hooks/README.md`, and `tests/test_hooks.py` exercises sample guard commands.
4. The mandatory Distiller role has been removed. `CompileHint` is only an optional agent-local compression hint.

## Highest-priority remaining blockers

### 1. Complete Relation oracle graph workflow

Current state:

- Relation oracle task contract exists.
- `relation-oracle` CLI can run a subprocess and validate the result.
- `hooks/dte_guard.py relation ...` can validate relation outputs.
- Deterministic merge still only performs conservative exact normalized-claim merging.

Required change:

- Add a relation workflow module that turns a validated `RelationOracleResult` into one of:
  - an `equivalent_merge` proposal;
  - a `complementary_merge` proposal;
  - a `conflict_merge` proposal;
  - a discriminator SearchNode/task for unresolved conflict;
  - no-op for `independent`.
- Do not let relation oracle output mutate the graph directly.
- Backend must validate first, then create a typed proposal/task.

### 2. Add Codex subagent prompt templates

Current state:

- Oracle contracts exist in Python.
- Mock adapters exist.
- There are not yet clear prompt templates for a real Codex Judge subagent or Relation subagent.

Required change:

- Add `prompts/judge_oracle.md`.
- Add `prompts/relation_oracle.md`.
- Add `prompts/executor_subagent.md` if missing or stale.
- Each prompt must require JSON-only machine output and explicitly forbid final synthesis.

### 3. Add relation workflow tests

Required tests:

- equivalent relation result -> equivalent merge proposal;
- complementary relation result -> complementary merge proposal;
- conflict relation result with discriminator question -> discriminator SearchNode or conflict proposal;
- independent relation result -> no graph mutation.

### 4. Add a single smoke command that exercises the complete current workflow

Current validation requires several manual commands.

Required change:

- Add either a script or documented command sequence that runs:
  - spec guard;
  - judge oracle;
  - relation oracle;
  - DTE run with `--judge-command`;
  - output artifact check.

Do not add a web UI. The Codex app plus markdown artifacts remain the frontend.

## Important constraints

- Do not introduce LangChain/LangGraph as a required dependency.
- Do not turn the repo into a web UI or SaaS.
- Do not make UCB cost-aware by default.
- Do not lower Gemini embedding dimension below `3072` for real geometry.
- Do not treat Judge as an embedding model; closed Judge only returns observable judgments.
- Do not let executor or oracle subagents produce the final answer directly.
- Do not restore mandatory Distiller. Compile remains optional and agent-local.

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
python -m dte_backend run --spec examples/run_spec.json --out-dir artifacts/judge-smoke --cache-path .dte_cache/cache.json --judge-command "python examples/mock_judge_adapter.py"
```

## Expected architecture

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
  -> backend converts relation results into proposals/tasks
  -> DTE synthesis produces final report
```
