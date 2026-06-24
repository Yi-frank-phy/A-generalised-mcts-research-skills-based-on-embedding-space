# CODEX_NEXT_STEPS.md — read this first

This file is intentionally placed at the repository root so Codex can find the current state quickly. Do **not** redesign the architecture. Continue by hardening the remaining workflow edges while preserving the DTE protocol.

## Current status

The repo now has a runnable DTE backend with:

- fixed DTE protocol and AGENTS/SKILL instructions;
- role-isolated seeding with optional compile hints instead of a mandatory Distiller role;
- max-dimensional embedding geometry by default (`embedding_dimension=3072`);
- cache-friendly canonical context envelopes for embeddings/Judge evaluation;
- split embedding/Judge cache keys and file-backed persistent cache;
- embedding/KDE/entropy/temperature/UCB/Boltzmann controller;
- executor adapter validation;
- Judge and Relation oracle task contracts;
- subprocess oracle runners and mock adapters;
- `run --judge-command ...` wired into the main loop;
- relation-oracle result conversion into `MergeProposal` or discriminator task;
- deterministic relation candidate-pair selection;
- relation proposal/discriminator machine artifacts via `relation-artifacts`;
- Codex-app-facing artifacts, including `relation_candidates.md`;
- `hooks/dte_guard.py`, `hooks/README.md`, and hook tests for boundary checks;
- subagent prompt templates in `prompts/`;
- Codex app workflow guide in `docs/CODEX_APP_WORKFLOW.md`;
- workflow smoke script at `scripts/smoke_workflow.py`;
- optional Gemini smoke script at `scripts/gemini_smoke.py`.

## Done — do not redo

These items were previous blockers and are now complete:

1. Judge oracle can be run standalone via `judge-oracle` and can also be passed into the main run with `--judge-command`.
2. Gemini Embedding 2 provider defaults to `3072`, and the run spec/schema/example also default to or explicitly use `3072`.
3. Hook documentation exists in `hooks/README.md`, and `tests/test_hooks.py` exercises sample guard commands.
4. The mandatory Distiller role has been removed. `CompileHint` is only an optional agent-local compression hint.
5. Relation oracle outputs can be converted to typed merge proposals or discriminator task nodes.
6. Prompt templates exist for Judge, Relation, and Executor subagents.
7. Relation candidate pairs can be selected deterministically and rendered to `relation_candidates.md`.
8. Codex app workflow documentation exists.
9. Optional Gemini smoke script exists and should only run when `GEMINI_API_KEY` or `GOOGLE_API_KEY` is set.
10. Context cache identity has been upgraded from unstable shortest-context hashes to canonical semantic envelopes.
11. Relation oracle results can be persisted as `relation_proposals.json` and `discriminator_tasks.json`.

## Highest-priority remaining blockers

### 1. Real Codex subagent integration examples

Current state:

- Prompt templates exist.
- Mock subprocess adapters exist.
- `docs/CODEX_APP_WORKFLOW.md` explains the main-agent workflow.

Required change:

- Add example JSON transcripts for Judge, Executor, and Relation subagent calls.
- Add one end-to-end documented example using mock adapters and artifacts.

### 2. Decide relation-oracle execution policy

Current recommendation: keep relation oracle as a main-agent step first, not automatic inside every `run`, to avoid extra subagent calls and latency.

If the user later wants automatic invocation, add an optional `--relation-command` but only call it at safe trigger points:

- when `relation_candidates.md` is non-empty;
- when entropy plateaus;
- when top branches are near-tied;
- when exact duplicates are detected.

Do not call relation oracle on every pair.

### 3. Optional Gemini real smoke

Do not run Gemini API in CI by default. Manual check:

```bash
python scripts/gemini_smoke.py
```

Only run when `GEMINI_API_KEY` or `GOOGLE_API_KEY` is set. Respect free-tier rate limits by embedding only node summaries and using `--cache-path` in real runs.

## Important constraints

- Do not introduce LangChain/LangGraph as a required dependency.
- Do not turn the repo into a web UI or SaaS.
- Do not make UCB cost-aware by default.
- Do not lower Gemini embedding dimension below `3072` for real geometry.
- Do not treat Judge as an embedding model; closed Judge only returns observable judgments.
- Do not let executor or oracle subagents produce the final answer directly.
- Do not restore mandatory Distiller. Compile remains optional and agent-local.
- Do not reintroduce shortest-context cache keys that depend on logs, parent ids, controller metrics, or transient summaries.

## Minimum validation after changes

Run:

```bash
python -m pip install -e .[dev]
pytest
python scripts/smoke_workflow.py
```

Manual individual checks:

```bash
python -m dte_backend validate examples/run_spec.json
python hooks/dte_guard.py spec examples/run_spec.json
python -m dte_backend judge-oracle --nodes examples/frontier_nodes.json --judge-command "python examples/mock_judge_adapter.py"
python -m dte_backend relation-oracle --nodes examples/frontier_nodes.json --relation-command "python examples/mock_relation_adapter.py"
python -m dte_backend relation-artifacts --nodes examples/frontier_nodes.json --relation-output examples/relation_result.json --out-dir artifacts/relation
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
  -> Relation candidate artifact tells main agent which pairs to classify
  -> Relation oracle subagent classifies merge/conflict/complementarity
  -> backend validates all outputs
  -> backend converts relation results into proposals/tasks
  -> DTE synthesis produces final report
```
