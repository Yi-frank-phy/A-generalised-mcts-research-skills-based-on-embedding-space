# CODEX_NEXT_STEPS.md — read this first

This file is intentionally placed at the repository root so Codex can find the current state quickly. Do **not** redesign the architecture. Continue by hardening the remaining workflow edges while preserving the DTE protocol.

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
- relation-oracle result conversion into `MergeProposal` or discriminator task;
- file-backed cache;
- Codex-app-facing artifacts;
- `hooks/dte_guard.py`, `hooks/README.md`, and hook tests for boundary checks;
- subagent prompt templates in `prompts/`;
- workflow smoke script at `scripts/smoke_workflow.py`.

## Done — do not redo

These items were previous blockers and are now complete:

1. Judge oracle can be run standalone via `judge-oracle` and can also be passed into the main run with `--judge-command`.
2. Gemini Embedding 2 provider defaults to `3072`, and the run spec/schema/example also default to or explicitly use `3072`.
3. Hook documentation exists in `hooks/README.md`, and `tests/test_hooks.py` exercises sample guard commands.
4. The mandatory Distiller role has been removed. `CompileHint` is only an optional agent-local compression hint.
5. Relation oracle outputs can be converted to typed merge proposals or discriminator task nodes.
6. Prompt templates exist for Judge, Relation, and Executor subagents.

## Highest-priority remaining blockers

### 1. Integrate relation workflow into artifacts or runner policy

Current state:

- `src/dte_backend/relation_workflow.py` can convert a validated relation result into a proposal/task.
- `relation-oracle` CLI can run and validate a relation oracle.
- The main `run` loop still only applies deterministic exact equivalent-claim merge automatically.

Required change:

- Decide whether relation-oracle should be invoked inside `run` or remain a main-agent step between runs.
- If inside `run`, add an optional `--relation-command` and only call it at safe trigger points:
  - after expansion;
  - when frontier nodes are semantically close;
  - when branches conflict;
  - when entropy plateaus.
- If outside `run`, document the main-agent workflow and emit an artifact listing candidate node pairs for relation-oracle calls.

Recommended default: keep relation oracle as a main-agent step first, not automatic inside every run, to avoid extra subagent calls.

### 2. Add candidate-pair selection for relation oracle

Required change:

- Add a small deterministic function that selects node pairs/sets likely worth relation classification:
  - semantically close in embedding/KDE space;
  - near-tied UCB branches;
  - entropy plateau branches;
  - exact duplicate fallback.
- Output candidates to a markdown/json artifact so the main agent knows when to call `relation-oracle`.

### 3. Make real Codex subagent usage explicit

Current state:

- Prompt templates exist.
- Mock adapters exist.
- There is not yet a documented Codex-app procedure for launching subagents with those prompts and feeding validated JSON back to the backend.

Required change:

- Add `docs/CODEX_APP_WORKFLOW.md` describing how the main agent should:
  - run DTE;
  - launch Judge subagent;
  - launch Executor subagent;
  - launch Relation subagent;
  - ask human in chat when `human_questions.md` requests it;
  - summarize `main_agent_status.md` instead of inventing a separate frontend.

### 4. Add optional real Gemini smoke test guard

Do not run Gemini API in CI by default. Add a manual command that only runs when `GEMINI_API_KEY` is set, and explain the free-tier rate-limit discipline.

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
python scripts/smoke_workflow.py
```

Manual individual checks:

```bash
python -m dte_backend validate examples/run_spec.json
python hooks/dte_guard.py spec examples/run_spec.json
python -m dte_backend judge-oracle --nodes examples/frontier_nodes.json --judge-command "python examples/mock_judge_adapter.py"
python -m dte_backend relation-oracle --nodes examples/frontier_nodes.json --relation-command "python examples/mock_relation_adapter.py"
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
