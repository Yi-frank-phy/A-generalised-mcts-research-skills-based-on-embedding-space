# Codex App Workflow for DTE

This document describes how the Codex main agent should use this repository as a DTE skill/backend. The Codex app and markdown artifacts are the frontend; do not build a separate web dashboard.

## Roles

- **Main agent**: owns the DTE session, summarizes current state, launches subagents, asks the user short questions when needed, and never bypasses DTE synthesis.
- **Judge Oracle subagent**: scores SearchNodes and returns observable JSON scores/reasoning/risks.
- **Executor subagent**: expands one assigned SearchNode into child SearchNodes.
- **Relation Oracle subagent**: classifies selected node pairs/sets as equivalent, complementary, conflict, or independent.
- **EvolutionController**: Python backend that computes embedding/KDE/entropy/temperature/UCB/Boltzmann allocation.

## Start a run

1. Create or edit a `DTERunSpec`.
2. Validate it:

```bash
python hooks/dte_guard.py spec examples/run_spec.json
```

3. Run the backend with a cache:

```bash
python -m dte_backend run \
  --spec examples/run_spec.json \
  --out-dir artifacts/session \
  --cache-path .dte_cache/cache.json
```

For real Gemini geometry, set `GEMINI_API_KEY` in the environment and use:

```json
"embedding_provider": "gemini-embedding-2",
"embedding_dimension": 3072
```

## Judge Oracle subagent

Use `prompts/judge_oracle.md`. The subagent must return JSON only.

Smoke command:

```bash
python -m dte_backend judge-oracle \
  --nodes examples/frontier_nodes.json \
  --judge-command "python examples/mock_judge_adapter.py"
```

Integrated run command:

```bash
python -m dte_backend run \
  --spec examples/run_spec.json \
  --out-dir artifacts/judge-session \
  --cache-path .dte_cache/cache.json \
  --judge-command "python examples/mock_judge_adapter.py"
```

A real Codex Judge subagent should follow the same JSON contract as the mock adapter.

## Executor subagent

Use `prompts/executor_subagent.md`. The executor receives an `ExpansionRequest` and returns child `SearchNode` objects.

After a subagent returns output, validate it before consumption:

```bash
python hooks/dte_guard.py executor \
  --parent examples/executor_parent.json \
  --output examples/executor_output.json \
  --child-count 1
```

## Relation Oracle subagent

Use `prompts/relation_oracle.md`. Do not call relation oracle for every pair. First select candidates using deterministic signals:

- exact normalized-claim duplicates;
- semantically close embeddings;
- near-tied UCB/score branches;
- entropy plateau branches.

The backend helper is `select_relation_candidate_pairs()` in `src/dte_backend/relation_candidates.py`.

Smoke command:

```bash
python -m dte_backend relation-oracle \
  --nodes examples/frontier_nodes.json \
  --relation-command "python examples/mock_relation_adapter.py"
```

After validation, convert the result through `relation_result_to_outputs()` in `src/dte_backend/relation_workflow.py`. The relation oracle itself must not mutate the graph.

## Human questions

If `artifacts/session/human_questions.md` contains a question, the main agent should ask the user in chat. Do not invent an answer silently. Keep the question short and branch-oriented.

## Main agent summary loop

After each run, summarize these files to the user:

- `main_agent_status.md`: current state and search phase;
- `frontier.md`: active frontier branches;
- `entropy_trace.md`: entropy/temperature and stop reason;
- `human_questions.md`: whether user input is needed;
- `report.md`: DTE synthesis.

The main agent should summarize what happened, why the controller continued or stopped, which nodes were expanded, and whether another run is needed.

## Full smoke workflow

```bash
python scripts/smoke_workflow.py
```

This checks spec guard, Judge oracle, Relation oracle, DTE run with Judge command, and required artifact generation.
