# Codex App Workflow for DTE

This document describes how the Codex main agent should use this repository as a DTE skill/backend. The Codex app and markdown artifacts are the frontend; do not build a separate web dashboard.

## Recommended invocation when Codex Skills are available

Point Codex at this repository/skill and ask it to run DTE, not to redesign DTE:

```text
Use the dte-extreme-research skill/backend in this repository.
Read AGENTS.md, SKILL.md, CODEX_NEXT_STEPS.md, and this workflow document.
Run the smoke workflow first.
Then run the requested research task through DTE with cache enabled.
Summarize main_agent_status.md, frontier.md, entropy_trace.md, relation_candidates.md, human_questions.md, and report.md after each run.
Do not bypass DTE synthesis.
```

Minimum command sequence:

```bash
python -m pip install -e .[dev]
python scripts/smoke_workflow.py
python -m dte_backend run \
  --spec examples/run_spec.json \
  --out-dir artifacts/session \
  --cache-path .dte_cache/cache.json \
  --judge-command "python examples/mock_judge_adapter.py"
```

For real geometry, set `GEMINI_API_KEY` or `GOOGLE_API_KEY` and use `embedding_provider=gemini-embedding-2`, `embedding_dimension=3072`.

## Roles

- **Main agent**: owns the DTE session, summarizes current state, launches subagents, asks the user short questions when needed, and never bypasses DTE synthesis.
- **Judge Oracle subagent**: scores SearchNodes and returns observable JSON scores/reasoning/risks.
- **Executor subagent**: expands one assigned SearchNode into child SearchNodes.
- **Relation Oracle subagent**: classifies selected node pairs/sets as equivalent, complementary, conflict, or independent.
- **EvolutionController**: Python backend that computes embedding/KDE/entropy/temperature/UCB/Boltzmann allocation.

## Cache-friendly context policy

Do not optimize context by making every subagent invent a fresh shortest possible summary. That was useful in older context-engineering workflows, but it causes high cache miss rates.

Use stable context envelopes:

- preserve `claim`, explicit assumptions, evidence, counterexamples, and risks;
- drop temporary logs, timestamps, stdout/stderr, ids, paths, tool chatter, and style-only rewrites;
- do not include parent ids, score, UCB, status, or expansion budget in semantic cache identity;
- use the same `--cache-path` across runs for one project/session;
- embed node summaries, not full transcripts.

The backend implements this through `context_envelope.py`, split embedding/Judge cache keys, and canonical embedding input.

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

Read `relation_candidates.md` after each run. The backend helper is `select_relation_candidate_pairs()` in `src/dte_backend/relation_candidates.py`.

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
- `relation_candidates.md`: node pairs worth Relation Oracle classification;
- `human_questions.md`: whether user input is needed;
- `report.md`: DTE synthesis.

The main agent should summarize what happened, why the controller continued or stopped, which nodes were expanded, and whether another run is needed.

## Full smoke workflow

```bash
python scripts/smoke_workflow.py
```

This checks spec guard, Judge oracle, Relation oracle, DTE run with Judge command, and required artifact generation.
