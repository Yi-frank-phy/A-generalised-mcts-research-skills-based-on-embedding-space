# Codex App Workflow for DTE

This document describes how the Codex main agent should use this repository as a DTE skill/backend. The Codex app and markdown artifacts are the frontend; do not build a separate web dashboard.

## Recommended invocation when Codex Skills are available

Point Codex at this repository/skill and ask it to run DTE, not to redesign DTE:

```text
Use the dte-extreme-research skill/backend in this repository.
Read AGENTS.md, SKILL.md, CODEX_NEXT_STEPS.md, and this workflow document.
For subagent prompts, place prompts/DTE_STATIC_PREFIX.md first, then the role prompt, then dynamic JSON last.
Run the smoke workflow first.
Then run the requested research task through DTE with cache enabled.
Summarize main_agent_status.md, frontier.md, entropy_trace.md, relation_candidates.md, human_questions.md, and report.md after each run.
Do not bypass DTE synthesis.
```

Minimum command sequence:

```bash
python -m pip install -e .[dev]
python scripts/smoke_workflow.py
DTE_ALLOW_MOCK_ADAPTER=1 \
python -m dte_backend run \
  --spec examples/run_spec.json \
  --out-dir artifacts/session \
  --cache-path .dte_cache/cache.json \
  --judge-command "python examples/mock_judge_adapter.py"
```

Direct mock-adapter commands require `DTE_ALLOW_MOCK_ADAPTER=1`; the wrapper `python scripts/smoke_workflow.py` sets it for smoke checks. Do not set it for real research.

For real geometry, set `GEMINI_API_KEY` or `GOOGLE_API_KEY` and use `embedding_provider=gemini-embedding-2`, `embedding_dimension=3072`.

## Roles

- **Main agent**: owns the DTE session, summarizes current state, launches subagents, asks the user short questions when needed, and never bypasses DTE synthesis.
- **Judge Oracle subagent**: scores SearchNodes and returns observable JSON scores/reasoning/risks.
- **Executor subagent**: expands one assigned SearchNode into child SearchNodes.
- **Relation Oracle subagent**: classifies selected node pairs/sets as equivalent, complementary, conflict, or independent.
- **EvolutionController**: Python backend that computes embedding/KDE/entropy/temperature/UCB/Boltzmann allocation.

## Two different caches

There are two cache layers. Do not confuse them.

### 1. LLM prefix cache / Codex backend context cache

This is the model-serving cache. It depends on exact shared prompt prefixes. To maximize hits:

- put `prompts/DTE_STATIC_PREFIX.md` first, byte-for-byte;
- put role-specific prompt second;
- put dynamic task JSON, node content, logs, and user request last;
- keep schemas and tool descriptions stable;
- avoid putting repo status, timestamps, git SHAs, paths, or human messages before the static prefix;
- reuse the same role prompt family across Judge, Executor, and Relation calls.

The helper `build_cached_subagent_prompt()` in `src/dte_backend/prompt_builder.py` builds prompts in this order.

### 2. DTE backend semantic cache

This is the project-owned cache for embeddings and Judge scores. It is keyed by canonical node envelopes, not raw prompt prefixes.

Use stable context envelopes:

- preserve `claim`, explicit assumptions, evidence, counterexamples, and risks;
- drop temporary logs, timestamps, stdout/stderr, ids, paths, tool chatter, and style-only rewrites;
- do not include parent ids, score, UCB, status, or expansion budget in semantic cache identity;
- use the same `--cache-path` across runs for one project/session;
- embed node summaries, not full transcripts.

The backend implements this through `context_envelope.py`, split embedding/Judge cache keys, canonical embedding input, and file-backed cache.

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

Use `prompts/DTE_STATIC_PREFIX.md` first, then `prompts/judge_oracle.md`, then dynamic input JSON. The subagent must return JSON only.
Validate Judge output before consuming scores. The `judge-oracle` CLI command runs the same validator; for outputs produced outside the CLI, run the guard explicitly:

```bash
python hooks/dte_guard.py judge \
  --nodes examples/frontier_nodes.json \
  --output examples/judge_output.json
```

Smoke command:

```bash
DTE_ALLOW_MOCK_ADAPTER=1 \
python -m dte_backend judge-oracle \
  --nodes examples/frontier_nodes.json \
  --judge-command "python examples/mock_judge_adapter.py"
```

Integrated run command:

```bash
DTE_ALLOW_MOCK_ADAPTER=1 \
python -m dte_backend run \
  --spec examples/run_spec.json \
  --out-dir artifacts/judge-session \
  --cache-path .dte_cache/cache.json \
  --judge-command "python examples/mock_judge_adapter.py"
```

A real Codex Judge subagent should follow the same JSON contract as the mock adapter. See `examples/subagent_transcripts/judge_call.json` for a concrete Codex-style transcript with the shared static prefix first and dynamic node JSON last.

## Executor subagent

Use `prompts/DTE_STATIC_PREFIX.md` first, then `prompts/executor_subagent.md`, then dynamic ExpansionRequest JSON. The executor receives an `ExpansionRequest` and returns child `SearchNode` objects.

After a subagent returns output, validate it before consumption:

```bash
python hooks/dte_guard.py executor \
  --parent examples/executor_parent.json \
  --output examples/executor_output.json \
  --child-count 1
```

See `examples/subagent_transcripts/executor_call.json` for a concrete Codex-style transcript. The response contains child `SearchNode` objects only and does not include Judge or controller metrics.

## Relation Oracle subagent

Use `prompts/DTE_STATIC_PREFIX.md` first, then `prompts/relation_oracle.md`, then dynamic relation-task JSON. Do not call relation oracle for every pair. First select candidates using deterministic signals:

- exact normalized-claim duplicates;
- semantically close embeddings;
- near-tied UCB/score branches;
- entropy plateau branches.

Read `relation_candidates.md` after each run. The backend helper is `select_relation_candidate_pairs()` in `src/dte_backend/relation_candidates.py`.
Validate Relation output before converting it into merge proposals or discriminator tasks. The `relation-oracle` CLI command runs the same validator; for outputs produced outside the CLI, run the guard explicitly:

```bash
python hooks/dte_guard.py relation \
  --nodes examples/frontier_nodes.json \
  --output examples/relation_output.json
```

Smoke command:

```bash
python -m dte_backend relation-oracle \
  --nodes examples/frontier_nodes.json \
  --relation-command "python examples/mock_relation_adapter.py"
```

After validation, convert the result through `relation_result_to_outputs()` in `src/dte_backend/relation_workflow.py`, or write machine artifacts with:

```bash
python -m dte_backend relation-artifacts \
  --nodes examples/frontier_nodes.json \
  --relation-output examples/relation_result.json \
  --out-dir artifacts/relation
```

The relation oracle itself must not mutate the graph. See `examples/subagent_transcripts/relation_call.json` for a concrete Codex-style transcript.

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

This checks spec guard, Judge oracle, Relation oracle, DTE run with Judge command, relation artifact conversion, and required artifact generation.

For a documented end-to-end mock example, see `docs/MOCK_END_TO_END_EXAMPLE.md`. It references the transcript fixtures and the standard smoke artifacts that the main agent should inspect.
