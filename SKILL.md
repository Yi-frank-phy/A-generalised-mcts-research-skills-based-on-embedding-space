---
name: evolving-frontier-research
description: "Use when running structured frontier-search research for deep mathematical, physical, academic, proof, derivation, or conceptual questions with DTE protocol, strict-run, or Codex App fast subagent orchestration."
---

# Evolving Frontier Research Skill

This file is the primary runtime instruction for slash-command usage. A Codex agent invoking this skill should not require the user to read separate docs before the first run.

The old internal shorthand may appear in Python module names such as `dte_backend`, but the public skill name is **Evolving Frontier Research Skill**.

## Slash-command intent

Use this skill when the user invokes a structured deep-research command, asks for extreme-depth research, wants a Codex-backed research agent, or wants a structured mathematical/physical/academic exploration rather than a direct one-shot answer.

Typical invocation:

```text
/evolving-frontier-research <research task>
```

On invocation, Codex should use the locked DTE workflow, not the flexible `run` helper, unless the user explicitly asks only for explanation or planning. The default reproducible entrypoint is `python -m dte_backend strict-run`; when the user asks for lower latency or visible Codex App progress and current-session subagents are available, Codex may use the `app-orchestrated-real` workflow below.

## Critical real-run rule

Mock adapters are smoke-test tools only. They are not research Judges, not research Relation Oracles, and not acceptable for real slash-command runs.

Forbidden in real research runs:

```bash
--judge-command "python examples/mock_judge_adapter.py"
--relation-command "python examples/mock_relation_adapter.py"
```

The mock adapters are blocked by default and only run when `DTE_ALLOW_MOCK_ADAPTER=1` is set by smoke tests. If a real Codex Judge/Relation/Executor subagent is unavailable, Codex must either:

1. run only a clearly labelled `--mode smoke` or `--mode dry-run`; or
2. stop and tell the user that the real oracle layer is not available.

It must not present mock-oracle output as research judgment.

## One-screen execution protocol

1. Read this `SKILL.md`, then `AGENTS.md` when repository-specific operating rules are needed.
2. If not installed, run `python -m pip install -e .[dev]`.
3. Run `python scripts/smoke_workflow.py` once before serious use or after repo changes. This is the only default place where mock adapters are allowed.
4. Convert the user task into a run specification.
5. Use `embedding_provider=gemini-embedding-2` and `embedding_dimension=3072` for real mode. Hash embedding is allowed only in `--mode smoke` or `--mode dry-run`.
6. Always provide `--cache-path .dte_cache/cache.json` outside smoke mode.
7. For a fully machine-reproducible real research run, call `strict-run --mode real` with `--judge-command "python scripts/codex_judge_adapter.py"`. For a lower-latency Codex App run, use `app-orchestrated-real` exactly as specified below. Do not use the mock Judge.
8. Summarize these artifacts after each run:
   - `main_agent_status.md`
   - `frontier.md`
   - `entropy_trace.md`
   - `relation_candidates.md`
   - `human_questions.md`
   - `report.md`
   - `strict_run_status.json`
9. If `human_questions.md` asks the user a branch question, ask it in chat instead of guessing.
10. Final answer must come from validated synthesis, not directly from any subagent.

## Strict run modes

### Smoke mode

Use only to verify machinery:

```bash
python -m dte_backend strict-run \
  --mode smoke \
  --spec examples/run_spec.json \
  --out-dir artifacts/smoke-strict \
  --judge-command "python examples/mock_judge_adapter.py"
```

Smoke results are not research judgments.

### Dry-run mode

Use when the real oracle or Gemini geometry is unavailable. Dry-run output must be labelled as degraded:

```bash
python -m dte_backend strict-run \
  --mode dry-run \
  --spec examples/run_spec.json \
  --out-dir artifacts/dry-run \
  --cache-path .dte_cache/cache.json
```

### Real mode

Use for actual slash-command research:

```bash
python -m dte_backend strict-run \
  --mode real \
  --spec <run_spec.json> \
  --out-dir artifacts/session \
  --cache-path .dte_cache/cache.json \
  --judge-command "python scripts/codex_judge_adapter.py"
```

Real mode refuses mock adapters, missing Judge oracle, hash geometry, missing cache path, or missing Gemini key when Gemini geometry is selected.

The adapter calls `codex exec` by default and asks the model to return Judge JSON only. Set `DTE_CODEX_JUDGE_COMMAND` only to override the Codex command; the override must read the prompt from stdin and print JSON that passes the Judge output validator.

### Strict-run visibility and main-agent synthesis

For strict-run sessions, the main agent may monitor DTE artifacts and recommend or request synthesis when further expansion has low expected value. This is not a separate HIL system: the user already collaborates with the main agent in chat and may interrupt the run or redirect the main agent directly.

The main agent must base any synthesis recommendation on DTE-generated task state, not on a free-form hunch. Use the latest available `checkpoint_summary.md` / task summary, including the run objective, iteration, entropy state, frontier claims, Judge scores, uncertainty, UCB/allocation, risks, relation candidates, and unresolved human questions.

`strict-run` watches `<out-dir>/strict_run_control.json` by default; `--control-path <path>` may override it. To request synthesis after the current safe task finishes, write:

```json
{
  "action": "force_synthesis_after_current_task",
  "requested_by": "main_agent",
  "reason": "checkpoint has enough coverage",
  "scope": "all"
}
```

For a direct user interruption, use `"requested_by": "user"`. For targeted synthesis, use `"scope": "node_ids"` with a non-empty `"node_ids"` list. The backend reads this file only after the current Judge/controller checkpoint or current expanded node finishes; it does not kill an in-flight oracle subprocess.

If the main agent or user forces synthesis before natural entropy convergence, record it honestly:

```json
"stop_reason": "main_agent_requested_synthesis"
```

or, when the user directly interrupts:

```json
"stop_reason": "user_interrupted_for_synthesis"
```

Do not relabel this as `entropy_plateau`. The final report must cite the checkpoint/task summary used for the decision and state which frontier branches may have been left unexplored.

### Codex App fast orchestration

Use `app-orchestrated-real` only when the current Codex App session exposes real subagent tools and the user values lower end-to-end latency or in-app progress visibility. This mode is DTE-compatible but is not the same as a one-command `strict-run`; label artifacts and final metadata honestly as `execution_style=app-orchestrated-real`.

Mandatory invariants:

1. The main agent is the orchestrator, not the researcher of record.
2. Subagents may act only as Judge, Executor, or Relation oracles. They must return JSON only and must not decide stopping, allocation, graph mutation, or final synthesis.
3. The DTE backend remains the controller: use the existing embedding/KDE/entropy, temperature, UCB, allocation, merge, and synthesis helpers. Do not hand-compute or ask a subagent to decide these values.
4. The same run spec, SearchNode schema, prompt order, cache discipline, guard commands, and artifact contract apply as in real mode.
5. If a required subagent tool, guard, schema, real embedding provider, or backend controller helper is unavailable, stop or fall back to an explicitly labelled dry-run. Do not silently degrade to free-form research.

Minimal loop:

1. Create `run_spec.json` and initial SearchNodes, then run `python hooks/dte_guard.py spec <run_spec.json>`.
2. For each iteration, build Judge prompts with `build_cached_subagent_prompt("judge", ...)` or the exact static-prefix order, spawn Judge subagents, write `judge_output.iter<N>.json`, then run `python hooks/dte_guard.py judge --nodes <frontier_nodes.json> --output <judge_output.iter<N>.json>`.
3. Feed only validated Judge results into the backend controller. The controller computes embeddings, KDE density, entropy, uncertainty, temperature, UCB, and expansion budgets.
4. Stop only when `evaluate_entropy_state()` says to synthesize after `min_iterations_before_synthesis`, or when `max_iterations` is reached. A subagent cannot stop the run.
5. Spawn Executor subagents only for nodes with positive controller allocation. Write each `executor_output.iter<N>.<node_id>.json` and run `python hooks/dte_guard.py executor --parent <parent.json> --output <executor_output.json> --child-count <allocated_child_count>` before adding children to the graph.
6. Render relation candidates from backend state. If relation candidates are non-empty, spawn Relation subagents only for selected pairs/sets, write `relation_output.iter<N>.json`, run the relation guard, and convert validated results through backend relation helpers.
7. Produce final synthesis only from validated graph state and backend artifacts. Do not copy a Judge, Executor, or Relation conclusion as the final answer.

## Real Gemini geometry

Use max geometry by default:

```json
"embedding_provider": "gemini-embedding-2",
"embedding_dimension": 3072
```

Manual check:

```bash
python scripts/gemini_smoke.py
```

Only run Gemini smoke when `GEMINI_API_KEY` or `GOOGLE_API_KEY` is set. Respect free-tier limits by embedding only node summaries and always using `--cache-path`.

## Default budget profile

When the user has not specified a stricter or cheaper budget, use this balanced research profile:

```json
"budget": {
  "max_iterations": 5,
  "min_iterations_before_synthesis": 3,
  "entropy_change_threshold": 0.01
}
```

Keep `total_child_budget` explicit for the task. Do not set `min_iterations_before_synthesis` equal to `max_iterations`; the entropy stop condition must have room to operate before the hard cap.

## Prefix-cache rule for Codex backend LLM calls

LLM backend prefix caches require stable prompt prefixes. Do not optimize by making every subagent prompt a different shortest summary.

For every Judge, Executor, Relation, or future Synthesis subagent call, build the prompt in this exact order:

```text
prompts/DTE_STATIC_PREFIX.md
↓
role-specific prompt
↓
dynamic JSON input
```

Role-specific prompts:

```text
prompts/judge_oracle.md
prompts/executor_subagent.md
prompts/relation_oracle.md
```

Never put user task text, SearchNode content, retrieved documents, repo status, timestamps, git SHAs, file paths, stdout/stderr, tool logs, or scratchpad summaries before `prompts/DTE_STATIC_PREFIX.md`.

Use this helper when possible:

```python
from dte_backend.prompt_builder import build_cached_subagent_prompt
```

## Backend semantic cache rule

This is separate from LLM prefix cache. The backend caches embeddings and Judge scores using canonical context envelopes.

Preserve in SearchNodes:

- claim;
- explicit assumptions;
- evidence;
- counterexamples;
- risks;
- discriminator questions.

Do not include volatile material in semantic identity:

- parent ids;
- controller metrics;
- score;
- UCB;
- expansion budget;
- status;
- timestamps;
- logs;
- paths;
- transient summaries.

## Required research flow

This flow applies to both `strict-run --mode real` and `app-orchestrated-real`.

1. Generate or ingest initial SearchNodes.
2. Validate all SearchNodes against schema.
3. Score frontier nodes through a real Judge Oracle, or explicitly declare dry-run mode.
4. The controller computes embedding/KDE density, entropy, uncertainty, UCB, temperature, and expansion budgets.
5. Executor expands selected frontier nodes into structured child SearchNodes.
6. Validate executor output before adding it to the graph.
7. Render `relation_candidates.md`.
8. Relation Oracle classifies only selected candidate pairs, not every pair.
9. Convert validated relation output into `MergeProposal` or discriminator task through backend helpers.
10. Produce final synthesis through validated synthesis.

## Oracle definitions

Oracle means a bounded external/subagent judgment function. It is not the controller.

- Judge Oracle: `SearchNodes -> score/reasoning/risks`.
- Relation Oracle: `SearchNodes -> equivalent|complementary|conflict|independent`.
- Executor: `ExpansionRequest -> child SearchNodes`.

Oracles must return machine-readable JSON and must pass validators.

## Relation oracle workflow

After each run, inspect `relation_candidates.md`. If non-empty, Codex may call a real Relation Oracle on the listed pairs. Do not use the mock Relation adapter for research judgment.

Convert validated relation result to machine artifacts:

```bash
python -m dte_backend relation-artifacts \
  --nodes <nodes.json> \
  --relation-output <relation_output.json> \
  --out-dir artifacts/relation
```

Outputs:

```text
relation_proposals.json
discriminator_tasks.json
```

Relation Oracle output must not directly mutate the graph.

## Validation hooks

The repository includes guard commands for machine-facing outputs:

```bash
python hooks/dte_guard.py spec examples/run_spec.json
python hooks/dte_guard.py judge --nodes examples/frontier_nodes.json --output <judge_output.json>
python hooks/dte_guard.py relation --nodes examples/frontier_nodes.json --output <relation_output.json>
python hooks/dte_guard.py executor --parent <parent.json> --output <executor_output.json> --child-count <n>
```

`strict-run` performs the core pre-run policy checks. In `app-orchestrated-real`, the main agent must explicitly write each oracle output to disk and run the matching guard before consuming it.

## Compile behavior

Compile is not a mandatory Distiller phase. Codex or subagents may compile their own local context when useful, but compilation must happen after the stable prompt prefix and must preserve assumptions, evidence, counterexamples, branch conflicts, and uncertainty.

Do not restore a mandatory Distiller role.

## Output contract

The final answer/report must include:

- answer or research report;
- selected search path;
- key rejected alternatives;
- assumptions;
- confidence levels;
- unresolved risks;
- reproducibility metadata: run spec, budget, execution style (`strict-run --mode real`, `app-orchestrated-real`, smoke, or dry-run), embedding provider, Judge/Executor/Relation backends, cache path, and whether any dry-run fallback was used.
- if synthesis was forced before natural convergence: the stop reason, the checkpoint/task summary used, and the frontier branches left unexplored.

## Prohibited behavior

- Do not use the flexible `run` helper as the slash-command entrypoint; use `strict-run` or the explicitly labelled `app-orchestrated-real` workflow.
- Do not call `app-orchestrated-real` a `strict-run` artifact.
- Do not use mock adapters for real research judgment.
- Do not return a final answer directly from an executor episode.
- Do not silently skip Judge or controller stages.
- Do not let a subagent decide entropy convergence, UCB, allocation, or the stop condition.
- Do not call main-agent-requested or user-interrupted synthesis `entropy_plateau`.
- Do not invent a separate HIL workflow when the user is already supervising the main agent through chat interruption.
- Do not replace role isolation with a single all-in-one agent.
- Do not modify UCB to be cost-aware by default.
- Do not rely on free-form Markdown as machine truth.
- Do not let executor adapters return synthesis nodes or pre-filled Judge/Evolution metrics.
- Do not treat Judge as an embedding model; Judge is a closed oracle that returns observable judgments only.
- Do not let relation-oracle output directly rewrite graph state before validation.
- Do not place dynamic content before `prompts/DTE_STATIC_PREFIX.md` in subagent prompts.
