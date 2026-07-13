---
name: evolving-frontier-research
description: "Use when running structured frontier-search research for deep mathematical, physical, academic, proof, derivation, or conceptual questions through the backend-controlled DTE protocol and Codex App native episode loop."
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

On invocation in Codex App / Work, use the App-native driver loop below. The current main agent performs bounded episodes itself and may use opaque native subagents. Do not use the flexible `run` helper or spawn a second Codex process to simulate native orchestration. `strict-run --mode real` remains the compatible headless/legacy entrypoint.

## Codex App native driver loop

The normal App path is:

```text
start/resume DTE run
-> next-episode
-> perform the bounded request in the current App runtime
-> submit one strict EpisodeResult
-> inspect CommitOutcome/controller action
-> repeat
```

Backend commands:

```bash
python -m dte_backend create-run --run-dir <run-dir> --spec <spec.json> --nodes <committed-nodes.json>
python -m dte_backend next-episode --run-dir <run-dir>
python -m dte_backend submit-episode-result --run-dir <run-dir> --result <result.json>
python -m dte_backend run-status --run-dir <run-dir>
```

Use `fail-episode`, `cancel-episode`, and `retry-episode` for explicit attempt transitions. A retry must use the newly granted `attempt_id`; late output from cancelled, expired, failed, superseded, rejected, or committed attempts cannot be resubmitted as success.

The main agent may reason, use tools, and delegate native subagents inside the request. It must not choose global allocation, hand-fill controller fields, directly edit graph state, skip submit validation, or substitute a chat/Markdown answer for committed output. Keep progress concise. Do not expose or reconstruct internal subagent count, names, routing, transcripts, hidden reasoning, tokens, or quota. App usage telemetry is `unavailable` unless the platform directly supplies it.

`next-episode` may return a strict `role=judge` request for ordinary unscored frontier nodes. Judge every granted node exactly once and return only observable score, reasoning, risks, and optional uncertainty evidence. After a valid Judge submit, call `next-episode` again: the backend, not the main agent, computes embedding/KDE density, entropy, uncertainty, UCB, and allocation before returning an Executor grant or a terminal controller action. Normal App operation must not require the main agent to interpret `continue_controller`.

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

## Headless strict-run compatibility protocol

1. Read this `SKILL.md`, then `AGENTS.md` when repository-specific operating rules are needed.
2. If not installed, run `python -m pip install -e .[dev]`.
3. Run `python scripts/smoke_workflow.py` once before serious use or after repo changes. This is the only default place where mock adapters are allowed.
4. Convert the user task into a run specification.
5. Use `embedding_provider=gemini-embedding-2` and `embedding_dimension=3072` for real mode. Hash embedding is allowed only in `--mode smoke` or `--mode dry-run`.
6. Always provide `--cache-path .dte_cache/cache.json` outside smoke mode.
7. For a headless real research run, call `python -m dte_backend strict-run --mode real` with `--judge-command "python scripts/codex_judge_adapter.py"`. Do not use the mock Judge. This legacy subprocess route is not the Codex App native path.
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

### Strict-run visibility and operator commands

For strict-run sessions, the main agent is a user-delegated operator proxy. It may monitor DTE artifacts, start and supervise the backend, and submit a synthesis request through the validated control interface when `DTERunSpec.operator_policy.main_agent_may_request_synthesis` is true. It may not directly mutate graph state, controller fields, allocation, stop metadata, or the synthesis result.

```text
observation != authority
delegation + policy + validated command = authority
```

Base any recommendation on DTE-generated task state, not on a free-form hunch. Use the latest available `checkpoint_summary.md` / task summary, including the run objective, iteration, entropy state, frontier claims, Judge scores, uncertainty, UCB/allocation, risks, relation candidates, and unresolved human questions.

`strict-run` polls `<out-dir>/strict_run_control.json` by default; an external caller/operator may select another location with `--control-path <path>`. A request by the main-agent operator proxy is:

```json
{
  "action": "force_synthesis_after_current_task",
  "requested_by": "main_agent",
  "reason": "checkpoint has sufficient coverage for synthesis",
  "scope": "all"
}
```

For targeted synthesis, use `"scope": "node_ids"` with a non-empty `"node_ids"` list. The backend reads this file only after the complete current Judge/controller checkpoint or after an already-started node expansion has returned complete output and passed Executor validation. It does not kill an in-flight oracle subprocess or consume partial output.

`requested_by` identifies the actor for audit; `operator_policy` determines whether that actor is authorized. The field does not authenticate the writer or create permission by itself. The current protocol trusts the root/operator execution context invoking the backend; stronger actor/capability isolation is deferred to a future external DTE Driver.

Record the two actors distinctly:

```json
"stop_reason": "main_agent_requested_synthesis"
```

or `user_interrupted_for_synthesis` for a direct user request. Neither is `entropy_plateau` or algorithmic convergence. The final report must cite the checkpoint/task summary used for the decision and state which frontier branches may have been left unexplored.

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

Keep `allocation_mass_per_iteration` and `max_children_per_iteration` explicit for the task. Do not set `min_iterations_before_synthesis` equal to `max_iterations`; the entropy stop condition must have room to operate before the hard cap.

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

This flow is owned by `strict-run --mode real` in production.

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

## Relation adapter development

`relation_candidates.md` is observable backend state. In production, only a backend-controlled Relation adapter may consume a candidate and return a result to the DTE state machine. Until that integration is backend-enforced, the main agent must report unresolved candidates rather than manually extending the run and presenting the result as DTE synthesis. Do not use the mock Relation adapter for research judgment.

For adapter development and smoke validation, convert a validated relation result to machine artifacts with:

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

`strict-run` performs the core pre-run policy checks and owns the production call sequence. The standalone guard commands are for adapter development, fixtures, and smoke validation; they do not define a second real-run mode.

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
- reproducibility metadata: run spec, budget, strict mode (`real`, `smoke`, or `dry-run`), embedding provider, Judge/Executor/Relation backends, cache path, and whether any dry-run fallback was used.
- if an operator requested synthesis before natural convergence: the actor-specific stop reason, the checkpoint/task summary used, and the frontier branches left unexplored.

## Prohibited behavior

- Do not use the flexible `run` helper or a model-orchestrated manual harness as the slash-command entrypoint; use `strict-run`.
- Do not use mock adapters for real research judgment.
- Do not return a final answer directly from an executor episode.
- Do not silently skip Judge or controller stages.
- Do not let a subagent decide entropy convergence, UCB, allocation, or the stop condition.
- Do not call user- or main-agent-requested synthesis `entropy_plateau`.
- Do not let the main agent infer authority from checkpoint access or directly mutate controller-owned state; require `OperatorPolicy` and a validated controller command.
- Do not invent a separate HIL workflow when the user is already supervising the main agent through chat interruption.
- Do not replace role isolation with a single all-in-one agent.
- Do not modify UCB to be cost-aware by default.
- Do not rely on free-form Markdown as machine truth.
- Do not let executor adapters return synthesis nodes or pre-filled Judge/Evolution metrics.
- Do not treat Judge as an embedding model; Judge is a closed oracle that returns observable judgments only.
- Do not let relation-oracle output directly rewrite graph state before validation.
- Do not place dynamic content before `prompts/DTE_STATIC_PREFIX.md` in subagent prompts.
- In Codex App / Work, do not launch `codex exec`, SDK/App Server, or another Codex process as the normal native episode path; the current App main agent performs the episode.
- Do not require hidden App subagent topology, traces, token usage, or quota before accepting an otherwise valid structured result.
