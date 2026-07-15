# Codex App Workflow for DTE

This document describes how Codex should expose the DTE backend without becoming its outer controller. The Codex app and Markdown artifacts are an observation surface; the Python backend owns the state machine.

## Production boundary

The compatible headless production entrypoint is:

```bash
python -m dte_backend strict-run \
  --mode real \
  --spec <run_spec.json> \
  --out-dir artifacts/session \
  --cache-path .dte_cache/cache.json \
  --judge-command "python scripts/codex_judge_adapter.py"
```

In Codex App / Work, the normal native production path is the persistent `create-run` / `next-episode` / current-App work / `submit-episode-result` loop. Repository code does not launch a second Codex process.

The model-facing main agent must not manually replay Judge → controller → Executor → Relation, hand-compute controller fields, claim algorithmic convergence, mutate the graph, or commit synthesis. It may supervise the production backend and issue a synthesis request through its validated command interface when `OperatorPolicy` authorizes it. Standalone role commands and guards are development interfaces, not a second production mode.

The transport-neutral `EpisodeRequest -> EpisodeResult` boundary is backend-enforced for App-native Judge, Executor, and Relation roles. A free-form model-orchestrated harness remains non-production because it bypasses the persistent lifecycle and commit boundary.

## Control ownership

| Component | May do | Must not do |
|---|---|---|
| DTE backend | validate the RunSpec; own graph state; invoke bounded adapters; compute Judge/controller state, entropy, UCB, and allocation; apply accepted mutations; stop; select and commit synthesis | delegate outer-control decisions to a model-facing agent |
| Main agent | start and supervise the backend; display and summarize artifacts; ask user questions; submit a synthesis request when `OperatorPolicy` authorizes it | advance the state machine outside the backend; allocate; directly mutate graph/controller state; claim convergence; commit synthesis |
| Judge adapter | return validated observable scores, reasoning, and risks | allocate, mutate, expand, or stop |
| Executor adapter | return validated child SearchNodes for a granted expansion | pre-fill controller fields, mutate graph storage, or synthesize |
| Relation adapter | return a validated semantic classification or discriminator proposal | apply a merge or delete graph state |
| User | delegate operator authority and explicitly request interruption after reviewing observable state | impersonate controller convergence; an interruption remains distinct from entropy convergence |

## Smoke and dry-run checks

Install the project and run the local smoke workflow before serious use or after code changes:

```bash
python -m pip install -e .[dev]
python scripts/smoke_workflow.py
```

Mock adapters and hash geometry are allowed only for smoke or explicitly degraded dry-run checks. They are not research judgments.

```bash
python -m dte_backend strict-run \
  --mode smoke \
  --spec examples/run_spec.json \
  --out-dir artifacts/smoke-strict \
  --judge-command "python examples/mock_judge_adapter.py"
```

Real mode requires a real Judge command, a cache path, Gemini geometry at dimension 3072, and `GEMINI_API_KEY` or `GOOGLE_API_KEY`.

## Observation surface

During a run, Codex may read and summarize:

- `checkpoint_summary.md`;
- `main_agent_status.md`;
- `frontier.md`;
- `entropy_trace.md`;
- `strict_run_status.json`;
- `relation_candidates.md`;
- `human_questions.md`.

App-native Relation state is persisted separately under `<run-dir>/relations/` as candidate, relation-ledger, and synthesis-readiness JSON artifacts. They are controller-owned observations; direct file editing has no commit effect.

```text
observation != authority
```

Reading these files does not itself grant authority. The main agent may supervise or issue a synthesis request only as an operator proxy authorized by the validated `OperatorPolicy`, and only the backend may advance, allocate, apply graph changes, stop, or commit synthesis.

Authority is not derived from observation. It is granted by the validated run policy and exercised through backend commands:

```text
delegation + policy + validated command = authority
```

## Validated synthesis request

`strict-run` polls `<out-dir>/strict_run_control.json` by default; `--control-path` may select another operator-controlled path. An authorized main-agent request is:

```json
{
  "action": "force_synthesis_after_current_task",
  "requested_by": "main_agent",
  "reason": "operator proxy found sufficient coverage for synthesis",
  "scope": "all"
}
```

Targeted synthesis uses `"scope": "node_ids"` and a non-empty `"node_ids"` list. The checked-in contract is `schemas/synthesis_control_request.schema.json`, with `examples/synthesis_control_request.json` as the canonical example.

`requested_by` identifies the actor for audit; `operator_policy` determines authorization. The JSON field does not prove who wrote it or create authority by itself. This phase trusts the root/operator execution context invoking the backend. Stronger actor/capability isolation belongs to a future external DTE Driver.

The backend polls only at safe points:

1. after a complete Judge/EvolutionController/allocation checkpoint; or
2. after an already-started node expansion has returned complete Executor output, passed validation, and committed that node-level result.

It does not kill an in-flight oracle, consume partial output, skip an Executor guard, or interpret the request as convergence. A direct user request records `user_interrupted_for_synthesis`; an authorized main-agent request records `main_agent_requested_synthesis`. Neither is `entropy_plateau`, and the two actors are never silently remapped.

## Cache discipline

There are two separate cache layers:

- Model-serving prefix cache: keep `prompts/DTE_STATIC_PREFIX.md` first, the role prompt second, and dynamic JSON last.
- DTE semantic cache: key embeddings and Judge results by canonical node content plus provider/model/dimension/rubric/prompt/schema namespaces.

Do not put volatile logs, timestamps, paths, git state, controller metrics, or transient summaries into semantic identity. Real runs should use one explicit `--cache-path` for the project/session.

## Adapter and guard development

These commands validate individual boundaries for development, fixtures, and smoke checks:

```bash
python hooks/dte_guard.py spec examples/run_spec.json
python hooks/dte_guard.py judge --nodes examples/frontier_nodes.json --output examples/judge_output.json
python hooks/dte_guard.py relation --nodes examples/frontier_nodes.json --output examples/relation_output.json
python hooks/dte_guard.py executor --parent examples/executor_parent.json --output examples/executor_output.json --child-count 1
```

For App-native Relation output, pass the exact granted request instead of a free-standing node list:

```bash
python hooks/dte_guard.py relation --request <request.json> --output <result.json>
```

The standalone `judge-oracle`, `relation-oracle`, `validate-executor`, and flexible `run` helpers are not production outer controllers. A main agent must not compose them into a manual real-run state machine.

## Handoff after completion

After the backend stops and commits synthesis, summarize:

- why the DTE controller stopped, or that the user interrupted;
- the selected and rejected branches;
- unresolved risks and human questions;
- run mode, budget, geometry, adapters, cache path, and any degraded fallback;
- `report.md` as the backend-committed synthesis.

Do not present a checkpoint, standalone oracle output, or manually assembled subagent summary as the final DTE report.
