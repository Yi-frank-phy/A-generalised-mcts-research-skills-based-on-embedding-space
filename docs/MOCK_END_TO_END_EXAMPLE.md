# Mock End-to-End DTE Example

This example records the safe mock workflow used for protocol checks. Mock adapters are only valid in smoke mode or tests. Real research must replace them with Codex Judge, Executor, and Relation subagents that preserve the same JSON contracts.

## Subagent Call Examples

- Judge transcript: `examples/subagent_transcripts/judge_call.json`
- Executor transcript: `examples/subagent_transcripts/executor_call.json`
- Relation transcript: `examples/subagent_transcripts/relation_call.json`

Each transcript keeps the stable prompt prefix first:

```text
prompts/DTE_STATIC_PREFIX.md
role-specific prompt
dynamic_json_payload
```

The dynamic payload is intentionally last so Codex backend prefix caching can reuse the static role contract. The subagent response is JSON only and can be passed to the existing guard validators before the backend consumes it.

## Command Sequence

```bash
export DTE_ALLOW_MOCK_ADAPTER=1
python hooks/dte_guard.py spec examples/run_spec.json
python -m dte_backend judge-oracle --nodes examples/frontier_nodes.json --judge-command "python examples/mock_judge_adapter.py"
python -m dte_backend relation-oracle --nodes examples/frontier_nodes.json --relation-command "python examples/mock_relation_adapter.py"
python -m dte_backend relation-artifacts --nodes examples/frontier_nodes.json --relation-output examples/relation_output.json --out-dir artifacts/smoke-workflow/relation
python -m dte_backend strict-run --mode smoke --spec examples/run_spec.json --out-dir artifacts/smoke-workflow --cache-path .dte_cache/smoke_cache.json --judge-command "python examples/mock_judge_adapter.py"
```

The repository wrapper `python scripts/smoke_workflow.py` runs this sequence with `DTE_ALLOW_MOCK_ADAPTER=1`.

## Artifacts To Inspect

After the smoke run, the main agent should summarize:

- `artifacts/smoke-workflow/main_agent_status.md`
- `artifacts/smoke-workflow/frontier.md`
- `artifacts/smoke-workflow/entropy_trace.md`
- `artifacts/smoke-workflow/relation_candidates.md`
- `artifacts/smoke-workflow/human_questions.md`
- `artifacts/smoke-workflow/report.md`

Relation conversion artifacts are written under `artifacts/smoke-workflow/relation/`.

Codex backend cached-token metrics are not emitted by this repository's CLI artifacts. The smoke run does write DTE backend cache information to `artifacts/smoke-workflow/cache_stats.json`, which is a project semantic-cache signal, not an LLM prefix-cache metric.

## Boundary Notes

- Judge output contains scores, reasoning, and risks only.
- Executor output contains child `SearchNode` objects only; it must not pre-fill score, uncertainty, UCB, expansion budget, or final synthesis.
- Relation output classifies selected candidate pairs only; it must not directly merge or mutate graph state.
- Mock adapters are not research oracles. They exist to prove the DTE workflow, guards, and artifacts are wired correctly.
