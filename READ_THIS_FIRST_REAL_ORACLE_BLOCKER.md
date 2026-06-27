# READ THIS FIRST: real-oracle bridge status

The repository has a locked slash-command entrypoint:

```bash
python -m dte_backend strict-run --mode smoke|dry-run|real ...
```

`strict-run` is the correct entrypoint for `/dte-extreme-research`. The older `run` command is a flexible backend helper and must not be used as the slash-command entrypoint.

## Current status

The real Codex Judge command now exists:

```bash
python scripts/codex_judge_adapter.py
```

Use it with real mode:

```bash
python -m dte_backend strict-run \
  --mode real \
  --spec <run_spec.json> \
  --out-dir artifacts/session \
  --cache-path .dte_cache/cache.json \
  --judge-command "python scripts/codex_judge_adapter.py"
```

The adapter:

1. receives the Judge task payload on stdin;
2. builds the prompt in this order: `prompts/DTE_STATIC_PREFIX.md -> prompts/judge_oracle.md -> dynamic JSON`;
3. calls `codex exec` by default;
4. accepts only valid Judge JSON;
5. normalizes and prints `{"results": [...]}` for the backend validator.

Set `DTE_CODEX_JUDGE_COMMAND` only if the local Codex command must be overridden. The override command must read the prompt from stdin and print valid Judge JSON.

## Remaining work

The remaining work is not DTE architecture. Continue by hardening workflow edges:

- decide whether to add matching real Codex Executor and Relation adapter commands, or keep those as main-agent mediated steps;
- wire artifact-boundary guards where the Codex runtime supports them;
- record Codex cached-token behavior if the runtime exposes prompt/cache metrics.

## Do not do these

- Do not redesign DTE.
- Do not restore mandatory Distiller.
- Do not introduce LangChain/LangGraph as a required dependency.
- Do not make UCB cost-aware by default.
- Do not use `examples/mock_judge_adapter.py` for real research.
- Do not use `examples/mock_relation_adapter.py` for real research.
- Do not use hash embedding as real geometry.
- Do not put dynamic task/user/repo/log content before `prompts/DTE_STATIC_PREFIX.md`.
- Do not produce a final report directly from a subagent.

## Boundary guards

Run artifact-boundary guards before consuming machine-facing outputs:

```bash
python hooks/dte_guard.py spec <run_spec.json>
python hooks/dte_guard.py judge --nodes <nodes.json> --output <judge_output.json>
python hooks/dte_guard.py relation --nodes <nodes.json> --output <relation_output.json>
python hooks/dte_guard.py executor --parent <parent.json> --output <executor_output.json> --child-count <n>
```

## Validation

```bash
python -m pip install -e .[dev]
pytest
python scripts/smoke_workflow.py
```

Then run real mode with the Judge command above when `GEMINI_API_KEY` or `GOOGLE_API_KEY` and Codex CLI auth are available.
