# READ THIS FIRST: real-oracle blocker

The repository now has a locked slash-command entrypoint:

```bash
python -m dte_backend strict-run --mode smoke|dry-run|real ...
```

`strict-run` is the correct entrypoint for `/dte-extreme-research`. The older `run` command is a flexible backend helper and must not be used as the slash-command entrypoint.

## Current blocker

The remaining blocker is not DTE architecture. The blocker is real Codex oracle integration.

`strict-run --mode real` requires a real Judge Oracle command:

```bash
python -m dte_backend strict-run \
  --mode real \
  --spec <run_spec.json> \
  --out-dir artifacts/session \
  --cache-path .dte_cache/cache.json \
  --judge-command "<real Codex Judge Oracle command>"
```

The command must:

1. receive the Judge task payload on stdin;
2. use the prompt order `prompts/DTE_STATIC_PREFIX.md -> prompts/judge_oracle.md -> dynamic JSON`;
3. call a real Codex/LLM Judge subagent or equivalent real oracle;
4. return valid Judge JSON only:

```json
{
  "results": [
    {
      "node_id": "...",
      "score": 0.0,
      "reasoning": "...",
      "risks": []
    }
  ]
}
```

5. pass `hooks/dte_guard.py judge --nodes <nodes.json> --output <judge_output.json>` or the equivalent backend validator.

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

## What Codex should implement next

Implement the smallest repo-local real-oracle bridge supported by the Codex runtime. Acceptable forms include:

- a script under `scripts/` that launches a real Codex Judge subagent and prints validated Judge JSON;
- a hook/config integration that turns a Judge task payload into a real Codex subagent call;
- a documented runtime command that can be passed to `strict-run --mode real --judge-command`.

After that, wire artifact-boundary guards:

```bash
python hooks/dte_guard.py spec <run_spec.json>
python hooks/dte_guard.py judge --nodes <nodes.json> --output <judge_output.json>
python hooks/dte_guard.py relation --nodes <nodes.json> --output <relation_output.json>
python hooks/dte_guard.py executor --parent <parent.json> --output <executor_output.json> --child-count <n>
```

## Validation after implementation

```bash
python -m pip install -e .[dev]
pytest
python scripts/smoke_workflow.py
```

Then run a real-mode check with a real Judge command:

```bash
python -m dte_backend strict-run \
  --mode real \
  --spec <run_spec.json> \
  --out-dir artifacts/real-check \
  --cache-path .dte_cache/cache.json \
  --judge-command "<real Codex Judge Oracle command>"
```

If this cannot be done in the current Codex environment, stop and report the missing runtime capability. Do not fall back to mock output while calling it real research.
