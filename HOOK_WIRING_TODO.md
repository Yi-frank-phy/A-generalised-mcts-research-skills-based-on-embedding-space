# HOOK_WIRING_TODO.md

This repo now has a locked slash-command entrypoint:

```bash
python -m dte_backend strict-run --mode smoke|dry-run|real ...
```

`strict-run` is the default entrypoint for `/dte-extreme-research` and is documented in `SKILL.md`. The flexible `run` command remains a backend helper, not the slash-command entrypoint.

Do not redesign the architecture.
Do not rewrite DTE.
Do not restore mandatory Distiller.
Do not change UCB to cost-aware by default.
Do not move dynamic task content before `prompts/DTE_STATIC_PREFIX.md`.
Do not use mock adapters in real mode.

## Existing guard commands

```bash
python hooks/dte_guard.py spec examples/run_spec.json
```

```bash
python hooks/dte_guard.py judge --nodes examples/frontier_nodes.json --output <judge_output.json>
```

```bash
python hooks/dte_guard.py relation --nodes examples/frontier_nodes.json --output <relation_output.json>
```

```bash
python hooks/dte_guard.py executor --parent <parent.json> --output <executor_output.json> --child-count <n>
```

## What strict-run already enforces

- smoke mode may use mock adapters and hash geometry;
- dry-run mode requires cache path and must be treated as degraded;
- real mode requires cache path;
- real mode forbids mock Judge;
- real mode forbids missing Judge command;
- real mode forbids hash geometry;
- real mode requires Gemini geometry to use 3072 dimensions;
- real mode requires `GEMINI_API_KEY` or `GOOGLE_API_KEY` when Gemini geometry is selected;
- every strict run writes `strict_run_status.json`.

## Remaining hook wiring target

Wire the Codex runtime hook system so these checks also run at workflow boundaries when artifacts are produced outside the backend CLI:

1. spec guard before any backend run not launched through `strict-run`;
2. judge guard after any raw Judge Oracle output;
3. relation guard after any raw Relation Oracle output;
4. executor guard after any raw Executor output;
5. failed guard prevents consuming that output.

If the Codex hook config format is environment-specific, add the smallest repo-local config or script supported by that environment, and document it in `hooks/README.md`.

The `UserPromptSubmit` hook is only a reminder and cannot replace these artifact-boundary guards.

## Next real-integration task

Provide or wire a real Codex Judge Oracle command for:

```bash
python -m dte_backend strict-run --mode real --judge-command "<real Judge oracle command>" ...
```

The command must follow `prompts/DTE_STATIC_PREFIX.md -> prompts/judge_oracle.md -> dynamic JSON` and return the validated Judge JSON shape.

## Required validation

```bash
python -m pip install -e .[dev]
pytest
python scripts/smoke_workflow.py
```
