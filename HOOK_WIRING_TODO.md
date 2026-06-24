# HOOK_WIRING_TODO.md

This repo is ready for slash-command skill use through `SKILL.md`. The remaining integration task is to wire existing guard commands into the Codex hook system.

Do not redesign the architecture.
Do not rewrite DTE.
Do not restore mandatory Distiller.
Do not change UCB to cost-aware by default.
Do not move dynamic task content before `prompts/DTE_STATIC_PREFIX.md`.

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

## Hook wiring target

Wire the Codex runtime hook system so these checks run at the correct workflow boundaries:

1. spec guard before backend run;
2. judge guard after Judge Oracle output;
3. relation guard after Relation Oracle output;
4. executor guard after Executor output;
5. failed guard prevents consuming that output.

If the Codex hook config format is environment-specific, add the smallest repo-local config or script supported by that environment, and document it in `hooks/README.md`.

## Required validation

```bash
python -m pip install -e .[dev]
pytest
python scripts/smoke_workflow.py
```
