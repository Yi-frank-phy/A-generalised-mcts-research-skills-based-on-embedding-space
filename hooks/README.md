# Hooks

Hooks are guardrails, not the DTE engine. Use them to validate machine-facing
artifacts at role boundaries before the main loop consumes the output.

## When to run

Run these checks at the following points:

1. `spec`: before starting a DTE run or accepting a user-edited run spec.
2. `executor`: immediately after an executor episode returns child nodes.
3. `judge`: immediately after a Judge oracle returns scores.
4. `relation`: immediately after a relation oracle classifies frontier nodes.

Do not put search, scoring, allocation, merge mutation, or synthesis inside a
hook. The hook only validates that an artifact is allowed to cross a boundary.

## Exact commands

Validate a run spec:

```bash
python hooks/dte_guard.py spec examples/run_spec.json
```

Validate executor output against the parent and allocated child count:

```bash
python hooks/dte_guard.py executor \
  --parent examples/executor_parent.json \
  --output examples/executor_output.json \
  --child-count 1
```

Validate Judge oracle output:

```bash
python hooks/dte_guard.py judge \
  --nodes examples/frontier_nodes.json \
  --output examples/judge_output.json
```

Validate relation oracle output:

```bash
python hooks/dte_guard.py relation \
  --nodes examples/frontier_nodes.json \
  --output examples/relation_output.json
```

Run all hook sample checks through pytest:

```bash
python -m pytest tests/test_hooks.py
```

## Codex UserPromptSubmit install

The Codex app hook layer can inject workflow reminders before the agent starts a
DTE task. Install `hooks/dte_prompt_guard.py` into the user hook directory and
add it to `C:\Users\zhaoy\.codex\hooks.json` under `UserPromptSubmit`.

Example command entry:

```json
{
  "type": "command",
  "command": "python C:\\Users\\zhaoy\\.codex\\hooks\\dte_prompt_guard.py"
}
```

This prompt hook is only a workflow reminder. The hard artifact checks still
come from `hooks/dte_guard.py` at the exact spec, Judge, Relation, and Executor
boundaries listed above. If any guard command fails, the agent must stop before
consuming that artifact.

## Backend entrypoint hardening

The backend CLI also enforces the spec guard in `python -m dte_backend validate`
and `python -m dte_backend run`. This prevents a non-compliant run spec from
reaching embedding provider setup or the DTE loop when an agent forgets the
manual `hooks/dte_guard.py spec ...` command.

Judge, Relation, and Executor outputs are already validated by their subprocess
entrypoints. Codex platform hooks are still useful as an additional runtime
boundary when those outputs are produced outside the backend CLI.
