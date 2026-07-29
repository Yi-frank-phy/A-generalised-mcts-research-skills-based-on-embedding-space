# DTE lifecycle enforcement hooks

`dte_enforcement_hook.py` is the unified Codex lifecycle dispatcher for the
production App-native path. It handles `UserPromptSubmit`, `PreToolUse`,
`PostToolUse`, `Stop`, and `SessionStart`.

Its authority is deliberately narrow:

- activate or restore one DTE session for the current Codex session;
- allow only the current root turn to call `hook-driver`;
- inject and rotate a single-use execution capability;
- block direct App mutators through either Python-module or `dte-backend`
  console entrypoints;
- protect manifest/run/receipt paths and the complete supplying
  `src/dte_backend` tree while a run is active;
- validate strict driver receipts after tool completion;
- prevent the turn from ending before the backend terminal handoff exists.

The hook does not compute Judge decisions, embeddings, entropy, UCB,
allocation, Relation output, readiness, or termination. Those remain in the
DTE backend. `PostToolUse` is verification/feedback, not rollback; the backend
execution contract and atomic commit boundary provide the anti-bypass check.

## Production commands

```text
python -m dte_backend hook-driver activate
python -m dte_backend hook-driver init --spec <spec.json> --nodes <nodes.json>
python -m dte_backend hook-driver step
python -m dte_backend hook-driver submit --result <result.json>
python -m dte_backend hook-driver control --action retry|cancel|request-synthesis
python -m dte_backend hook-driver status
python -m dte_backend hook-driver handoff
```

Every command prints exactly one `dte-hook-receipt.v1` JSON object. A failed or
replayed capability cannot produce a successful receipt or advance graph
revision.

## Install, verify, and rollback

```text
python scripts/install_dte_hooks.py --scope user
python scripts/install_dte_hooks.py --scope managed-template
python scripts/install_dte_hooks.py --verify
python scripts/install_dte_hooks.py --rollback
```

User installation merges the existing `~/.codex/hooks.json`, removes only old
DTE reminder/enforcement handlers, preserves unrelated hooks, backs up affected
files under `~/.codex/hook-backups`, and atomically installs the dispatcher.
The command definition includes both the dispatcher content SHA-256 and the
normalized supplying Skill root. Changed bytes or a moved/replaced Skill tree
therefore produce a new trust definition. Before importing `dte_backend`, the
dispatcher checks its own bytes and resolves the exact pinned Skill root rather
than whichever editable package happens to sort first. While an enforced run is
active, PreToolUse also protects the complete backend package directory against
ordinary tool writes. This is a workflow integrity boundary for normal agents,
not a cryptographic identity for every backend module or a defence against an
administrator. Relative paths are resolved against `tool_input.workdir` (or the
event `cwd`), so changing a tool's working directory does not bypass the
protected tree.

`--verify` proves only the static configuration, matcher, dispatcher hash,
pinned Skill root, and local self-test; it does not prove that the running App
delivered a lifecycle event. After installation, use `/hooks` to review and
trust the new definition, then fully restart Codex.

Verify actual App event delivery with a nonce that does not create DTE state:

```text
/evolving-frontier-research --hook-probe ProbeNonce_1234
```

The prompt must receive `DTE_HOOK_PROBE_ACK:ProbeNonce_1234`. Then execute the
sentinel command `dte-hook-probe ProbeNonce_1234`; `PreToolUse` must deny it with
the same acknowledgement, proving that the command never reached the shell. A
successful manual script self-test plus static verification is not a substitute
for these two App event-bus observations.

On Windows, production `hook-driver` commands must use the workspace's
PowerShell shell. The dispatcher denies an explicitly selected `cmd.exe` or
other shell rather than emitting incompatible environment-assignment syntax.
The installer emits the PowerShell call operator plus single-quoted literal
arguments, including when `python.exe`, the Hook path, or the Skill root
contains spaces.

If plain `python -m dte_backend` resolves another editable checkout, do not
uninstall it implicitly. Inspect the import origin, obtain user approval to
remove the conflicting distribution, and repeat static verification plus the
live probe. The installed command independently pins the supplying Skill root
with `--dte-skill-root`, so the enforcement dispatcher imports its `src` tree
even when plain Python remains ambiguous.

The managed template targets
`C:\ProgramData\Codex\DTE\hooks\dte_enforcement_hook.py`, pins
`[features].hooks = true`, and intentionally does not enable
`allow_managed_hooks_only`. An administrator must deploy the script and make the
complete DTE root read-only for ordinary users. The backend package must be
deployed beside it at `C:\ProgramData\Codex\DTE\src\dte_backend`; the generated
trusted definitions pin that root with `--dte-skill-root`. Otherwise a managed
hook would still depend on a user-writable Skill copy. Generating the template
does not claim that managed installation occurred.

## Development-only guards

`dte_guard.py` remains useful for adapter development, fixtures, and smoke
validation:

```text
python hooks/dte_guard.py spec examples/run_spec.json
python hooks/dte_guard.py judge --nodes examples/frontier_nodes.json --output <judge_output.json>
python hooks/dte_guard.py relation --nodes examples/frontier_nodes.json --output <relation_output.json>
python hooks/dte_guard.py executor --parent <parent.json> --output <executor_output.json> --child-count <n>
```

These standalone checks are not manual prerequisites for production App runs
and do not define a second controller.
