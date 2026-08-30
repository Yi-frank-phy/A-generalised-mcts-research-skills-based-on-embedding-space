# Workspace Agent DTE execution profile

This branch is the Workspace Agent compatibility line. It preserves the DTE
backend/controller semantics while replacing Codex host lifecycle hooks with an
explicit Workspace adapter.

## Assurance boundary

Use `scripts/workspace_driver.py` for all production DTE lifecycle operations.
The adapter injects the session, root-turn identity and single-use backend
capability, then re-audits the persisted receipt chain after every mutating
command.

The backend remains strict about state transitions, receipt hashes, capability
rotation, graph revisions and commit validation. The Workspace host, however,
does not currently force the agent to call this wrapper. Therefore:

- `backend_receipt_chain_verified=true` may be reported after a successful audit;
- `host_hook_enforcement=false` must always be reported for this profile;
- `wrapper_use_host_enforced=false` must always be reported;
- `context_isolation_verified=false` unless the host supplies independent
  attestation;
- native subagent delegation is useful for breadth/parallelism but must not be
  counted as independently isolated rollouts merely because it was delegated.

Do not claim Codex hook-grade enforcement in Workspace Agent runs.

## Production loop

At the start of a new user/root turn:

```bash
python scripts/workspace_driver.py preflight
python scripts/workspace_driver.py new-turn
```

Then drive the same backend protocol through the adapter:

```bash
python scripts/workspace_driver.py driver -- activate
python scripts/workspace_driver.py driver -- init --spec <spec.json> --nodes <committed-nodes.json>
python scripts/workspace_driver.py driver -- step
python scripts/workspace_driver.py driver -- request --chunk-index <n>
python scripts/workspace_driver.py driver -- submit --result <result.json>
python scripts/workspace_driver.py driver -- control --action retry|fail-attempt|cancel-attempt|cancel-run|request-synthesis
python scripts/workspace_driver.py driver -- resume --run-id <run-id>
python scripts/workspace_driver.py driver -- handoff
python scripts/workspace_driver.py status
```

The adapter deliberately reuses the existing backend `hook-driver` protocol as
an internal transaction/capability API. No host hook installation is required
or claimed on this branch. The legacy hook files remain for compatibility with
the Codex-native branch and bundle identity; they are inert in this Workspace
profile.

## Native subagents

For separable research branches, the main Workspace Agent should request native
subagents and prefer `fork_turns="none"` when the host interface permits it.
This is a requested parent-turn policy, not proof of full fresh-context
isolation. Record unavailable model/reasoning/context attestations as null or
unverified rather than inferring them.

Use native subagents as opportunistic executors. Controller decisions, graph
mutation, allocation, stopping, receipt validation and final handoff remain
backend-owned.

## Gemini secret: never commit it

The public repository must contain no Gemini API key. The Workspace adapter
loads `GEMINI_API_KEY` / `GOOGLE_API_KEY` in this order:

1. an already-set process environment variable;
2. a file selected by `DTE_SECRETS_FILE`;
3. the user-level DTE secret file;
4. the checkout-root `.env` file, which is gitignored.

The user-level default is:

- Windows: `%LOCALAPPDATA%\\DTE\\secrets.env`
- Linux/macOS: `${XDG_CONFIG_HOME:-~/.config}/dte/secrets.env`

To store the key without placing it on the command line or in shell history:

```bash
python scripts/configure_workspace_secret.py
```

The prompt uses hidden input and writes only the local user secret file. The
adapter never prints the key; `preflight` reports only whether a Gemini secret
is available and a non-sensitive source label.

If the Workspace runtime is ephemeral and does not preserve user files or
arbitrary environment variables, do **not** paste the key into agent
instructions, a Skill file, an attached shared file, or Git. Put the Gemini
call behind a private custom MCP/runner service and keep the provider key in
that service's secret store instead.

## Release / reporting rule

A Workspace run is valid DTE output only when the backend receipt chain audits
successfully and the terminal handoff is produced through the adapter. A model
can still choose to bypass the wrapper because no host hook forces usage; such
output must not be labelled as an enforced DTE run.
