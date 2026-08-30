# Paste-ready Workspace Agent instructions

Use the `workspace-agent` branch of the DTE repository.

For DTE research runs, read `WORKSPACE_AGENT.md` before `SKILL.md`. Where the
Codex App hook instructions in `SKILL.md` conflict with `WORKSPACE_AGENT.md`,
the Workspace profile wins.

Mandatory Workspace profile:

1. Do not install, enable, or claim Codex lifecycle hooks.
2. Use `python scripts/workspace_driver.py ...` as the only production DTE
   lifecycle entrypoint.
3. At each new user/root turn, run `preflight` and `new-turn` before lifecycle
   mutations.
4. Drive `activate/init/step/request/submit/control/handoff` only through
   `workspace_driver.py driver -- ...` and require the post-command receipt
   audit to succeed.
5. Keep controller decisions, graph mutation, allocation, stopping, receipt
   validation and terminal handoff backend-owned.
6. For separable research branches, prefer native subagent delegation. Request
   `fork_turns="none"` when the host supports it, but record isolation as
   unverified unless the host supplies attestation. Do not infer model or
   reasoning effort when the runtime does not report them.
7. Never read out, print, commit, paste into instructions, or place in a shared
   Skill/file any Gemini API key. The adapter may consume the key only from its
   supported runtime secret sources.
8. A Workspace run may claim `backend_receipt_chain_verified=true` only after
   the adapter audit succeeds. It must still report `host_hook_enforcement=false`
   and `wrapper_use_host_enforced=false`.
9. If the adapter is bypassed, or terminal handoff is missing, do not label the
   result as an enforced DTE run.
