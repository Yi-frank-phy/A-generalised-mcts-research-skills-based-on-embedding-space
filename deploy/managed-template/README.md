# Managed DTE hook template

Deploy `dte_enforcement_hook.py` to `C:\ProgramData\Codex\DTE\hooks`, deploy this skill's `src/dte_backend` tree to `C:\ProgramData\Codex\DTE\src\dte_backend`, make the complete `C:\ProgramData\Codex\DTE` tree read-only for ordinary users, and install `requirements.toml` through the administrator-managed Codex configuration layer. The trusted command pins that protected root with `--dte-skill-root`. This template intentionally does not set `allow_managed_hooks_only`.
