# Public Release Checklist

Use this checklist before changing the repository visibility to public.

## Required checks

```bash
python -m pip install -e .[dev]
pytest
python scripts/smoke_workflow.py
```

## Repository hygiene

- [x] `LICENSE` exists and matches the intended license.
- [x] `.gitignore` excludes `.env`, `.dte_cache/`, `artifacts/`, local logs, and virtual environments.
- [x] No API keys, tokens, local cache files, private transcripts, or personal research artifacts are committed.
- [x] README explains that `strict-run` is the slash-command entrypoint.
- [x] README explains that mock adapters are smoke-only.
- [x] `SKILL.md` is the primary runtime contract for skill invocation.
- [x] `scripts/smoke_workflow.py` passes without external API keys.
- [x] `scripts/gemini_smoke.py` is manual only and runs only when `GEMINI_API_KEY` or `GOOGLE_API_KEY` is set.
- [x] Real research mode uses `scripts/codex_judge_adapter.py` or another real Judge command, not mock adapters.

## Public-safety notes

- This repository is a local research skill/backend, not a hosted service.
- It must not be used to sell, proxy, or resell Codex or API access.
- Mock adapters are protocol tests only and do not provide research judgments.
- Hash embedding is a debug/dry-run fallback, not real geometry.

## Recommended visibility change

After all checks pass:

1. Confirm the default branch is `main`.
2. Confirm CI is green.
3. Use GitHub repository settings to change visibility from private to public.
4. Create a `v0.1.0-alpha` release tag if desired.
