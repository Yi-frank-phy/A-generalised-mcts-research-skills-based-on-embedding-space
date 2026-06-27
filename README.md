# DTE Codex Skill Backend

> **Public alpha note:** this is a local Codex/agent skill backend, not a hosted service. Before making the repository public, read [`PUBLIC_RELEASE_CHECKLIST.md`](./PUBLIC_RELEASE_CHECKLIST.md). The DTE architecture is frozen; continue by hardening workflow edges and oracle integrations, not redesigning the system.

**DTE Codex Skill Backend** packages Deep Think Evolving as a fixed research backend that can be driven by Codex/Kimi/OpenClaw-style agents while preserving the DTE controller.

Core idea:

```text
Codex / Kimi / OpenClaw executor
        ↓
DTE Skill Protocol
        ↓
Fixed role-isolated research flow
        ↓
Python math backend: Judge scores → entropy/novelty → UCB → Boltzmann expansion
        ↓
structured SearchNode outputs
        ↓
DTE synthesis protocol
```

This repository is intentionally **not** a new architecture. It is a packaging layer around an existing architecture:

- Role separation is preserved as anti-bias isolation.
- The DTE math engine remains mandatory for research runs.
- Subagents can execute local episodes but cannot directly produce final conclusions.
- Skills and hooks enforce structured outputs and phase boundaries.
- UCB remains value/uncertainty driven; cost is handled by hard budgets and run profiles, not by changing the UCB objective by default.

## Status

This is an **alpha skill/backend**. The smoke path is fully local and should pass without external API keys. Real research mode requires a real Judge command, such as:

```bash
python -m dte_backend strict-run \
  --mode real \
  --spec <run_spec.json> \
  --out-dir artifacts/session \
  --cache-path .dte_cache/cache.json \
  --judge-command "python scripts/codex_judge_adapter.py"
```

`examples/mock_*_adapter.py` are smoke-test tools only. Hash embedding is a debug/dry-run fallback, not real geometry.

## Repository layout

```text
PUBLIC_RELEASE_CHECKLIST.md              public release checklist
CODEX_NEXT_STEPS.md                      current top-priority next steps
READ_THIS_FIRST_REAL_ORACLE_BLOCKER.md   historical real-oracle note, now points to implemented Judge bridge
HOOK_WIRING_TODO.md                      hook and real-oracle wiring notes
AGENTS.md                               Codex/Kimi/OpenClaw operating instructions
SKILL.md                                DTE slash-command skill contract
PRD.md                                  product requirements
SPEC.md                                 technical specification
ARCHITECTURE.md                         architecture decision record
src/dte_backend/                        Python backend skeleton
schemas/                                JSON schemas
hooks/                                  validation hook examples
examples/                               example run specs and node outputs
tests/                                  tests
```

## Minimal local check

```bash
python -m pip install -e .[dev]
pytest
python scripts/smoke_workflow.py
```

Smoke checks may use mock adapters. Real research can use the Codex Judge adapter:

```bash
python -m dte_backend strict-run \
  --mode real \
  --spec <run_spec.json> \
  --out-dir artifacts/session \
  --cache-path .dte_cache/cache.json \
  --judge-command "python scripts/codex_judge_adapter.py"
```

`code scripts/codex_judge_adapter.py` calls `codex exec` by default. Set `DTE_CODEX_JUDGE_COMMAND` only when you need to override the Codex command used by that adapter.

## License

Apache-2.0. See [`LICENSE`](./LICENSE).

## Design stance

Freeze the DTE architecture. Package it as a skill-backed research backend. The user should provide task parameters, not rewrite the architecture for each run.
