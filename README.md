# DTE Codex Skill Backend

> **Read this first:** [`READ_THIS_FIRST_REAL_ORACLE_BLOCKER.md`](./READ_THIS_FIRST_REAL_ORACLE_BLOCKER.md). The remaining blocker is real Codex oracle integration for `strict-run --mode real`, not DTE architecture. Do not redesign the architecture.

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

## Repository layout

```text
READ_THIS_FIRST_REAL_ORACLE_BLOCKER.md   current top-priority blocker
HOOK_WIRING_TODO.md                      hook and real-oracle wiring notes
CODEX_NEXT_STEPS.md                      historical/current next steps
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

Smoke checks may use mock adapters. Real research must use:

```bash
python -m dte_backend strict-run --mode real --judge-command "<real Codex Judge Oracle command>" ...
```

## Design stance

Freeze the DTE architecture. Package it as a skill-backed research backend. The user should provide task parameters, not rewrite the architecture for each run.
