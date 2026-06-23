# DTE Codex Skill Backend

> **Codex / maintainer note:** read [`CODEX_NEXT_STEPS.md`](./CODEX_NEXT_STEPS.md) before making changes. It contains the current blockers, known mismatches, and exact next implementation steps. Do not redesign the architecture.

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
CODEX_NEXT_STEPS.md        current blockers and exact next steps for Codex
AGENTS.md                  Codex/Kimi/OpenClaw operating instructions
SKILL.md                   DTE skill contract
PRD.md                     product requirements
SPEC.md                    technical specification
ARCHITECTURE.md            architecture decision record
src/dte_backend/           Python backend skeleton
schemas/                   JSON schemas
hooks/                     validation hook examples
examples/                  example run specs and node outputs
tests/                     tests
```

## Minimal local check

```bash
python -m pip install -e .[dev]
pytest
python -m dte_backend validate examples/run_spec.json
python hooks/dte_guard.py spec examples/run_spec.json
python -m dte_backend judge-oracle --nodes examples/frontier_nodes.json --judge-command "python examples/mock_judge_adapter.py"
python -m dte_backend relation-oracle --nodes examples/frontier_nodes.json --relation-command "python examples/mock_relation_adapter.py"
python -m dte_backend run --spec examples/run_spec.json --out-dir artifacts/prototype --cache-path .dte_cache/cache.json
```

## Design stance

Freeze the DTE architecture. Package it as a skill-backed research backend. The user should provide task parameters, not rewrite the architecture for each run.
