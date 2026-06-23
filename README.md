# DTE Codex Skill Backend

**DTE Codex Skill Backend** is a minimal scaffold for packaging Deep Think Evolving as a fixed, mandatory research protocol that can be driven by Codex/Kimi/OpenClaw-style agents without letting those agents bypass the DTE search engine.

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

## What this repo is for

Use this scaffold when you want Codex or a similar coding/research agent to run high-depth personal research while still obeying the DTE protocol.

It is designed for:

- mathematical / physical derivation exploration;
- academic idea search;
- high-depth interest research;
- long-form critical discussion;
- structured comparison of competing hypotheses.

It is not designed for:

- SaaS usage;
- selling Codex quota;
- bypassing platform limits;
- replacing DTE with a generic swarm.

## Repository layout

```text
AGENTS.md                  Codex/Kimi/OpenClaw operating instructions
SKILL.md                   DTE skill contract
PRD.md                     product requirements
SPEC.md                    technical specification
ARCHITECTURE.md            architecture decision record
pyproject.toml             Python package metadata
src/dte_backend/           minimal Python backend skeleton
prompts/                   role-isolated prompt resources
schemas/                   JSON schemas for run specs and search nodes
hooks/                     validation hook examples
examples/                  example run specs and node outputs
tests/                     basic schema and allocator tests
```

## Minimal local check

```bash
python -m pip install -e .[dev]
pytest
python -m dte_backend validate examples/run_spec.json
python -m dte_backend allocate examples/frontier_nodes.json --budget 4
python -m dte_backend validate-executor --request examples/expansion_request.json --executor-command "python examples/echo_executor_adapter.py"
python -m dte_backend run --spec examples/run_spec.json --nodes examples/frontier_nodes.json --out-dir artifacts/prototype
```

The current prototype is offline and deterministic: it uses a heuristic batch Judge, local hashed text features as a novelty proxy, UCB/Boltzmann allocation, deterministic expansion, and final synthesis. See `PROTOTYPE.md`.

The CLI is intended for agents, hooks, or CI. The human-facing interface should remain parameter-based: you provide a problem, goal, constraints, and budget; Codex invokes the backend.

## Design stance

The main design stance is:

> Freeze the DTE architecture. Package it as a skill-backed research backend.

Do not ask the user to rewrite the architecture for each run. The user should only provide task parameters.
