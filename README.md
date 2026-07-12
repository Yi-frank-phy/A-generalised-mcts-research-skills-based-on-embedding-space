# Evolving Frontier Research Skill

> **Public alpha note:** this is a local Codex/agent skill backend, not a hosted service. The research workflow is frozen; continue by hardening workflow edges and oracle integrations, not redesigning the system.

**Evolving Frontier Research Skill** packages a fixed frontier-search research protocol for Codex-style agents. It turns open-ended mathematical, physical, academic, or conceptual research into a controlled loop of structured hypotheses, external judgment, geometric exploration, bounded expansion, relation checks, and final synthesis.

Core idea:

```text
User / validated RunSpec
        ↓
DTE backend (only outer controller)
        ↓
bounded role adapters → validated structured outputs
        ↓
Judge scores → entropy/novelty → UCB → Boltzmann expansion
        ↓
DTE-selected synthesis checkpoint → validated report
```

This repository is intentionally **not** a new general agent architecture. It is a packaging layer around a fixed research workflow:

- Role separation is preserved as anti-bias isolation.
- The math controller remains mandatory for research runs.
- Subagents can execute local episodes but cannot directly produce final conclusions.
- Skills and hooks enforce structured outputs and phase boundaries.
- UCB remains value/uncertainty driven; cost is handled by hard budgets and run profiles, not by changing the UCB objective by default.

The internal Python package still uses `dte_backend` for backward compatibility. Public-facing docs use the clearer name **Evolving Frontier Research Skill**.

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
AGENTS.md             Codex/Kimi/OpenClaw operating instructions
SKILL.md              slash-command skill contract
PRD.md                product requirements
SPEC.md               technical specification
ARCHITECTURE.md       architecture decision record
src/dte_backend/      Python backend implementation
schemas/              JSON schemas
hooks/                validation hook examples
examples/             example run specs and node outputs
tests/                tests
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

`scripts/codex_judge_adapter.py` calls `codex exec` by default. Set `DTE_CODEX_JUDGE_COMMAND` only when you need to override the Codex command used by that adapter.

The only production real-run entrypoint is `python -m dte_backend strict-run --mode real`. `strict-run` writes observable progress artifacts and polls `<out-dir>/strict_run_control.json` by default; `--control-path` may select another operator-controlled location. The main agent is an authorized operator proxy under `OperatorPolicy`: it may issue a validated synthesis request, but it cannot directly mutate controller-owned state or bypass safe-point and schema validation.

`requested_by` identifies the actor for audit; `operator_policy` determines whether that actor is authorized. The JSON field does not authenticate the writer. The current protocol trusts the root/operator execution context that invokes the backend; stronger actor/capability isolation belongs to a future external DTE Driver.

The real-mode controller and provider wiring are tested with a deterministic embedding-provider stub. Live Gemini API connectivity is intentionally not exercised because no production credential is available. This is not a merge blocker. CI still verifies the Gemini provider wiring, 3072-dimensional policy, cache namespace, and fail-closed behavior when neither supported API-key environment variable is present.

## License

Apache-2.0. See [`LICENSE`](./LICENSE).

## Design stance

Freeze the research workflow. Package it as a skill-backed research backend. The user should provide task parameters, not rewrite the architecture for each run.
