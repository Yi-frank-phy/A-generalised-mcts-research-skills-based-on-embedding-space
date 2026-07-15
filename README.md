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

`strict-run --mode real` remains the compatible headless/legacy entrypoint. In Codex App / Work, the normal native path is the persistent driver protocol below: the current App main agent requests and performs each bounded episode itself; the repository does not launch a second Codex process. In both paths the main agent is an authorized operator proxy under `OperatorPolicy`, not a second controller.

`requested_by` identifies the actor for audit; `operator_policy` determines whether that actor is authorized. The JSON field does not authenticate the writer. The current protocol trusts the root/operator execution context that invokes the backend; stronger actor/capability isolation belongs to a future external DTE Driver.

The real-mode controller and provider wiring are tested with a deterministic embedding-provider stub. Live Gemini API connectivity is intentionally not exercised because no production credential is available. This is not a merge blocker. CI still verifies the Gemini provider wiring, 3072-dimensional policy, cache namespace, and fail-closed behavior when neither supported API-key environment variable is present.

## App-native Judge → controller → Executor vertical slice

Ordinary unscored frontier nodes now progress through transport-neutral, versioned Judge and Executor boundaries driven by the current Codex App main agent:

```text
unscored frontier -> Judge EpisodeRequest -> current App native work
    -> Judge EpisodeResult -> atomic backend commit
    -> backend embedding/KDE/entropy/UCB/allocation
    -> Executor EpisodeRequest -> current App native work
    -> Executor EpisodeResult -> atomic backend commit
```

`commit_episode_result(...)` dispatches by the committed request role and validates the complete result before replacing graph state. Judge commits require exactly one observable score/reasoning observation per granted node and reject stale revisions, missing/extra/duplicate node IDs, invalid scores, or controller-owned pollution. Executor validation retains the existing over-grant, collision, ancestry, type/status, lifecycle, hash, and revision firewall. Every rejection leaves graph and node revisions unchanged.

The App-native backend command loop is:

```bash
python -m dte_backend create-run --run-dir <run-dir> --spec <spec.json> --nodes <committed-nodes.json>
python -m dte_backend next-episode --run-dir <run-dir>
# Current App main agent performs the returned request with native tools/subagents.
python -m dte_backend submit-episode-result --run-dir <run-dir> --result <result.json>
python -m dte_backend run-status --run-dir <run-dir>
```

`fail-episode`, `cancel-episode`, and `retry-episode` provide explicit attempt transitions; retries receive a new `attempt_id`, and cancelled, expired, failed, superseded, rejected, or already committed attempts cannot commit. Requests, results, and status records live under `<run-dir>/episodes/<episode-id>/<attempt-id>/`; writing a result file alone never mutates the graph.

The existing subprocess Executor is preserved only as a legacy/headless fallback and regression baseline through `CommandAgentEpisodeAdapter`. `NativeStubEpisodeAdapter` is a deterministic test fixture. Neither is the normal Codex App path, and neither is described as Ultra integration. SDK/App Server transports are deferred by the normative App profile.

Episode telemetry is append-only JSONL at `<run-dir>/episode_events.jsonl`. Because hidden App runtime usage and subagent topology are not available to repository code, App events record `usage_source=unavailable` and do not estimate tokens, quota, subagent count, or routing traces.

App-path embedding vectors are cached in the run-scoped `<run-dir>/dte_cache.json` through the existing `FileDTECache` namespace contract (provider, model/snapshot, dimension, and embedding contract version). The cache is not graph state; a cache failure cannot partially commit controller fields or revisions. Terminal `ready_for_synthesis` / `run_complete` actions are sticky, and after already-allocated Executor grants are consumed the iteration cap is enforced before any new Judge grant.

App-native Relation episodes now maintain a versioned semantic relation layer before a new Synthesis terminal action. The backend completely inventories selected-set exact duplicates and potential material conflicts (at most 28 pairs for the default eight-node provisional set) before readiness, grants bounded Relation episodes, and then may spend at most `max_relation_enrichment_pairs` successful nonblocking semantic classifications across the run (default 3). Known candidate/record identities are removed before enrichment truncation, so previously seen pairs cannot hide unseen pairs.

Relation state persists under `<run-dir>/relations/` as `candidates.json`, `relation_ledger.json`, and `synthesis_readiness.json`. Relation remains equivalent/complementary/conflict/independent semantic classification, not scientific verification; discriminator proposals are persisted but not executed. Terminal `ready_for_synthesis` / `run_complete` remains sticky; legacy terminal runs are not reopened. Native Seed and final Synthesis episodes remain deferred, as does full production role closure. The headless Judge/Relation commands remain regression/legacy paths rather than the normal App runtime.

## License

Apache-2.0. See [`LICENSE`](./LICENSE).

## Design stance

Freeze the research workflow. Package it as a skill-backed research backend. The user should provide task parameters, not rewrite the architecture for each run.
