# P0.5 Anti-bypass Reconciliation: Implementation Map

## Scope

This change removes the soft-bypass paths introduced by `0eea202` without implementing the future `AgentEpisode` P1 boundary.

## Mapping

| Conflict | Enforcement change | Regression coverage |
|---|---|---|
| A model-facing root agent can select or manually replay `app-orchestrated-real` | Make `python -m dte_backend strict-run --mode real` the only documented production real-run entrypoint; remove the manual harness from active runtime instructions and agent metadata while retaining the transport-neutral `AgentEpisode` direction in `SPEC.md` and `ARCHITECTURE.md` | Repository invariant tests for `SKILL.md`, `agents/openai.yaml`, README, and workflow docs |
| `requested_by="main_agent"` could be mistaken for self-authorizing direct stopping authority | Treat `requested_by` as actor/audit metadata; authorize it against the validated `DTERunSpec.operator_policy`; apply the request only through the backend at a safe point; record `main_agent_requested_synthesis` separately from convergence | Model, authorization, control-file, CLI, artifact, and checked-in schema/example tests |
| Checkpoints can be mistaken for controller authority | Preserve checkpoint/frontier/entropy/status artifacts, but state that observation alone is not authority; delegated authority requires `OperatorPolicy` plus a validated backend command | Documentation/artifact invariant tests |
| A control request could be consumed at an unsafe time | Poll only after a complete Judge/controller checkpoint or after a started node expansion has returned validated Executor output and been committed; never interrupt an oracle or reinterpret interruption as convergence | Ordering tests with recording callbacks/adapters and fail-closed invalid-control tests |
| P0 allocation, cache, and Executor boundaries could regress during reconciliation | Do not change allocation or cache algorithms and run the full existing suite plus smoke and role guards | Existing P0, Judge, Executor, Relation, smoke, and guard tests |

## Explicitly deferred

`EpisodeRequest` / `EpisodeResult`, graph revisions, stale-result rejection, atomic episode commit, event ledger, native transports, Seed Episode, and Relation-loop redesign remain P1/P2 work.
