# P0.5 Anti-bypass Reconciliation: Implementation Map

## Scope

This change removes the soft-bypass paths introduced by `0eea202` without implementing the future `AgentEpisode` P1 boundary.

## Mapping

| Conflict | Enforcement change | Regression coverage |
|---|---|---|
| A model-facing root agent can select or manually replay `app-orchestrated-real` | Make `python -m dte_backend strict-run --mode real` the only documented production real-run entrypoint; remove the manual harness from active runtime instructions and agent metadata while retaining the transport-neutral `AgentEpisode` direction in `SPEC.md` and `ARCHITECTURE.md` | Repository invariant tests for `SKILL.md`, `agents/openai.yaml`, README, and workflow docs |
| `requested_by="main_agent"` grants a model root stopping authority | Restrict `SynthesisControlRequest.requested_by`, `ForcedSynthesisRecord.requested_by`, and forced stop metadata to `user`; reject legacy main-agent files through strict validation | Model, control-file, CLI, artifact, and checked-in schema/example tests |
| Checkpoints can be mistaken for controller authority | Preserve checkpoint/frontier/entropy/status artifacts, but state that observation is not authority and that the main agent may recommend while only the user may interrupt | Documentation/artifact invariant tests |
| A control request could be consumed at an unsafe time | Poll only after a complete Judge/controller checkpoint or after a started node expansion has returned validated Executor output and been committed; never interrupt an oracle or reinterpret interruption as convergence | Ordering tests with recording callbacks/adapters and fail-closed invalid-control tests |
| P0 allocation, cache, and Executor boundaries could regress during reconciliation | Do not change allocation or cache algorithms and run the full existing suite plus smoke and role guards | Existing P0, Judge, Executor, Relation, smoke, and guard tests |

## Explicitly deferred

`EpisodeRequest` / `EpisodeResult`, graph revisions, stale-result rejection, atomic episode commit, event ledger, native transports, Seed Episode, and Relation-loop redesign remain P1/P2 work.
