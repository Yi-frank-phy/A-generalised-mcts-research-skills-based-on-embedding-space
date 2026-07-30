# Episode reasoning firewall

DTE separates two requirements that are often conflated:

1. a model may need continuous reasoning, tool state, and compaction while it is
   completing one bounded role episode;
2. a later branch, Judge, Relation, or Synthesis role must not treat that private
   reasoning as an authorized cross-role handoff.

The operational rule is therefore:

```text
within one AgentEpisode: reasoning continuity is permitted
across AgentEpisodes: only structured committed artifacts may cross
```

This is **micro-continuity with macro-restart**. It does not claim that a model
can retain all previous private reasoning and simultaneously act as an
independent reviewer in the same conversation.

## Machine contract

Every newly built `EpisodeRequest` carries the backend-reserved transport hint:

```json
{
  "dte_reasoning_boundary": {
    "schema_version": "dte-reasoning-boundary.v1",
    "continuity_scope": "within_episode",
    "cross_episode_private_reasoning_allowed": false,
    "provider_retained_reasoning_attested": false,
    "provider_compaction_attested": false
  }
}
```

The hint is installed before the role-context manifest is hashed. A caller may
supply unrelated transport hints, but it cannot replace this object with a more
permissive contract. A changed reasoning boundary therefore invalidates the
manifest presented by a strict runtime.

The backend does not read, persist, or verify provider-private chain of thought.
`provider_retained_reasoning_attested=false` and
`provider_compaction_attested=false` mean only that DTE has no trusted runtime
fact for those provider-internal mechanisms. They do not request that the model
disable reasoning continuity or compaction within the episode.

## Authorized cross-episode state

A later episode may receive only state that its exact `EpisodeRequest` grants,
such as:

- the assigned parent or blinded candidate nodes;
- explicit claims, assumptions, evidence, risks, and coverage obligations;
- committed epistemic contributions and relation records selected by the
  backend;
- the terminal deterministic handoff for Synthesis.

The following are not authorized handoff material:

- private reasoning or hidden scratchpad from an earlier episode;
- the full prior role transcript;
- ungranted frontier nodes;
- Judge conclusions hidden from Relation;
- controller-owned score, UCB, allocation, selection, or readiness state omitted
  from the role payload.

## Isolation remains a separate claim

The reasoning firewall constrains what may be transferred. It does not prove
that the runtime actually created a fresh context.

- `shared_context_single_agent` remains correlated fallback execution.
- `strict_fresh_context` still requires a new role-session identity and an exact
  manifest attestation accepted by the backend.
- Provider retained reasoning or compaction is not evidence of fresh-context
  isolation.

Thus a shared Codex App conversation can benefit from continuity inside an
episode while still being reported honestly as non-independent across roles.
A future transport that can create and attest fresh role sessions can satisfy
the same request contract without changing DTE search or allocation mathematics.

## Compatibility

The firewall is injected only when a new role request is built. No field was
added to the persisted `EpisodeRequest` or `RoleExecutionContract` Pydantic
schemas, so historical serialized requests and their stored request hashes are
not rewritten during load or migration.
