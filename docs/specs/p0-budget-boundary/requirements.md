# P0 Budget and Executor Boundary Requirements

## Introduction

This feature aligns the existing DTE backend with the P0 semantics in `SPEC.md`,
`ARCHITECTURE.md`, and issue #2. It separates per-iteration soft allocation mass
from the hard committed-child cap, makes discretization deterministic, closes
Executor-owned-field bypasses, and namespaces semantic caches without introducing
the future AgentEpisode, ledger, evaluator, or transport architecture.

## Requirements

### 1. Canonical budget contract

**User story:** As the DTE controller, I want separate soft allocation mass and
hard child-cap fields, so that continuous allocation is not confused with strict
integer conservation.

1. WHEN a budget is omitted or uses defaults, THE SYSTEM SHALL use
   `allocation_mass_per_iteration = 3` and `max_children_per_iteration = 5`.
2. WHEN a legacy `total_child_budget` is supplied, THE SYSTEM SHALL treat it only
   as a deprecated alias for `allocation_mass_per_iteration`.
3. WHEN canonical and legacy fields are supplied with different values, THE
   SYSTEM SHALL reject the run spec with an explicit validation error.
4. WHEN a run spec is serialized or a schema/example is generated, THE SYSTEM
   SHALL emit canonical budget fields and SHALL NOT emit the legacy field.

### 2. Deterministic allocation

**User story:** As the EvolutionController, I want normative discretization and
hard-cap trimming, so that allocations match the specification and are invariant
to frontier input order.

1. WHEN `q_i < 1`, THE SYSTEM SHALL discretize with round-half-up, mathematically
   `floor(q_i + 0.5)`.
2. WHEN `q_i >= 1`, THE SYSTEM SHALL discretize with `ceil(q_i)`.
3. WHEN tentative children exceed the hard cap, THE SYSTEM SHALL retain slots by
   descending marginal support, descending allocation value, ascending `node_id`,
   then ascending slot index.
4. WHEN the same nodes are supplied in another order, THE SYSTEM SHALL produce
   the same allocation mapped by `node_id`.
5. THE SYSTEM SHALL NOT alter the UCB formula or add a cost penalty.

### 3. Executor authority boundary

**User story:** As the DTE backend, I want Executor results treated as untrusted
structured input, so that model episodes cannot pre-fill controller state.

1. WHEN Executor output contains `local_embedding`, `judge_reasoning`, `score`,
   `uncertainty`, `ucb_score`, or `expansion_budget`, THE SYSTEM SHALL reject it
   even when the supplied value is `null` or zero.
2. WHEN an otherwise legal parsed `SearchNode` relies only on model defaults, THE
   SYSTEM SHALL continue accepting it.
3. THE SYSTEM SHALL continue rejecting synthesis nodes, non-frontier children,
   missing parent IDs, and child counts above the controller grant.

### 4. Cache namespace isolation

**User story:** As a DTE operator, I want cache entries isolated by provider and
evaluation contract, so that incompatible configurations cannot reuse results.

1. THE embedding cache identity SHALL include provider, model or snapshot,
   dimension, and embedding contract version.
2. THE Judge cache identity SHALL include model or snapshot, reasoning profile,
   rubric version, prompt version, and output schema version.
3. WHEN a namespace component changes, THE SYSTEM SHALL generate a different
   cache key without deleting or migrating legacy cache files.

### 5. Verification and delivery

**User story:** As the repository maintainer, I want regression tests and scoped
delivery, so that the P0 change can be safely merged on `main`.

1. THE SYSTEM SHALL pass targeted allocation, schema, Executor, cache, guard, and
   full pytest validation.
2. THE implementation SHALL update affected schemas, examples, and documentation.
3. THE delivery SHALL keep issue #2 open and mark only actually completed P0
   checklist items.
4. THE implementation SHALL NOT add AgentEpisode, revisions, ledger, SDK/App
   Server transport, evaluator, Relation-loop, or synthesis redesign work.
