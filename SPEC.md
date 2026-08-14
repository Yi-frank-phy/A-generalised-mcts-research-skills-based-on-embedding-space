# DTE `new` Release Specification

## 1. Authority and scope

This specification defines the executable contract of the `new` release line.

Normative authority is intentionally split by responsibility:

1. `docs/PHYSICS.md` — mathematical/controller physics. It is the only authority for the meaning of geometry, proper volume, value, occupancy, entropy, uncertainty, UCB, and allocation temperature.
2. `docs/DESIGN.md` — release architecture and product-line boundaries.
3. `SPEC.md` — executable data, lifecycle, persistence, packaging, and CI requirements.

If prose elsewhere conflicts with these files on branch `new`, these three files win in the order appropriate to their responsibility. Ordinary engineering changes must not reinterpret `docs/PHYSICS.md` or `docs/DESIGN.md`.

The parallel branch `old` preserves the direct-node-embedding/RBF-KDE controller. `new` and `old` are independent release lines; neither is conceptually subordinate to `main`.

## 2. Canonical searchable state

Every actively searchable node on `new` must contain one completed research transition with all of:

```text
retrospective_method: non-empty string
epistemic_change_kind: new_understanding | sharper_unknown | no_material_change
epistemic_change: non-empty string
```

The canonical controller embedding is derived only from the completed transition `(method, epistemic change)` contract defined in `docs/PHYSICS.md`.

The following fields are not part of canonical controller embedding:

```text
claim
rationale
question/context Q
Judge score
parent identifiers
UCB/allocation telemetry
runtime/session metadata
```

Seed producers and Executor outputs must provide completed-transition fields directly. The `new` controller fails closed when an active node lacks them; it must not reconstruct them from claim text as a compatibility fallback.

## 3. Frontier lifecycle

The active frontier contains completed transitions that are available for continuation.

For a selected parent:

1. the parent remains in immutable history/provenance;
2. the parent leaves the active frontier after its continuation is committed;
3. committed completed children enter the frontier according to the granted expansion contract;
4. continuation therefore replaces consumed frontier state rather than appending an indefinitely reusable parent.

Rejected, cancelled, failed, or empty Executor attempts do not create completed child transitions.

## 4. Controller ownership

`src/dte_backend/**` is the production controller implementation on `new`.

The backend alone may:

- freeze/reconstruct the reference atlas;
- embed canonical completed transitions;
- compute intrinsic distances and proper-volume observables;
- reconstruct realized parent→child returns and local `V`;
- compute live occupancy, entropy-matched reward uncertainty `SD`, and `U = V + SD`;
- determine allocation temperature and expansion allocation;
- enforce hard run/node/episode budgets;
- commit graph-state transitions;
- decide whether another iteration, relation work, or synthesis is permitted.

External model runtimes may perform bounded Seed, Judge, Executor, Relation, or Synthesis episodes, but they cannot directly mutate controller state or allocate search budget.

Judge output remains an observable assessment for provenance, risk, and synthesis support. Judge score is not controller value `V` on `new`.

No production runtime on `new` may depend on `src/dte_nextgen/**`.

## 5. Frozen reference atlas

A run uses the frozen-atlas contract in `docs/PHYSICS.md` and `docs/DESIGN.md`.

The release provides a problem-independent geometry-only reference atlas. At run initialization, the backend combines the packaged reference cells with the run's initial completed transitions and freezes the embedding/geometry/volume identity required by the controller.

Reference cells:

- are never live frontier nodes;
- are never realized research evidence;
- do not count as UCB visits;
- do not receive Judge scores;
- exist only to define geometry and measure.

Within one run, the atlas identity and volume gauge are fixed. Raw transition edges are durable evidence; numeric returns from a different atlas identity must not be reused without remeasurement.

## 6. Episode boundary

The normative external execution boundary is transport-neutral:

```text
EpisodeRequest -> EpisodeResult
```

The backend owns request grants, revisions, role, parent selection, maximum returned children, allowed output schema, deadlines, and commit validation.

Executor results on `new` must return completed-transition data for every proposed child. Model-authored results cannot pre-authorize controller metrics or graph revisions.

Runtime thread IDs, response IDs, subagent traces, and compaction summaries are observability metadata rather than graph facts.

## 7. Budgets and allocation

Hard resource limits remain outside the UCB objective. Cost or quota penalties must not be added to `U` unless `docs/PHYSICS.md` is intentionally changed.

The backend enforces the configured run contracts for at least:

```text
max_committed_search_nodes
max_iterations
allocation_mass_per_iteration
max_children_per_iteration
max_relation_pairs_per_episode
max_relation_enrichment_pairs
```

Committed-node budget is non-renewable. Merge or archival may reduce current graph complexity but does not refund already-spent node budget.

Allocation uses the controller output defined in `docs/PHYSICS.md` and is then clipped by hard remaining budgets at request and commit boundaries.

## 8. Persistence and replay

Persistent state must retain enough information to reconstruct the same controller semantics after restart, including:

- completed transition fields;
- graph and node revisions;
- parent→child transition edges;
- initial completed transitions needed by the frozen-atlas contract;
- embedding provider/model/dimension or equivalent provider identity;
- controller configuration that participates in atlas identity;
- episode and provenance ledgers required for accepted commits.

Compatibility telemetry names may survive migration only when their `new` meaning is explicit. A legacy field name must never silently restore legacy controller physics.

## 9. Synthesis and relation boundaries

Relation/Merge and Synthesis remain bounded semantic graph operations. They do not redefine controller reward, uncertainty, or UCB.

Relation output may classify semantic relationships and propose graph-maintenance actions, but backend validation/commit remains authoritative. Synthesis consumes the terminal handoff selected by backend readiness logic; it does not retroactively alter historical controller evidence.

## 10. Packaging and release

`new` must be independently releasable as:

- the repository Skill bundle; and
- the backend wheel.

The production Skill bundle inventory is generated from the current release tree by `scripts/generate_bundle_manifest.py`. CI regenerates the manifest before verification and package installation, then synchronizes the managed-template copy. Installed bundles remain tamper-checked against their generated manifest.

A release candidate is gated by the repository CI matrix across supported Linux/Windows Python versions, full `pytest`, smoke workflow, wheel installation outside the checkout, repository Skill-bundle installation, and manifest/tamper verification.

Direct pushes to `new` and `old` must trigger CI in addition to pull-request CI.

## 11. Repository hygiene

Production code lives under `src/dte_backend/**`.

`delete/**` is a temporary manual-review quarantine and is not authority, runtime code, or production bundle content. Files there may be removed after owner review.

The final `new` tree should not retain active development handoffs, intermediate theory drafts, superseded nextgen controller implementations, or duplicate mathematical authorities.

## 12. Superseded semantics

On branch `new`, the following must not regain production authority without an explicit theory change to `docs/PHYSICS.md`:

- direct claim embedding as controller state;
- ordinary RBF-KDE `1/sqrt(N rho)` uncertainty;
- Judge score as `V`;
- MMD return as canonical `V` evidence;
- `V + T log 2` as UCB;
- node-local `[0,1]` proper-volume normalization;
- required tangent/metric-tensor/value-gradient machinery;
- hidden compatibility fallbacks that reconstruct missing transition state from claim text.

## 13. Change policy

A normal code change may update implementation, tests, operational docs, generated release metadata, or this executable specification when behavior is clarified without changing controller physics.

A theory-affecting change must be explicit: update `docs/PHYSICS.md`, `docs/DESIGN.md`, relevant tests, and the theory-lock digests together, with the reason/evidence recorded in the change itself. Silent mathematical drift is a release failure.
