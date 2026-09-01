# DTE New — Release Design

This file is the normative architecture authority for the `new` release line. Ordinary refactors and feature work do not alter its controller boundaries. Theory-affecting changes must update `docs/PHYSICS.md`, this design, tests, and the theory lock together.

## Product-line rule

The repository has two parallel release lines: `old` preserves the direct-node-embedding/RBF-KDE controller; `new` uses the completed-transition/proper-volume metric-measure controller defined in `docs/PHYSICS.md`. Neither line is conceptually subordinate to `main`; release identity is the branch/tag line.

## New runtime data contract

Every actively searchable node carries a completed research transition: non-empty `retrospective_method`, `epistemic_change_kind` in `new_understanding | sharper_unknown | no_material_change`, and non-empty `epistemic_change`. The canonical controller embedding contains only those fields. Claim, rationale, question/context, Judge score, parent IDs, UCB, allocation, and runtime metadata are excluded. Seed producers and Executor outputs must supply the transition fields; the new controller fails closed otherwise.

## Frozen reference atlas

The `new` release ships a problem-independent geometry-only atlas spanning canonical research methods and epistemic outcomes. Reference cells are never live frontier nodes, realized evidence, or Judge observations. At run initialization the controller combines these fixed cells with the run's initial completed transitions and freezes their embeddings, connected sparse angular graph, geodesic matrix, and volume gauge. Since canonical transition embedding excludes problem/context `Q`, the release atlas is reusable across research problems.

The graph starts from the configured kNN degree and increases `k` only as needed to obtain one connected frozen graph; the resolved `k` is part of atlas identity. The atlas may be cached only when provider/model/dimension, configured graph degree, and initial canonical transitions match exactly. Raw parent→child transition evidence is remeasured on the current atlas rather than reusing numeric returns from another atlas.

The atlas is a numerical landmark/quadrature structure, not a discrete ontology of live/query states. Frozen reference vertices retain their exact shortest-path graph metric and their finite cumulative proper-volume profiles. Arbitrary off-atlas live/query embeddings use one continuous partition of unity to interpolate both distance-to-landmark profiles and proper-volume profiles. Query separation is measured in distance-profile space, while realized return, occupancy, and reward SD use the interpolated proper-volume field. Hard nearest-reference rounding and hard off-atlas cell-inclusion are legacy zero-order approximations only.

## Runtime controller

The production `dte_backend` path on `new` owns the proper-volume implementation; `dte_nextgen` is removed after migration. Each iteration embeds completed transitions, evaluates their continuous off-atlas distance and proper-volume fields against the frozen atlas, reconstructs proper-volume historical returns and local `V`, computes live occupancy and entropy-matched reward `SD` from that same continuous reward field, forms `U=V+SD`, entropy-matches the one-action Boltzmann allocator under hard budgets, executes the selected continuation, commits completed children, retires consumed parents, and repeats.

Judge remains an observable research-assessment role for provenance/risk/synthesis support. Its score is not controller value on `new`.

## Persistence

Persistent runs retain the initial completed transitions and provider/configuration needed to deterministically reconstruct the same frozen atlas after process restart. Raw transition edges are the durable value evidence; numeric return values are not portable across atlas identities. Compatibility telemetry names may survive schema migration only when their new semantics are explicit.

## Packaging and release

`new` and `old` are independently buildable and releasable as repository Skill bundles plus backend wheels. New tags use a distinct prefix such as `new-v0.3.0-alpha.1`; old tags use the parallel old-line prefix. Full Linux/Windows CI and package verification gate releases.

## Repository hygiene

The final `new` release tree keeps current operational docs plus formal `PHYSICS.md`, `DESIGN.md`, `SPEC.md`, `ARCHITECTURE.md`, and user-facing workflow documentation. Temporary handoffs, implementation plans, intermediate theory audits, superseded nextgen specs, and experiment notes are deleted. CI pins SHA-256 digests of `docs/PHYSICS.md` and `docs/DESIGN.md`; ordinary changes must not alter them.
