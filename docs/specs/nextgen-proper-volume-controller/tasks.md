# Tasks

> The list below records the original PR #48 migration plan. Those tasks are historical. The current continuum geometry work is tracked in issue #54 and the active repair PR/branch. See [`../../PROPER_VOLUME_GEOMETRY.md`](../../PROPER_VOLUME_GEOMETRY.md).

## Original PR #48 tasks — completed/historical

1. Add tests for absolute/non-node-normalized cumulative proper volume and reward-SD on the same variable.
2. Add tests for angular sparse-graph geodesic and the then-used nearest-reference anchoring.
3. Add tests that live duplicates have higher occupancy/lower reward-SD than an isolated direction.
4. Add tests for end-to-end `U=V+SD` with default equal-weight frozen reference measure.
5. Add tests for a stateful session: frozen atlas, zero value before history, parent->child return writeback, slot replacement, history-local value regression, and rejection of unscoped pre-numeric history.
6. Implement the minimum modules needed to make those tests green while preserving legacy RBF/MMD baselines.
7. Update next-generation status documentation.
8. Validate through the public CI matrix.
9. Keep the private experiment PR as provenance/research history rather than the release target.

## Continuum repair tasks — current

1. Remove hard nearest-reference anchoring from authoritative live/query geometry while keeping the legacy helper only for diagnostics/compatibility.
2. Preserve every frozen reference-vertex graph distance exactly under the replacement estimator.
3. Add off-atlas tests showing that motion inside an old nearest cell is nonzero and that crossing an old Voronoi boundary no longer produces a finite jump.
4. Route realized proper-volume return, historical value regression, occupancy and Boltzmann reward SD through the same continuous query geometry.
5. Lock the definition: proper volume is `Omega(metric ball)`, not inverse sample density/KDE or empirical cosine calibration.
6. Preserve equal-weight atlas cells as the default declared quadrature measure; keep supplied density correction optional/numerical only.
7. Run full CI on the repair PR.
8. Separately test graph-distance convergence and finite measure/common-volume-gauge convergence under atlas refinement/resampling.
9. Only after those numerical checks, decide whether issue #54 can close; do not reopen the proper-volume definition unless falsification evidence requires it.
