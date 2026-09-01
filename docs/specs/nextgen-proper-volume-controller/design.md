# Design

> **Historical PR #48 implementation plan.** This file records the original migration design and is retained for provenance. Its `nearest-reference anchoring` and hard off-atlas cell-inclusion steps were later identified as finite-grid bugs and are superseded by the continuous estimators in [`../../PHYSICS.md`](../../PHYSICS.md) and [`../../PROPER_VOLUME_GEOMETRY.md`](../../PROPER_VOLUME_GEOMETRY.md). `new` is now a standalone release line with production code under `src/dte_backend/**`.

## Original module boundary

The initial proper-volume prototype was isolated inside `dte_nextgen.thought_space` before PR #49 promoted the `new` release implementation into `dte_backend`.

Original planned modules:

- `geometry.py`: angular kNN graph, shortest-path geodesic, originally nearest-reference anchoring;
- `proper_volume.py`: atlas cell volumes, radius->cumulative proper volume, occupancy entropy, radial Boltzmann inversion, reward statistics;
- `occupancy.py`: live-live source-centred proper-volume displacement and soft occupancy;
- `controller.py`: retain legacy RBF functions and add proper-volume scoring from radii/embeddings;
- `return_metric.py`: retain MMD baseline and add proper-volume parent->child return;
- `history.py`: run-local realized-return evidence store;
- `session.py`: frozen-atlas stateful score/select/complete loop.

The `(method, epistemic change)` serializer and one-action Boltzmann scheduler were retained.

## Original data flow — still conceptually valid

One session freezes the reference atlas and computes its sparse geodesic matrix once.

For each scoring pass:

1. embed current completed-transition frontier;
2. estimate continuous live/query distance and proper-volume fields from the frozen atlas;
3. compute source-centred proper-volume separations `D_ij`;
4. estimate `rho_i` from current live mass only;
5. set `S_i=-log(rho_i)`;
6. invert each node's radial Boltzmann entropy;
7. push Boltzmann cell mass through the same continuous proper-volume reward field;
8. use the resulting ordinary reward SD as controller uncertainty;
9. regress historical realized proper-volume returns locally to obtain `V_i`;
10. score `U_i=V_i+SD_i`;
11. use mean node entropy as the target entropy for one-action Boltzmann allocation.

After external execution returns one completed child, measure parent->child proper-volume return on the same frozen estimator, record it against the retired parent, replace the parent slot with the child, and recompute on the next pass.

## Continuum repair addendum

The original finite implementation contained two zero-order approximations.

First, arbitrary live/query embeddings were rounded to one nearest atlas vertex. Current `dte_backend.space_geometry` instead represents each frozen reference vertex by its full graph-distance row, continuously interpolates those distance profiles with Shepard partition-of-unity weights, and compares profiles with the L-infinity norm. This preserves the graph metric exactly on reference vertices while removing old Voronoi-cell aliasing and boundary jumps.

Second, directly accumulating finite cells from interpolated off-atlas radii causes a self-cell discontinuity: the exact reference source excludes its zero-radius cell, while an infinitesimally displaced source can suddenly include that entire finite cell. Current `dte_backend.controller_value` therefore preserves the reference cumulative proper-volume profiles and extends the **profiles themselves** with the same partition of unity:

`D_hat_x(r) = sum_a lambda_a(x) D_a^A(r)`.

Realized return, occupancy/value separation, and Boltzmann reward values all use this same continuous field. The finite atlas remains a numerical landmark/quadrature device; proper volume remains `Omega(metric ball)` and is not redefined by either interpolation scheme.

## Failure boundaries

Reject zero-norm/nonfinite embeddings, invalid graph `k`, disconnected reference graphs, nonpositive quadrature weights, invalid occupancy, undersized reference atlases, and nonempty pre-numeric history that cannot prove atlas identity.

Default runtime never auto-fits a sampling-density correction. Experimental correction requires explicit positive `reference_density` input.

## Compatibility

Legacy RBF geometry, MMD helpers, empirical angular calibration, explicit nearest-neighbour diagnostics, and the hard off-atlas cell-centroid approximation remain useful only for preregistered falsification/backwards compatibility/history. They do not define current `new` proper-volume geometry.
