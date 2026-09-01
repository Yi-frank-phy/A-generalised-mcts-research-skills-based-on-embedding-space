# Proper-volume controller requirements

> **Current geometry authority:** [`../../PHYSICS.md`](../../PHYSICS.md). This file has been refreshed after the continuum repair. The original PR #48 requirements of nearest-reference anchoring and hard off-atlas cell inclusion are superseded.

## Value observable

The completed search coordinate remains the canonical `(method, epistemic change)` transition. Source question/context is excluded from the scoring embedding.

Proper volume is the measure of a source-centred research-space metric ball:

`D_x(r) = integral over 0 < d(x,y) <= r of dOmega(y)`.

It is not a reciprocal-density/KDE transform and is not a literal microstate count. Finite code estimates the measure from a frozen atlas.

For executed parent -> completed child, the finite controller evaluates `R = D_hat_parent(d_hat(parent, child))`. Historical expected realized proper-volume return is `V`.

## Uncertainty

Live occupancy uses the same continuous proper-volume field:

`D_ij = D_hat_i(d_hat(i,j))`

`rho_i = mean_j exp(-D_ij / h_V)`

`S_i = -log(rho_i)`.

`h_V` (`volume_bandwidth`) is a numerical estimator bandwidth, not a state size, pull count, novelty prior, or cost term.

Entropy matching produces a radial Boltzmann distribution over frozen-atlas cells. Push that distribution through the same reward field `A_ia = D_hat_i(r_ia)` and define controller uncertainty as the ordinary standard deviation of `A_ia`.

`U_i = V_i + SD[A_i]`.

Geometric `T_i log 2` may remain diagnostic but is not added directly to `V_i`.

## Distance geometry

The finite geometry estimator SHALL:

1. L2-normalize frozen reference embeddings;
2. use angular local edge length `arccos(cosine)`;
3. build a symmetric k-nearest-neighbour union graph and shortest-path metric on reference vertices;
4. preserve those reference-vertex graph distances exactly;
5. extend the graph metric continuously to arbitrary live/query embeddings without hard nearest-cell snapping;
6. introduce no required tangent vectors, metric tensor, inverse metric, Jacobian, explicit manifold dimension, or value-gradient field.

The current implementation represents reference vertex `a` by its distance-to-landmarks row `G[a,:]`, interpolates those profiles with continuous Shepard partition-of-unity weights, and measures profile separation in the L-infinity norm. A query's profile component for landmark `b` SHALL equal its finite query-to-`b` distance.

`nearest_reference_indices(...)` may remain only as an explicit legacy/diagnostic helper. It MUST NOT define authoritative live/query distance, realized return, value regression, occupancy, or controller SD.

## Proper-volume field

For each frozen reference source `a`, finite quadrature plus radius interpolation SHALL define a cumulative reference profile `D_a^A(r)`.

Arbitrary off-atlas sources MUST NOT construct proper volume by hard membership of finite atlas-cell centroids. Instead the current estimator SHALL extend the reference scalar profiles with the same partition of unity:

`D_hat_x(r) = sum_a lambda_a(x) D_a^A(r)`.

The finite proper-volume field SHALL be continuous in source and radius, monotone in radius, zero at radius zero, and exact on frozen reference sources. Realized return, historical/local value geometry, live occupancy, and Boltzmann reward values MUST use this same field.

This requirement prevents the finite self-cell discontinuity in which an exact reference source excludes a zero-radius cell but an infinitesimally displaced source suddenly includes that entire finite cell volume.

## Measure and quadrature

Default quadrature gives equal weight to frozen reference cells. Optional caller-supplied positive `reference_density` values are experimental numerical quadrature correction only. Automatic live-frontier or reference-KDE density correction is not authoritative runtime behaviour.

The finite atlas is a landmark/quadrature device, not a discrete ontology. Changing atlas resolution should ultimately change approximation error rather than controller semantics.

## Stateful run

One proper-volume transition session freezes the reference transitions/embeddings, sparse graph/geodesic, common atlas volume gauge, optional quadrature correction, and `volume_bandwidth` for its lifetime.

The active frontier changes by replacement only: the executed parent leaves the active frontier and the completed child occupies that slot.

Before realized evidence exists, `V_i = 0`. Later `V_i` is a local proper-volume kernel regression over run-local historical realized returns. Historical evidence affects `V` only and never defines UCB uncertainty.

Already-numeric return history from a different frozen atlas must not be silently reused. Cross-run reuse requires raw transition evidence to be remeasured on the new estimator/atlas.

## Continuum acceptance tests

The finite estimator SHALL be falsified under atlas refinement/resampling. At minimum test:

- exact preservation of on-atlas graph distances;
- exact preservation of reference proper-volume profiles;
- nonzero response to motion inside an old nearest-reference cell;
- removal of finite jumps across old Voronoi boundaries;
- continuity of proper-volume return when a source leaves a reference vertex;
- continuity of Boltzmann reward values under the same perturbation;
- pairwise proper-volume displacement convergence;
- occupancy/entropy/SD convergence;
- frontier ranking/allocation stability;
- graph-distance sensitivity to atlas refinement;
- finite measure/common volume-gauge sensitivity to atlas refinement;
- locality/stability of the partition-of-unity extension as atlas sampling changes.

The last three remain numerical-consistency questions and do not reopen the proper-volume definition.

## Compatibility

Existing RBF-KDE `SD=1/sqrt(N rho)`, one-replacement MMD return, empirical angular calibration, hard nearest-reference anchoring, and hard off-atlas cell inclusion remain callable only as compatibility/falsification/history where applicable. They are not current `new` controller authority.

Unit and synthetic tests establish implementation contracts only; research effectiveness remains an equal-budget experiment gate.
