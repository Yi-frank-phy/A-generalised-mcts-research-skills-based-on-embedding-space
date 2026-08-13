# Requirements

Normative scope: isolated `src/dte_nextgen/thought_space/**`; production `src/dte_backend/**` is unchanged.

## Value observable

The completed search coordinate remains the canonical `(method, epistemic change)` transition. Source question/context is excluded from the scoring embedding.

Raw embedding angle or sparse-graph geodesic radius is only a coordinate separation. On one frozen reference atlas, node `i` defines cumulative crossed proper volume

`D_i(r) = integral over 0 < d_g(i,x) <= r of dOmega(x)`.

Finite-sample code evaluates this by quadrature over atlas cells. `D_i(r)` is not divided by the node-local accessible total and is not a literal microstate count.

For executed parent -> completed child, `R_i = D_i(d_g(parent, child))`. Historical expected realized proper-volume return is `V_i`.

## Uncertainty

Live occupancy uses only current live-frontier mass on the frozen atlas:

`D_ij = D_i(d_g(i,j))`

`rho_i = mean_j exp(-D_ij / h_V)`

`S_i = -log(rho_i)`.

`h_V` (`volume_bandwidth`) is a numerical estimator bandwidth, not a state size, pull count, novelty prior, or cost term.

Entropy matching produces a radial Boltzmann distribution over frozen-atlas cells. Push that distribution through the same reward variable `A_ia = D_i(r_ia)` and define controller uncertainty as the ordinary standard deviation of `A_ia`.

`U_i = V_i + SD[A_i]`.

Geometric `T_i log 2` may remain diagnostic but is not added directly to `V_i`.

## Geometry and measure

Use L2-normalized embeddings, angular edge length `arccos(cosine)`, a symmetric k-nearest-neighbour union graph, shortest-path geodesic distance, and nearest-reference anchoring for live/query embeddings.

No tangent vectors, metric tensor, inverse metric, Jacobian, explicit manifold dimension, or value-gradient field is required.

Default quadrature gives equal weight to frozen reference cells. Optional caller-supplied positive reference-density weights are experimental numerical quadrature correction only. Automatic live-frontier or automatic reference-KDE correction is not authoritative runtime behaviour.

## Stateful run

One proper-volume transition session freezes the reference transitions/embeddings, sparse graph/geodesic, common atlas volume gauge, optional quadrature correction, and `volume_bandwidth` for its lifetime.

The active frontier changes by replacement only: the executed parent leaves the active frontier and the completed child occupies that slot.

Before realized evidence exists, `V_i = 0`. Later `V_i` is a local proper-volume kernel regression over run-local historical realized returns. Historical evidence affects `V` only and never defines UCB uncertainty.

Already-numeric return history from a different frozen atlas must not be silently reused. Cross-run reuse requires raw transition evidence to be remeasured on the new atlas.

## Compatibility

Existing RBF-KDE `SD=1/sqrt(N rho)` and one-replacement MMD return remain callable only as compatibility/falsification baselines in the isolated next-generation package.

Unit and synthetic tests establish implementation contracts only; research effectiveness remains an equal-budget experiment gate.
