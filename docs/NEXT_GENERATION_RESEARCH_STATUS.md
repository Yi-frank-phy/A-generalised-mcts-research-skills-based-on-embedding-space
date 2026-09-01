# DTE `new` release research status

> **Current status: 2026-09-01.** `new` is the standalone release line created by PR #49. Its production implementation is under `src/dte_backend/**`. The older `src/dte_nextgen/**` tree is retained as a historical/prototype mirror and is not the authority when it disagrees with `docs/PHYSICS.md` or `src/dte_backend/**` on `new`.

## Mathematical authority

Read these first:

1. [`PHYSICS.md`](PHYSICS.md) — normative controller mathematics;
2. [`PROPER_VOLUME_GEOMETRY.md`](PROPER_VOLUME_GEOMETRY.md) — canonical geometry lineage, continuum/finite-estimator distinction, and historical traps.

The microscopic scoring coordinate remains a completed retrospective `(method, epistemic change)` transition. `Q` is context only. Used parents leave the active frontier and remain in history/provenance; the completed child replaces the active slot.

## Proper-volume controller

Proper volume is the measure of a source-centred metric ball:

```text
D_x(r) = Omega({y : 0 < d(x,y) <= r}).
```

It is **not** defined by observed sample density, KDE, cosine-percentile calibration, or a literal microstate count.

For an executed parent -> child:

```text
R = D_parent(d(parent, child)).
```

Historical realized proper-volume returns define `V`. Current-frontier occupancy is computed in the same proper-volume coordinate; entropy matching produces a radial Boltzmann distribution; that distribution is pushed through the same `D_x(r)` reward and its ordinary SD becomes controller uncertainty. Therefore:

```text
U = V + SD
```

uses one common observable rather than an empirical scale bridge.

## Finite geometry realization

The frozen atlas remains a numerical landmark/quadrature structure:

1. L2-normalize reference embeddings;
2. use angular local edge length `arccos(cosine)`;
3. build a symmetric kNN-union graph;
4. compute shortest-path graph geodesics between reference vertices;
5. continuously extend the graph metric to arbitrary live/query points with interpolated distance-to-landmark profiles.

Nearest-reference anchoring is now legacy/diagnostic only. The current continuum repair represents reference vertex `a` by its full graph-distance profile `Phi(a)=G[a,:]`, continuously interpolates those profiles for off-atlas queries, and measures query-query separation with the L-infinity distance between interpolated profiles. The construction preserves every frozen reference-vertex graph distance exactly while removing Voronoi-cell aliasing/jumps for off-atlas queries.

No tangent vectors, metric tensor, inverse metric, Jacobian, explicit manifold dimension, or value-gradient field is required.

## Proper-volume quadrature

Finite code estimates `D_x(r)` by accumulating atlas cell volumes inside the estimated source-centred radius and interpolating between sampled radii.

Default quadrature uses equal frozen atlas-cell weights. Positive caller-supplied `reference_density` values remain optional numerical quadrature correction only. Automatic reference-KDE correction is not authoritative controller behaviour, and atlas density must not be silently interpreted as embedding compression.

The frozen atlas-wide volume gauge is still a finite numerical convention. Refinement/resampling consistency of that gauge is a separate convergence question, not a redefinition of proper volume.

## Stateful execution loop

The release implementation freezes one run's reference atlas, graph, volume gauge, optional quadrature correction, and `volume_bandwidth`. Before realized evidence exists, `V=0`. Historical evidence informs `V` only; current-frontier geometry supplies the uncertainty term.

```text
history -> V -> current-frontier occupancy/entropy/SD -> U -> select
-> external continuation -> realized proper-volume return
-> record retired parent -> replace slot with completed child -> rescore
```

Already-numeric returns from another atlas are not silently reused. Persist raw transition evidence so it can be remeasured as finite geometry estimators improve.

## Geometry issue resolved vs still open

### Continuum query bug repaired

The PR #48 implementation snapped every arbitrary live/query embedding to one nearest reference vertex. That zero-order approximation made same-cell queries identical and produced finite jumps across old Voronoi boundaries. The `new` release now uses a continuous off-atlas extension instead while retaining exact on-atlas graph geometry.

### Numerical convergence work still required

The following remain falsification/refinement questions:

- sparse angular graph-distance convergence under atlas refinement/resampling;
- finite measure/common volume-gauge convergence under atlas refinement;
- stability of `D`, occupancy, entropy, SD, UCB ranking and allocation for fixed off-atlas queries as the atlas is refined.

These questions are tracked in issue #54. They do not authorize returning to automatic density calibration or differential-geometric machinery.

## Legacy/falsification baselines

The following are historical, diagnostic, or falsification baselines rather than current `new` authority:

- empirical angular calibration `C_0(c)=2F_0(c)-1` from PR #42;
- ordinary RBF-KDE `1/sqrt(N rho)` uncertainty;
- one-replacement MMD value;
- direct `V + T log 2`;
- node-local `[0,1]` volume normalization;
- literal state-count / `Omega_0` interpretation;
- automatic q-density/KDE correction;
- free-volume or canonical W1 replacement;
- inverse-metric/tangent machinery;
- hard nearest-reference query anchoring.

Passing unit/CI contracts establishes implementation consistency only. Research effectiveness still requires matched equal-budget falsification against simpler controllers.
