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

For an executed parent -> child, the finite controller evaluates the continuous proper-volume field at the continuous query separation:

```text
R = D_hat_parent(d_hat(parent, child)).
```

Historical realized proper-volume returns define `V`. Current-frontier occupancy uses the same `D_hat`; entropy matching produces a radial Boltzmann distribution; that distribution is pushed through the same continuous reward field and its ordinary SD becomes controller uncertainty. Therefore `U = V + SD` uses one common observable rather than an empirical scale bridge.

## Finite geometry realization

The frozen atlas remains a numerical landmark/quadrature structure:

1. L2-normalize reference embeddings;
2. use angular local edge length `arccos(cosine)`;
3. build a symmetric kNN-union graph;
4. compute shortest-path graph geodesics between reference vertices;
5. represent each reference vertex by its full distance-to-landmark profile;
6. continuously interpolate those profiles for arbitrary live/query points;
7. compare interpolated profiles in the L-infinity norm.

This preserves every frozen reference-vertex graph distance exactly while removing nearest-cell aliasing/jumps. For a reference landmark `b`, component `b` of an interpolated query profile is exactly the finite query-to-`b` distance.

No tangent vectors, metric tensor, inverse metric, Jacobian, explicit manifold dimension, or value-gradient field is required.

## Proper-volume quadrature and source interpolation

For each frozen reference source `a`, finite quadrature plus radius interpolation defines its cumulative proper-volume profile `D_a^A(r)`. Default quadrature uses equal frozen atlas-cell weights. Positive caller-supplied `reference_density` values remain optional numerical quadrature correction only; automatic reference-KDE correction is not authoritative controller behaviour.

Arbitrary off-atlas sources do **not** re-run a hard cell-centroid inclusion sum. That would reintroduce a finite self-cell discontinuity when a source leaves an atlas vertex. Instead the same continuous partition of unity extends the proper-volume profiles themselves:

```text
D_hat_x(r) = sum_a lambda_a(x) D_a^A(r).
```

The resulting field is continuous in source and radius, monotone in radius, zero at radius zero, and exact at every reference source. Realized return, live occupancy, value regression, and Boltzmann reward SD all use this same field.

Atlas density must not be silently interpreted as embedding compression. The frozen atlas-wide volume gauge remains a finite numerical convention; refinement/resampling consistency is a separate convergence question, not a redefinition of proper volume.

## Stateful execution loop

The release implementation freezes one run's reference atlas, graph, volume gauge, optional quadrature correction, and `volume_bandwidth`. Before realized evidence exists, `V=0`. Historical evidence informs `V` only; current-frontier geometry supplies the uncertainty term.

```text
history -> V -> current-frontier occupancy/entropy/SD -> U -> select
-> external continuation -> realized proper-volume return
-> record retired parent -> replace slot with completed child -> rescore
```

Already-numeric returns from another atlas are not silently reused. Persist raw transition evidence so it can be remeasured as finite geometry estimators improve.

## Geometry issue resolved vs still open

### Finite-grid continuity bugs repaired in PR #55

PR #48 contained two zero-order artifacts:

- hard nearest-reference snapping of arbitrary live/query sources;
- hard off-atlas cell inclusion, which could turn a whole finite self-cell on after an infinitesimal displacement from a reference vertex.

PR #55 replaces both with continuous partition-of-unity extensions while preserving the original reference graph metric and reference proper-volume profiles exactly.

### Numerical convergence work still required

The following remain falsification/refinement questions:

- sparse angular graph-distance convergence under atlas refinement/resampling;
- finite measure/common volume-gauge convergence under atlas refinement;
- locality/stability of the partition-of-unity extensions as atlas sampling changes;
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
- hard nearest-reference query anchoring and hard off-atlas cell inclusion.

Passing unit/CI contracts establishes implementation consistency only. Research effectiveness still requires matched equal-budget falsification against simpler controllers.
