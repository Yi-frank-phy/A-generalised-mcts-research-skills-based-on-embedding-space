# Proper-volume geometry — canonical reconstruction and continuum repair

This note is the durable companion to [`PHYSICS.md`](PHYSICS.md). Its purpose is to prevent the proper-volume controller from being repeatedly re-derived from superseded calibration branches.

## 1. Canonical definition

The scientific object is a metric-measure space with a research-space separation `d` and a volume measure `Omega`. Proper volume is the measure of a source-centred metric ball:

```text
D_x(r) = Omega({y : 0 < d(x,y) <= r}).
```

For an executed parent -> child transition:

```text
R = D_parent(d(parent, child)).
```

This is the controller's research-displacement observable.

**Do not redefine proper volume as any of the following:**

- reciprocal sample density;
- KDE density correction;
- a percentile calibration of cosine similarity;
- a literal count of ontological research microstates;
- a Riemannian metric tensor, Jacobian, or GR construction.

Density may enter only as an explicitly supplied quadrature correction when the frozen atlas was sampled non-uniformly relative to the chosen volume measure.

## 2. Why the controller uses proper volume

Raw embedding/geodesic radius is only a coordinate separation. The controller instead asks how much research-space volume a displacement crosses. This gives one common scalar observable for both exploitation and exploration:

```text
realized rollout -> proper-volume return R -> historical mean V
predicted radial distribution -> D_x(r) -> proper-volume distribution -> SD
U = V + SD
```

`V` and `SD` therefore refer to the mean and standard deviation of the same proper-volume reward variable. No empirical bridge is needed to make a geometric width commensurate with an unrelated reward.

## 3. Historical lineage — do not collapse these stages

### 2026-08-12 — public PR #42: empirical angular calibration

PR #42 introduced the frozen empirical map

```text
C_0(c) = 2 F_0(c) - 1
```

to spread concentrated high-dimensional cosine values using a background cosine CDF. It was explicitly an experimental calibration layer and was not the final occupancy/proper-volume geometry.

### 2026-08-13 — private PR #135: common-observable theory audit

The scale audit found that the then-current geometric exploration width and MMD value return were not naturally commensurate. The discussion considered two choices:

1. calibrate geometric uncertainty into reward units externally; or
2. redefine value and uncertainty on one intrinsic observable.

The second route was preferred. Several temporary candidates appeared during the audit (free-volume closure, W1/radius gauge, inverse-metric/covariance language), but they were subsequently superseded. The simplification checkpoint explicitly returned to a metric-measure construction and introduced cumulative `Omega_i(r)` as the common observable.

The final proper-volume decision also removed automatic reference KDE/density correction from controller authority.

### 2026-08-13 — public PR #48: proper-volume nextgen migration

PR #48 publicly implemented the resulting structure:

- sparse angular kNN graph and shortest-path reference geometry;
- cumulative proper-volume `D_i(r)`;
- realized return on `D`;
- occupancy on source-centred `D_ij`;
- entropy-matched Boltzmann distribution;
- ordinary SD after pushing that distribution through the same `D` reward;
- `U = V + SD`;
- equal-weight frozen reference measure by default;
- density correction only as optional numerical quadrature machinery.

## 4. The finite-grid bug that survived PR #48

The core `D_i(r)` observable was already conceptually continuous in radius and the finite code interpolated between sampled radii. The grid artifact came from a different layer:

```text
arbitrary live/query embedding
-> nearest reference vertex
-> use that vertex's graph-distance row
```

This made the *source/query identity* piecewise constant over Voronoi cells. Consequences:

- different off-atlas queries in one cell became geometrically identical;
- a tiny move across a cell boundary could produce a finite graph-distance jump;
- the continuous proper-volume derivation was therefore evaluated through a zero-order query quantizer.

Nearest anchoring was an implementation shortcut, not part of the proper-volume definition.

## 5. Continuum repair: landmark distance-profile extension

Let the frozen atlas graph metric be `G_ab`. Represent every reference vertex by its full distance-to-landmarks profile:

```text
Phi(a) = (G_a1, ..., G_aN).
```

For any metric matrix, this representation is isometric in the L-infinity norm:

```text
||Phi(a) - Phi(b)||_infinity = G_ab.
```

The upper bound follows from the triangle inequality; equality is attained by using landmark `a` or `b`.

For an arbitrary query `x`, continuously interpolate the reference profiles with Shepard partition-of-unity weights over angular separation:

```text
Phi_hat(x) = sum_a lambda_a(x) Phi(a),
sum_a lambda_a(x) = 1.
```

A query exactly coincident with a reference vertex receives that vertex's profile exactly. Define the finite off-atlas distance estimator by

```text
d_hat(x,y) = ||Phi_hat(x) - Phi_hat(y)||_infinity.
```

Properties of this repair:

- continuous in the query embeddings away from ordinary numerical singularities;
- exact on all frozen reference vertices;
- no hard Voronoi-cell snapping;
- symmetric, non-negative, and triangle-respecting as a profile-space pseudometric;
- no tangent space, manifold dimension, metric tensor, or semantic taxonomy is introduced.

The source-centred proper-volume radii are simply the components of `Phi_hat(x)`, and the existing cumulative-volume interpolation is then reused unchanged.

This is a finite estimator of the existing theory, not a new definition of proper volume.

## 6. Definition versus quadrature

Keep these layers separate:

```text
continuum definition:
    D_x(r) = integral over the metric ball of dOmega

finite geometry estimator:
    graph metric + continuous landmark-profile extension

finite measure estimator:
    atlas cell weights DeltaOmega_a

finite cumulative evaluation:
    sort/interpolate radii and accumulate DeltaOmega_a
```

Equal cell weights are a valid declared finite measure convention; they do not mean that observed point density is the definition of volume. If atlas sampling is known to be biased relative to the intended measure, caller-supplied density can correct quadrature. The controller must not infer "embedding compression" from point density by itself.

## 7. What is fixed now, and what is still numerical research

### Fixed by the continuum query repair

- live/query points are no longer rounded to one atlas vertex for authoritative geometry;
- on-atlas graph distances are preserved exactly;
- same-cell off-atlas motion is observable;
- old Voronoi-boundary jumps are removed from the query extension;
- value, occupancy and uncertainty continue to use one proper-volume observable.

### Still to test under atlas refinement

1. **Graph convergence:** does the sparse angular kNN shortest-path metric stabilize toward a useful continuum separation as the atlas is refined/resampled?
2. **Measure/gauge convergence:** does the chosen finite atlas measure/common volume gauge stabilize or admit a consistent rescaling as atlas size changes?
3. **End-to-end convergence:** do `D`, occupancy, entropy, SD, UCB ranking and allocation stabilize for fixed off-atlas queries under refined/resampled atlases?

These are numerical-consistency/falsification questions. They must not be used to reopen the settled definition of proper volume or to resurrect automatic density calibration without evidence.

## 8. Retrieval pointers

When reconstructing this history, use these sources in this order:

1. `docs/PHYSICS.md` — current normative mathematics;
2. this file — canonical lineage and continuum implementation distinction;
3. public PR #48 — proper-volume implementation migration;
4. private `deep-think-evolving` PR #135 — historical theory audit only;
5. public PR #42 — earlier empirical angular-calibration branch only.

Public issue #54 tracks the continuum repair/refinement work. Any future note that describes proper volume as *fundamentally an inverse-density transform* should be treated as superseded unless `PHYSICS.md` is intentionally changed with falsification evidence.
