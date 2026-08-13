# Next-generation DTE research status

> **Authoritative next-generation status as of 2026-08-13.** This document applies only to the isolated `src/dte_nextgen/**` research line. It does not change production `src/dte_backend/**` v1 controller authority.

## Public production boundary

The public production Skill backend remains `src/dte_backend/**`. Its role authority, persistence, budgets, hook-enforced lifecycle, Relation/provenance machinery, and current production controller are unchanged by this next-generation migration.

The public next-generation implementation lives under `src/dte_nextgen/**` so controller research can be executable and CI-tested without silently changing production behaviour. Production migration remains evidence-gated.

## Search coordinate and frontier lifecycle

The authoritative microscopic search coordinate is a completed retrospective method--epistemic-change transition:

```text
(method, epistemic change)
```

Source question/tension `Q` is context only and never enters canonical scoring text.

The active frontier contains completed transitions that have not yet been continued. Normal continuation is replacement: the executed parent leaves the active frontier and remains in history/provenance; the completed child occupies that slot.

## Proper-volume controller

Raw embedding angle or sparse-graph geodesic radius is only a coordinate separation. On one frozen reference atlas, node `i` defines cumulative crossed proper research-space volume

```text
D_i(r) = integral over 0 < d_g(i,x) <= r of dOmega(x)
```

with finite-sample quadrature over atlas cells. `D_i(r)` is not normalized by each node's local accessible total and is not interpreted as a literal count of ontological microstates.

For an executed parent -> completed child,

```text
R_i = D_i(d_g(parent, child))
```

and `V_i` is the historical expected realized proper-volume return.

Live current-frontier occupancy uses the same source-centred volume coordinate:

```text
D_ij = D_i(d_g(i,j))
rho_i = mean_j exp(-D_ij / volume_bandwidth)
S_i = -log(rho_i)
```

`volume_bandwidth` is a finite-sample estimator bandwidth on the frozen atlas. It is not a physical state size, pull count, novelty prior, or cost term.

For each node, entropy matching produces a radial Boltzmann distribution over the frozen atlas. Push that mass through the same reward variable

```text
A_ia = D_i(r_ia)
```

and use the ordinary standard deviation of `A_i` as controller uncertainty:

```text
U_i = V_i + SD[A_i]
```

Geometric `T_i log 2` is retained only as a radial diagnostic. It is not added directly to `V_i`.

## Metric-measure realization

The finite-sample geometry intentionally uses ordinary CS machinery rather than differential geometry:

1. L2-normalize reference embeddings;
2. angular edge length `arccos(cosine)`;
3. symmetric k-nearest-neighbour union graph;
4. shortest-path geodesic distance;
5. nearest-reference anchoring for live/query embeddings.

No tangent vectors, metric tensor, inverse metric, Jacobian, explicit manifold dimension, or value-gradient field is required.

Default quadrature assigns equal weight to frozen reference cells. A caller may explicitly provide positive `reference_density` values as experimental numerical quadrature correction. Automatic reference-KDE correction and any live-frontier calibration of that correction are not authoritative controller behaviour.

## Stateful execution loop

`ProperVolumeTransitionSession` is the authoritative stateful next-generation loop. One session freezes its reference atlas, geodesic matrix, common atlas volume gauge, optional quadrature correction, and `volume_bandwidth`.

Before realized evidence exists, `V_i=0`. Once run-local realized returns exist, live-node `V_i` is a local kernel regression over that history using proper-volume separation. Historical evidence affects `V` only; current-frontier entropy/Boltzmann geometry remains the sole source of UCB uncertainty.

Each iteration is:

```text
history -> V -> current-frontier SD -> U -> select
-> external continuation -> realized proper-volume return
-> record retired parent -> replace slot with completed child -> rescore
```

Already-numeric return history from a different frozen atlas must not be silently imported. Cross-run reuse should retain raw transition evidence and remeasure it on the new atlas.

## Legacy/falsification baselines

The older next-generation RBF realization

```text
SD_i = 1 / sqrt(N * rho_i)
```

and one-replacement `MMD^2/2` return remain callable in their existing modules for compatibility and falsification. They are no longer the authoritative proper-volume controller semantics.

The earlier prospective-thought geometry, direct `V + T log 2`, node-local `[0,1]` volume normalization, literal state-count/Omega0 interpretation, and automatic density-correction proposals are superseded.

## Public implementation and tests

The proper-volume path is isolated under `src/dte_nextgen/thought_space/**` with dedicated public tests for:

- absolute/non-node-normalized cumulative proper volume;
- reward SD on the same proper-volume variable as realized return;
- angular sparse-graph geodesic and nearest-reference anchoring;
- duplicate versus isolated live occupancy;
- end-to-end `U=V+SD` scoring on a frozen equal-weight reference atlas;
- stateful select -> execute -> record -> replacement -> rescore behaviour;
- proper-volume-local historical value regression;
- rejection of unscoped pre-numeric cross-atlas history.

Passing CI establishes implementation contracts and compatibility only. It does not establish research effectiveness.

## Remaining experiment gate

The next scientific question is empirical: under equal model/token budget, does the metric-measure/statistical-physics controller materially outperform simpler alternatives such as raw distance, the legacy MMD/RBF controller, no-geometry tree search, best-of-N, and sequential reflection?

Synthetic/unit tests may test invariants and robustness to atlas refinement or sampling perturbations. Real-trajectory effectiveness claims still require a predeclared matched comparison. If geometry/statistical physics does not produce material benefit, it should be simplified or removed rather than protected as theory.
