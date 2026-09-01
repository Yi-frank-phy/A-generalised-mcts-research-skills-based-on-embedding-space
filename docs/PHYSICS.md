# DTE New — Controller Physics

This file is the normative mathematical authority for the `new` release line. Ordinary refactors and feature work do not change it; a theory change requires explicit falsification evidence, an intentional spec update, and an update of the theory-lock test.

For the geometry lineage, implementation/continuum distinction, and historical traps, see [`PROPER_VOLUME_GEOMETRY.md`](PROPER_VOLUME_GEOMETRY.md).

## Search state

A completed research transition is \(x_i=(m_i,\Delta\mathcal U_i)\): the retrospectively executed method plus the resulting epistemic change (`new_understanding`, `sharper_unknown`, or `no_material_change`). Current question/context `Q` is excluded from the canonical scoring embedding. Continuing a live parent retires that active transition and the completed child replaces its slot; the parent remains in history/provenance.

## Metric-measure geometry

The controller uses only a metric-measure space \((\mathcal M,d,\Omega)\). Proper volume is defined by the measure \(\Omega\); it is **not** defined as inverse sample density, KDE mass, or a calibration of cosine percentiles.

Raw embedding angle/radius is only a coordinate separation. The continuum research displacement from source \(x\) is the proper volume of a metric ball,

\[
D_x(r)=\Omega_x(r)=\int_{0<d(x,y)\le r} d\Omega(y).
\]

For parent→child,

\[
R=D_{parent}(d(parent,child)).
\]

No tangent vectors, metric tensor, Jacobian, explicit manifold dimension, value gradient, or general-relativistic machinery is required.

### Frozen graph and continuous off-atlas extension

Frozen transition embeddings are L2-normalized; local angular edge length is

\[
\theta(x,y)=\arccos\langle\hat z_x,\hat z_y\rangle.
\]

A symmetric kNN-union graph supplies the finite reference-atlas shortest-path metric \(G_{ab}\). The graph is a numerical estimator/landmark structure, not a claim that research space itself is a discrete grid.

Hard nearest-reference query anchoring is **not** authoritative geometry. It is retained only as a legacy diagnostic helper. Arbitrary live/query transitions use a continuous landmark distance-profile extension of the frozen graph metric.

For reference vertex \(a\), define its graph-distance profile

\[
\Phi(a)=(G_{a1},\ldots,G_{aN}).
\]

Because \(G\) is a metric,

\[
\|\Phi(a)-\Phi(b)\|_\infty=G_{ab},
\]

so the reference graph is represented exactly in distance-profile space.

For an off-atlas query \(x\), use continuous Shepard partition-of-unity weights \(\lambda_a(x)\) over angular separation, with exact recovery at a coincident reference vertex, and define

\[
\widehat\Phi(x)=\sum_a\lambda_a(x)\Phi(a),
\qquad
\widehat d(x,y)=\|\widehat\Phi(x)-\widehat\Phi(y)\|_\infty.
\]

Thus reference-vertex distances are unchanged exactly, while off-atlas query geometry varies continuously instead of being piecewise constant over Voronoi cells. At finite atlas size \(\widehat d\) is a landmark-induced pseudometric estimator; graph-refinement convergence remains a numerical question rather than a new controller semantic.

The components of \(\widehat\Phi(x)\) are also exactly the finite query-to-reference radii. For reference landmark \(b\),

\[
\widehat d(x,b)=\widehat\Phi_b(x).
\]

The inequality \(\widehat d(x,b)\le\widehat\Phi_b(x)\) follows from the graph triangle inequality, while equality is attained in coordinate \(b\).

## Proper-volume quadrature and continuous source field

For a frozen reference source \(a\), finite atlas quadrature defines the reference cumulative proper-volume profile

\[
D_a^{\mathcal A}(r)
=
\sum_{b:\,0<G_{ab}\le r}\Delta\Omega_b,
\]

with the existing interpolation between sampled radii. This is the discrete numerical approximation of the continuum ball measure at reference source \(a\).

A second zero-order artifact would arise if an off-atlas source simply re-ran the hard cell-inclusion sum using its interpolated radii: a source exactly on a reference vertex excludes the zero-radius self cell, while an infinitesimal displacement can make that whole cell suddenly positive-radius mass. Therefore arbitrary sources use the same partition of unity to interpolate the **proper-volume profiles themselves**:

\[
\widehat D_x(r)
=
\sum_a \lambda_a(x)D_a^{\mathcal A}(r).
\]

This finite field is continuous in \(x\) and \(r\), monotone in \(r\), satisfies \(\widehat D_x(0)=0\), and exactly recovers the original reference profile at every atlas vertex. Realized displacement and live occupancy use

\[
R_{x\to y}=\widehat D_x(\widehat d(x,y)),
\qquad
D_{xy}=\widehat D_x(\widehat d(x,y)).
\]

For a Boltzmann atlas cell \(b\), the reward variable is likewise

\[
A_{xb}=\widehat D_x(\widehat d(x,b)),
\]

so value, occupancy, and controller SD all use the same continuous finite proper-volume field.

The default frozen reference measure assigns equal atlas-cell weights. Caller-supplied positive `reference_density` values are **optional numerical quadrature correction only**; relative cell weights are then inverse to that supplied sampling density. Automatic reference-KDE correction is not part of the controller definition.

Consequently:

- proper volume is the measure of a ball, not an observed-point-density transform;
- atlas density is not automatically interpreted as embedding compression;
- the finite atlas supplies landmarks, reference volume profiles, and quadrature cells, not an ontology of research states;
- continuity is obtained by extending the finite distance and proper-volume fields, not by inventing a new geometric ontology.

The current finite implementation freezes one common atlas-wide multiplicative volume gauge for a run. Atlas-refinement invariance of that gauge/measure is a separate numerical-consistency target and must not be confused with the definition of proper volume.

## Value

`V_i` is expected historical realized proper-volume return, locally regressed in the same volume coordinate. A finite-sample kernel is

\[
w_{ij}=e^{-D_{ij}/h_V},\qquad
V_i=\frac{\sum_jw_{ij}R_j}{\sum_jw_{ij}}.
\]

No history gives `V_i=0`. Judge scores remain research observations but are not controller value.

## Occupancy, entropy and uncertainty

For current live transitions,

\[
D_{ij}=\widehat D_i(\widehat d(i,j)),\qquad
\rho_i=\frac1N\sum_j e^{-D_{ij}/h_V},\qquad
S_i=-\log\rho_i.
\]

`h_V` is the frozen numerical `volume_bandwidth`, not a physical state size or extra UCB term.

On frozen atlas cells with source-centred radius `r_ia` and volume `ΔΩ_a`, use

\[
p_{ia}(T)=\frac{\Delta\Omega_a e^{-r_{ia}/T}}{Z_i(T)},
\]

\[
H_i(T)=-\sum_a p_{ia}(T)\log\frac{p_{ia}(T)}{\Delta\Omega_a},
\]

and solve `H_i(T_i)=S_i`. Push that Boltzmann mass through the **same** reward observable `A_ia=\widehat D_i(r_ia)` and define

\[
SD_i=\sqrt{\sum_a p_{ia}(T_i)(A_{ia}-\langle A_i\rangle)^2}.
\]

`T_i log 2` is diagnostic only.

## UCB and allocation

\[
U_i=V_i+SD_i.
\]

The action-allocation entropy target is

\[
H_{frontier}=\frac1N\sum_i(-\log\rho_i),
\]

which lies in `[0,log N]` under self-inclusive occupancy. Boltzmann allocation matches this target subject to hard run budgets. Cost is not an additive UCB penalty.

## Frozen atlas and remeasurement

One run freezes its reference transitions/embeddings, sparse graph, common volume gauge, optional quadrature correction, and `volume_bandwidth`. Numeric returns from a different atlas are not reused without remeasurement. Raw transition evidence remains the durable object that can be remeasured as finite geometry estimators improve.

The intended continuum object is atlas-independent. Therefore changing atlas resolution should eventually change approximation error, not the meaning of the observable. Two numerical questions remain explicitly separate from the removed nearest-anchor bug:

1. sparse angular graph distances must be tested for convergence/stability under atlas refinement;
2. the finite quadrature measure/common volume gauge must be tested for refinement/resampling consistency.

Neither question reopens the definition of proper volume or authorizes automatic density-based calibration.

## Superseded on `new`

The live `new` controller does not use direct claim embedding, ordinary RBF-KDE `1/sqrt(N rho)`, Judge score as `V`, MMD return as `V`, `V+T log 2`, node-local `[0,1]` volume normalization, a required `Omega_0`, automatic q-density correction, cold-entropy subtraction, free-volume replacement, canonical W1 value, inverse-metric/tangent machinery, or hard nearest-reference anchoring as authoritative live/query geometry. Those belong to repository history, diagnostics, or the parallel old release line.
