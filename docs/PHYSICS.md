# DTE New — Controller Physics

This file is the normative mathematical authority for the `new` release line. Ordinary refactors and feature work do not change it; a theory change requires explicit falsification evidence, an intentional spec update, and an update of the theory-lock test.

## Search state

A completed research transition is \(x_i=(m_i,\Delta\mathcal U_i)\): the retrospectively executed method plus the resulting epistemic change (`new_understanding`, `sharper_unknown`, or `no_material_change`). Current question/context `Q` is excluded from the canonical scoring embedding. Continuing a live parent retires that active transition and the completed child replaces its slot; the parent remains in history/provenance.

## Metric-measure geometry

The controller uses only a metric-measure space \((\mathcal M,d,\mu)\). Frozen transition embeddings are L2-normalized; angular edge length is \(θ=\arccos\langle\hat z_x,\hat z_y\rangle\). A symmetric kNN-union graph defines shortest-path intrinsic distance `d_g`; live/query transitions anchor to the nearest frozen reference location. No tangent vectors, metric tensor, Jacobian, explicit manifold dimension, or value gradient is required.

Raw angle/radius is a coordinate separation. The research displacement is cumulative crossed proper volume

\[
D_i(r)=\Omega_i(r)=\int_{0<d_g(i,x)\le r}d\Omega(x),
\]

with finite-sample quadrature \(D_i(r)=\sum_{a:0<r_{ia}\le r}\Delta\Omega_a\). It is not node-locally normalized and may exceed one. For parent→child,

\[
R=D_{parent}(d_g(parent,child)).
\]

## Value

`V_i` is expected historical realized proper-volume return, locally regressed in the same volume coordinate. A finite-sample kernel is

\[
w_{ij}=e^{-D_{ij}/h_V},\qquad V_i=\frac{\sum_jw_{ij}R_j}{\sum_jw_{ij}}.
\]

No history gives `V_i=0`. Judge scores remain research observations but are not controller value.

## Occupancy, entropy and uncertainty

For current live transitions,

\[
D_{ij}=D_i(d_g(i,j)),\qquad \rho_i=\frac1N\sum_j e^{-D_{ij}/h_V},\qquad S_i=-\log\rho_i.
\]

`h_V` is the frozen numerical `volume_bandwidth`, not a physical state size or extra UCB term.

On frozen atlas cells with radius `r_ia` and volume `ΔΩ_a`, use

\[
p_{ia}(T)=\frac{\Delta\Omega_a e^{-r_{ia}/T}}{Z_i(T)},
\]

\[
H_i(T)=-\sum_a p_{ia}(T)\log\frac{p_{ia}(T)}{\Delta\Omega_a},
\]

and solve `H_i(T_i)=S_i`. Push that Boltzmann mass through the same reward observable `A_ia=D_i(r_ia)` and define

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

## Frozen atlas

One run freezes its reference transitions/embeddings, sparse graph, common volume gauge, optional quadrature correction, and `volume_bandwidth`. Numeric returns from a different atlas are not reused without remeasurement. Default quadrature uses equal atlas-cell weights; caller-supplied positive density weights are experimental numerical correction only. Automatic reference-KDE correction is not part of the default controller.

## Superseded on `new`

The live `new` controller does not use direct claim embedding, ordinary RBF-KDE `1/sqrt(N rho)`, Judge score as `V`, MMD return as `V`, `V+T log 2`, node-local `[0,1]` volume normalization, a required `Omega_0`, automatic q-density correction, cold-entropy subtraction, free-volume replacement, canonical W1 value, or inverse-metric/tangent machinery. Those belong to repository history or the parallel old release line.