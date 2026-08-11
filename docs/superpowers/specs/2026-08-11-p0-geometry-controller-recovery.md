# P0 DTE geometry/controller mathematics recovery

## Status

This specification supersedes the 2026-08-10 `H/log(N) -> T` controller restoration for the geometry/controller formulas below. It does **not** redesign the public Skill into the prospective-thought prototype and does not remove Judge/Executor in this patch.

The production compatibility interpretation is:

- current Judge score remains the provisional value input `V_i`;
- embedding geometry supplies the absolute uncertainty input `SD_i`;
- UCB remains exactly `U_i = V_i + SD_i`;
- current frontier geometry supplies a bounded soft-discrete entropy target;
- Boltzmann temperature is solved from that target for the **current UCB spectrum**, rather than linearly scaled from `H/log(N)`.

## 1. Restored geometric scale

For normalized embedding vectors `z_i`, use squared Euclidean distance

\[
d_{ij}^2=\lVert z_i-z_j\rVert^2.
\]

Let

\[
d_{\rm med}^2=\operatorname{median}_{i<j} d_{ij}^2.
\]

The legacy high-dimensional scale is

\[
\boxed{h=\frac{d_{\rm med}}{\sqrt2}},
\qquad
\boxed{h^2=\frac{d_{\rm med}^2}{2}}.
\]

Thus a typical pair has

\[
K_h(d_{\rm med})=
\exp\!\left(-\frac{d_{\rm med}^2}{2h^2}\right)=e^{-1}.
\]

The pre-P0 production implementation `bandwidth2 = median(d^2)` made the typical overlap `e^{-1/2}` and was therefore a factor-of-two scale regression.

## 2. Soft-discrete density and entropy

Use the self-including normalized RBF soft count

\[
\boxed{
\rho_i=\frac1N\sum_j
\exp\!\left(-\frac{d_{ij}^2}{2h^2}\right)
}.
\]

Because the self kernel is one,

\[
\frac1N\le\rho_i\le1.
\]

Define

\[
\boxed{
H_{\rm geom}=-\frac1N\sum_i\log\rho_i
},
\]

so

\[
0\le H_{\rm geom}\le\log N.
\]

This is a kernel-smoothed continuation of finite discrete Shannon entropy, not a coordinate-dependent `D`-dimensional differential entropy. No Gaussian density-normalization factor belongs in it.

## 3. Absolute uncertainty

At cold/current-frontier geometry, the effective local evidence count is

\[
\boxed{n_{{\rm eff},i}=N\rho_i}.
\]

The provisional absolute uncertainty is

\[
\boxed{
SD_i=\frac1{\sqrt{N\rho_i}}
}.
\]

Because `rho_i >= 1/N`, this already lies in `(0, 1]`.

Do **not** min-max normalize `-log rho_i` inside the current batch. Batch min-max normalization destroys absolute evidence scale: the densest branch is forced to uncertainty zero and the sparsest branch to one even when all branches are nearly equally dense or nearly equally isolated.

## 4. UCB

Keep

\[
\boxed{U_i=V_i+SD_i}.
\]

Temperature does not multiply `SD_i` and does not appear inside UCB.

## 5. Geometry entropy to Boltzmann temperature

For fixed current UCB values, define

\[
p_i(T)=\frac{e^{U_i/T}}{\sum_j e^{U_j/T}}
\]

and allocation entropy

\[
H_B(T)=-\sum_i p_i(T)\log p_i(T).
\]

For non-degenerate UCBs,

\[
\frac{dH_B}{dT}=\frac{\operatorname{Var}_{p_T}(U)}{T^3}>0.
\]

Therefore the current controller hypothesis is the uniquely invertible closure

\[
\boxed{H_B(T_t)=H_{\rm geom}(t)}.
\]

`T` must be solved from the current `U_i` values and current `H_geom`; it cannot be obtained from

\[
T=T_{\max}H/\log N,
\]

because the same entropy target requires different temperatures when the UCB spectrum is rescaled.

For exactly degenerate UCBs, Boltzmann allocation is uniform at every positive temperature and the only feasible allocation entropy is `log N`; the implementation may return a finite compatibility temperature while recording that the target was projected to the feasible uniform state.

## 6. Scope boundary

This P0 changes only the deterministic geometry/controller mathematics and the documentation/tests that define it.

It does **not** yet:

- replace Judge `V_i` with historical geometric-return regression;
- migrate to prospective-thought arms;
- introduce non-equilibrium stopping;
- make per-iteration allocation mass geometry-determined;
- change the existing global hard node budget.

Those remain separate research/design changes.

## 7. Verification evidence

The recovery was implemented test-first on PR #40. Characterization tests first demonstrated the pre-P0 regressions: the KDE scale test expected half the median squared pair distance while production returned the full median, and the absolute-SD test expected finite nonzero uncertainty for every branch while production min-max normalized the batch to endpoint values.

A separate controller characterization then required temperature to scale with the current UCB spectrum while reproducing the same target Boltzmann entropy. Production initially could not satisfy this contract because the entropy controller did not accept UCB scores at all.

After restoring the formulas and wiring both the App controller, replay validation, and legacy runner to the current UCB spectrum, the production bundle manifest was regenerated and verified. The branch then passed the full repository regression suite with **643 passed** on Python 3.12 before the final production-state commit `41953f2` was pushed.
