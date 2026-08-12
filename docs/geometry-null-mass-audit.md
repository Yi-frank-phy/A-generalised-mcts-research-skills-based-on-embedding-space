# Geometry null-mass audit

**Status:** theory constraint for any future replacement of the current runnable RBF occupancy geometry. This document does not change controller behavior.

## 1. Why this audit exists

The current DTE occupancy semantics are

\[
\rho_i=\frac1N\sum_{j=1}^{N}K_{ij},
\qquad
SD_i=\frac1{\sqrt{N\rho_i}},
\qquad
H_{\rm geom}=-\frac1N\sum_i\log\rho_i,
\]

with self-overlap `K_ii = 1`.

The intended physical/search meaning is:

- one genuinely isolated live research direction should remain approximately one effective microstate;
- `k` effectively duplicate live transition nodes in one otherwise isolated direction should give approximately `N rho_i = k`, hence `SD_i = 1/sqrt(k)`;
- adding many unrelated directions should increase effective breadth rather than create accumulated weak background overlap.

The previous `median distance / sqrt(2)` RBF bandwidth failed this last large-`N` sanity check because typical off-diagonal overlap stayed order-one.

## 2. Null-mass theorem

Assume an overlap realization satisfies

\[
K_{ii}=1,\qquad K_{ij}\ge 0.
\]

For a frontier whose other directions are unrelated to node `i`, let

\[
\mu_N=\mathbb E_{\rm null}[K_{ij}],\qquad j\ne i.
\]

Then exactly

\[
N\rho_i
=1+\sum_{j\ne i}K_{ij},
\]

so

\[
\boxed{\mathbb E[N\rho_i]=1+(N-1)\mu_N.}
\]

Therefore:

- if `mu_N -> mu > 0`, unrelated background mass grows linearly and the effective breadth eventually saturates;
- to keep total unrelated background contribution merely `O(1)`, one needs

\[
\boxed{\mu_N=O(1/N);}
\]

- to recover the strict isolated-direction limit `N rho_i -> 1`, one needs the stronger condition

\[
\boxed{\mu_N=o(1/N).}
\]

This is an architectural calibration requirement, not a property specific to RBF kernels.

## 3. Consequence for dimension-normalized cosine

The experimental frozen empirical angular calibration is

\[
C_0(c)=2F_0(c)-1.
\]

Under its own continuous frozen background null, the probability-integral transform makes `C_0` approximately uniform on `[-1,1]`.

A tempting nonnegative overlap would be

\[
K=[C_0]_+.
\]

But under a uniform `[-1,1]` null,

\[
\mathbb E[K]
=\frac12\int_0^1 c\,dc
=\frac14.
\]

Thus this naive mapping would again accumulate order-`N` unrelated mass and is rejected as an occupancy kernel, even though the signed calibrated angle remains useful as a geometry diagnostic.

Likewise, the affine mapping `(1+C_0)/2` has null mean `1/2` and is rejected.

## 4. Consequence for standard high-dimensional hubness corrections

High-dimensional nearest-neighbor literature uses secondary-distance methods such as Mutual Proximity (MP), Local Scaling (LS), and related approaches to repair distance concentration/hubness and asymmetric neighbor relations. These are important candidate **neighbor-ranking / relation-calibration** tools.

They are not automatically valid DTE occupancy kernels.

For example, in the homogeneous independent-distance approximation to Mutual Proximity, if

\[
q=F(d_{xy})
\]

is the distance quantile for a random unrelated pair, then `q` is uniform on `[0,1]` and

\[
MP=(1-q)^2.
\]

Hence

\[
\mathbb E_{\rm null}[MP]
=\int_0^1(1-q)^2dq
=\frac13.
\]

So even a hubness-corrected similarity may have an order-one null floor if all pairwise similarities are simply summed into `rho`.

This does **not** invalidate MP as a high-dimensional neighbor metric. It only shows that DTE's additive occupancy observable imposes an extra calibration requirement beyond nearest-neighbor ranking quality.

## 5. Resulting design boundary

Any future occupancy construction must separate two jobs:

1. **high-dimensional relation calibration** — determine whether two transition embeddings are unusually close relative to embedding-space background/hubness;
2. **occupancy sparsification / null control** — prevent the many unrelated pair relations from accumulating order-`N` positive mass.

Candidate mature tools for job (1) include empirical angular-background calibration, Mutual Proximity, and local scaling.

For job (2), the next audit should focus on sparse reciprocal-neighborhood constructions or statistically null-controlled edges rather than another dense positive kernel with a fixed floor.

No choice is frozen yet.

## 6. Hard acceptance tests for the next overlap

Before any candidate replaces RBF in `rho`, it must satisfy at least:

1. **identity:** identical transition embeddings have overlap 1;
2. **symmetry:** occupancy overlap is symmetric;
3. **nonnegativity:** `K_ij >= 0` so `rho` remains an occupancy mass;
4. **duplicate-count limit:** `k` effectively identical isolated live nodes yield `N rho approximately k`;
5. **null-mass scaling:** unrelated off-diagonal mean overlap is `O(1/N)` or smaller, preferably `o(1/N)`;
6. **breadth growth:** a growing frontier of unrelated directions has `H_geom = log N - O(1)` rather than entropy saturation;
7. **high-dimensional robustness:** related pairs are not discarded merely because raw cosine lies close to zero under concentration of measure;
8. **no live self-calibration loop:** the active frontier must not freely redefine its own semantic resolution each round unless that dependence is explicitly derived as part of the occupancy law.

Until a candidate passes these checks, the current RBF implementation remains runnable baseline under audit and the frozen angular layer remains experimental only.
