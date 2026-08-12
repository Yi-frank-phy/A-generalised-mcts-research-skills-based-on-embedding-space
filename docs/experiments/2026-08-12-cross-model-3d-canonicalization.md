# Cross-model 3D-equivalent angular canonicalization — 2026-08-12

## Scope

Geometry/occupancy experiment only. Do **not** wire this directly into MMD or any PSD-dependent construction. The goal is to test whether model-specific null geometry can be washed out before DTE applies its own fixed scale.

Models:

- `gemini-embedding-001`, 768-dimensional output, `SEMANTIC_SIMILARITY` task type.
- `gemini-embedding-2`, 768-dimensional output, prompt prefix `task: sentence similarity | query: ...`.

The same 12 method↔epistemic-change directions were used for both models, with two near-paraphrases per direction. Three 4-group folds were used. For each fold, the null CDF was estimated only from the other 8 method groups; held-out cross-direction pairs, same-direction pairs, and a small set of related-but-not-identical pairs were then mapped through that frozen fold-specific CDF.

Canonical coordinate:

\[
u = F_M(c),\qquad s_3 = 2u-1,\qquad \alpha_3 = \arcsin(s_3)\in[-90^\circ,90^\circ].
\]

Here `alpha_3` means 3D-equivalent signed angle relative to orthogonality. A finite empirical CDF cannot resolve arbitrarily deep positive tails; values beyond the largest calibration null should therefore be treated as right-censored, not as precise equal angles.

## Raw model geometries differ

Mean raw cross-direction cosine:

- Gemini Embedding 2: `0.84542`
- Gemini Embedding 001: `0.83454`

This re-confirms that raw cosine scales are model-specific nuisance geometry.

## Held-out canonical nulls are much closer across models

Gemini Embedding 2 held-out canonical null:

- n = 72
- mean angle = `10.76 deg`
- SD = `39.29 deg`
- quantiles 5/25/50/75/95% = `[-52.79, -27.40, 20.73, 43.65, 72.89] deg`
- fraction >45 deg = `0.236`
- fraction >60 deg = `0.083`

Gemini Embedding 001 held-out canonical null:

- n = 72
- mean angle = `10.08 deg`
- SD = `42.11 deg`
- quantiles 5/25/50/75/95% = `[-58.16, -27.40, 13.30, 39.58, 76.77] deg`
- fraction >45 deg = `0.250`
- fraction >60 deg = `0.097`

Interpretation: the two incompatible raw embedding spaces become substantially more comparable after model-specific null CDF calibration. The held-out null is not perfectly centered/uniformized in this small corpus, so the null definition/background corpus still needs work; this is not yet a production calibration claim.

## Same-direction and related pairs

All 12 near-paraphrase same-direction pairs in both models lie above the maximum resolved training-null tail. With the finite-sample mid-rank convention they all report `82.37 deg`; this should be read only as `beyond current background-tail resolution`.

For related-but-not-identical pairs:

Gemini Embedding 2:

- mean = `40.00 deg`
- fraction >45 deg = `0.50`
- fraction >60 deg = `0.25`

Gemini Embedding 001:

- mean = `46.66 deg`
- fraction >45 deg = `0.50`
- fraction >60 deg = `0.25`

The exact pairwise angles differ substantially for some relations, which is expected if the embedding models have different semantic competence. The important cross-model result is that a coarse fixed canonical threshold transfers much better than raw cosine thresholds.

## Candidate occupancy scale from the null-mass requirement

Let

\[
q_{ij}=1-F_M(c_{ij})
\]

be the positive-tail null probability. This is equivalent to the 3D-equivalent coordinate via

\[
s_3 = 1-2q.
\]

A parameter-light N-dependent soft occupancy candidate is

\[
K_N(q)=\exp\!\left[-\binom{N}{2}q\right],\qquad K_{ii}=1.
\]

This scale is chosen from the controller's null-mass requirement, not from any embedding model. Under an ideal calibrated null, `q ~ Uniform(0,1)`, so with `A = binom(N,2)`:

\[
\mathbb E[K_N]=\int_0^1e^{-Aq}\,dq=\frac{1-e^{-A}}{A}\sim \frac1A.
\]

Therefore the expected off-diagonal null mass per node is

\[
(N-1)\mathbb E[K_N]\sim \frac{2}{N},
\]

hence

\[
N\rho_i\to1,\qquad SD_i\to1,\qquad H_{\rm geom}\to\log N
\]

for unrelated directions. This satisfies the existing strict null-mass asymptotics without an embedding-specific bandwidth.

On the present finite-CDF experiment, for `N=12` the unresolved same-direction tail implies only a conservative lower bound around `K >= 0.747`, while every tested related-but-not-identical pair is at or below about `K = 0.416` in both models. This is encouraging but should not be over-read because the same-direction tail is censored by the small background sample.

## Current conclusion

Supported as a working hypothesis:

1. model-specific raw cosine should first be converted through a frozen null CDF;
2. the resulting 3D-equivalent angle is a reasonable model-independent canonical coordinate;
3. a fixed DTE scale can plausibly operate after this canonicalization;
4. for occupancy specifically, `K_N(q)=exp[-binom(N,2) q]` is attractive because its scale follows from the null-mass theorem rather than from a model-specific bandwidth.

Still open:

1. define a large, frozen, semantically defensible null/background corpus;
2. repeat the cross-model test with enough null pairs to resolve deep positive tails;
3. test k-duplicate / near-duplicate occupancy limits directly;
4. do not reuse this pairwise calibration as an MMD kernel unless PSD is separately established.
