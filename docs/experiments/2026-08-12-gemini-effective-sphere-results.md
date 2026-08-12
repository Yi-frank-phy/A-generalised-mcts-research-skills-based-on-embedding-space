# Gemini effective-sphere angular experiment — 2026-08-12

## Status

Geometry-only experiment. **Do not wire the tested quantile transform into controller occupancy, SD, entropy, UCB, temperature, MMD return, or production v1.**

Model: `gemini-embedding-001`, full 3072-dimensional output, `SEMANTIC_SIMILARITY` task type.

Input: 20 structurally different method↔epistemic-change directions, two near-paraphrases per direction, serialized with the nextgen canonical `(m, ΔU)` transition format. Same-direction pairs are sanity checks; cross-direction pairs are the null/background set.

The API key was supplied transiently at execution time and is not present in this repository or this report.

## 1. Raw Gemini angular geometry strongly rejects a nominal isotropic 3072-sphere

Across all 760 cross-direction null pairs:

- null mean cosine: `0.8527176`
- null SD: `0.0196806`
- near-same-direction mean cosine: `0.9661647`

A raw `d=3072` isotropic sphere is therefore not an acceptable null model. Gemini embeddings have a very large common angular component / anisotropy in this sample.

Fitting `d_eff = 1 / E[c^2]` directly to raw cosines is nonsensical here: it gives `d_eff ≈ 1.37`, and theoretical sphere quantiles disagree catastrophically with the observed null distribution.

## 2. In-sample mean centering makes the null much more sphere-like, but is biased

Subtracting the mean of the same 40 embeddings before L2 normalization gives:

- null mean: `-0.04603`
- null SD: `0.09996`
- `d_eff ≈ 82.67`
- observed/predicted fourth-moment ratio: `0.8656`
- maximum absolute fitted-sphere quantile gap on p={.01,.05,.25,.5,.75,.95,.99}: `0.0714`
- same-direction mean cosine: `0.7594`

Because the evaluated points also contribute to the fitted mean, this is only a diagnostic and not valid calibration evidence by itself.

## 3. Five-fold cross-fitted centering survives the leakage check

For each fold, the mean vector was estimated only from the other 16 method groups (32 embeddings). Held-out embeddings were centered by that external mean, normalized, and only within-fold cross-direction pairs were scored. Aggregating the five held-out folds gives 120 null pairs and 20 same-direction pairs.

Results:

- cross-fitted null mean: `0.02284`
- cross-fitted null SD: `0.07147`
- second-moment fitted `d_eff`: `178.97`
- fourth-moment observed/predicted ratio: `0.94045`
- maximum absolute fitted-sphere cosine quantile gap: `0.05056`
- same-direction mean cosine: `0.78007`
- same-direction minimum cosine: `0.68165`

Fold null means were `0.00884, 0.02444, 0.04489, -0.00532, 0.04134`.

Interpretation: after removing a frozen/common mean direction, the Gemini null is surprisingly compatible with an approximately isotropic sphere of effective angular dimension around O(10^2), far below the nominal 3072 dimensions. This is evidence for an effective-sphere *null model*, not evidence that all semantic geometry is spherical.

## 4. The proposed 3D-equivalent quantile cosine fails as a scale-preserving similarity

The candidate transform was

`S_d(c) = F_3^{-1}(F_d(c)) = 2 F_d(c) - 1`,

using the cross-fitted `d_eff ≈ 178.97`.

On the held-out null it did not become perfectly uniform:

- transformed null mean: `0.15761`
- transformed null SD: `0.55795`
- maximum tested quantile gap from Uniform[-1,1]: `0.22598`

More importantly, the transform saturates the semantically relevant positive tail. All 20 near-same-direction pairs mapped numerically to `S_d = 1`.

A separate set of related-but-not-identical method pairs confirms that this is not limited to paraphrases:

| relation | centered cosine | `S_d` |
|---|---:|---:|
| symmetry ↔ gauge fixing | 0.20305 | 0.99373 |
| symmetry ↔ selection rule | 0.16775 | 0.97560 |
| Fourier ↔ generating function | 0.34616 | 0.999998 |
| perturbation ↔ asymptotic balance | 0.24466 | 0.99907 |
| variational ↔ convex dual | 0.21663 | 0.99651 |
| invariant ↔ topology | 0.12462 | 0.90441 |
| counterexample ↔ contradiction | 0.40506 | 0.99999998 |
| dimensions ↔ asymptotic balance | 0.24080 | 0.99887 |
| graph separator ↔ recurrence | 0.20640 | 0.99456 |
| causal intervention ↔ Bayesian evidence | 0.17683 | 0.98242 |

Therefore the CDF/quantile map is useful as a **null-tail significance / rank coordinate**, but it is rejected as a drop-in continuous cosine replacement for DTE. It removes high-dimensional concentration by expanding the central null band so aggressively that ordinary meaningful similarities collapse against +1, changing downstream scale.

## Current conclusion

Supported:

1. raw Gemini cosine must not be interpreted with nominal high-dimensional sphere geometry;
2. frozen/background mean subtraction before L2 normalization is strongly motivated by this experiment;
3. after such centering, a low effective-dimensional sphere is a plausible null approximation worth further testing;
4. `F_d(c)` / tail probability is useful for asking how surprising a pair is relative to unrelated directions.

Rejected for now:

1. nominal `d=3072` sphere calibration;
2. `S_d(c)=2F_d(c)-1` as the similarity fed directly into existing entropy/SD/MMD/controller math;
3. any claim that the geometry problem is closed.

The next theory question is narrower: use the effective-sphere/null-tail information to correct *whether two points count as occupying the same direction* without replacing the continuous cosine scale everywhere. Any proposed occupancy construction must still satisfy the existing null-mass requirement and preserve the discrete limits of the entropy/SD formulas.
