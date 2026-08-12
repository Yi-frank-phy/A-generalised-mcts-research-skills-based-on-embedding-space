# Tail-resolution and k-duplicate occupancy probe — 2026-08-12

## Scope

Geometry/occupancy experiment only. Do **not** wire this directly into MMD or any PSD-dependent construction. The candidate under test is

\[
q_{ij}=1-F_M(c_{ij}),\qquad
K_N(q)=\exp\!\left[-\binom N2 q\right],\qquad K_{ii}=1.
\]

The null CDF is empirical and frozen for the probe. The canonical 3D coordinate remains an interpretation of the same tail probability,

\[
s_3=1-2q,\qquad \alpha_3=\arcsin(s_3),
\]

but the occupancy calculation uses `q` directly so deep-tail precision is not lost by bounded-angle saturation.

## Clean k-duplicate run

Model: `gemini-embedding-001`, 128-dimensional Matryoshka output, manually L2-normalized.

Calibration corpus: 6 broad method families × 8 method↔epistemic-change transitions. Only cross-family pairs were used as null/background pairs, giving

- `n_null = 960`
- empirical tail floor `q_min = 1/961 = 0.0010405827`

Probe: one `symmetry constraint` transition plus four independently serialized near-paraphrases of the same `(m, ΔU)` direction.

All 10 pairwise comparisons among the five duplicate/near-duplicate records landed at the empirical tail floor:

\[
q_{ij}=q_{\min}\quad\text{for all duplicate pairs.}
\]

For each scenario, `N rho_i = 1 + sum_{j != i} K_ij` was evaluated for the duplicate nodes while the remaining frontier nodes were selected from other method families.

### N = 12

| duplicate count `k` | mean `N rho` | min | max |
|---:|---:|---:|---:|
| 1 | 1.3658 | 1.3658 | 1.3658 |
| 2 | 2.3031 | 2.2994 | 2.3068 |
| 3 | 2.8916 | 2.8679 | 2.9072 |
| 5 | 4.7997 | 4.7345 | 4.9873 |

At this calibration depth,

\[
K_{\rm floor}(N=12)=\exp[-66/961]\approx0.9336.
\]

The duplicate-only contributions therefore predict `1 + (k-1) K_floor`; the observed values are close, with residual background mass accounting for the remaining difference.

### N = 25

| duplicate count `k` | mean `N rho` | min | max |
|---:|---:|---:|---:|
| 1 | 1.0104 | 1.0104 | 1.0104 |
| 2 | 1.7522 | 1.7423 | 1.7622 |
| 3 | 2.4773 | 2.4637 | 2.4941 |
| 5 | 3.9390 | 3.9274 | 3.9717 |

Here

\[
K_{\rm floor}(N=25)=\exp[-300/961]\approx0.73185.
\]

Thus the duplicate-only prediction is

\[
1+(k-1)K_{\rm floor},
\]

which gives `1.7319`, `2.4637`, and `3.9274` for `k=2,3,5`; the measured means `1.7522`, `2.4773`, and `3.9390` track those values closely. This is strong evidence that the current shortfall from `N rho ~= k` is dominated by finite null-tail resolution rather than an unexpected failure of the occupancy algebra.

## Required calibration depth

If a true same-direction pair only establishes that it lies beyond every calibration-null pair, the empirical floor is

\[
q_{\min}=\frac1{M+1},
\]

where `M` is the number of null pairs. Therefore

\[
K_{\rm floor}=\exp\!\left[-\frac{\binom N2}{M+1}\right].
\]

To guarantee a floor overlap at least `K*`, one needs

\[
M+1\ge \frac{\binom N2}{-\log K_*}.
\]

Concrete requirements:

| N | target floor overlap | minimum null-pair count M |
|---:|---:|---:|
| 12 | 0.95 | 1,286 |
| 12 | 0.99 | 6,566 |
| 25 | 0.95 | 5,848 |
| 25 | 0.99 | 29,849 |

These are small enough for an offline frozen calibration corpus.

## Attempted deeper calibration

A follow-up run attempted to increase the null pool to about 4,950–6,900 pairs using roughly 100–140 distinct transitions and 128-dimensional embeddings.

This did **not** produce a scientific result because the execution environment hit Gemini API limits before a valid combined embedding set was available:

- `batchEmbedContents` accepts at most 100 requests per batch;
- the free `gemini-embedding-2` quota returned HTTP 429 with a 1,000-request per-model limit;
- the free `gemini-embedding-001` path exposed a 100-request limit for the underlying embedding-1.0 quota, so the second half of a split batch was rejected.

Malformed/error responses from these failed attempts were explicitly discarded and must not be interpreted as geometry measurements.

## Current judgment

Supported as a working hypothesis:

1. model-specific null CDF calibration is still the cleanest model-independent interface;
2. storing `q` (or a monotone transform such as `-log q`) internally is preferable to storing only the bounded 3D-equivalent angle, because the angle saturates in deep positive tails;
3. `K_N(q)=exp[-binom(N,2) q]` passes a nontrivial structural k-duplicate check: measured duplicate occupancy follows the finite-tail-floor prediction closely;
4. at `N=25`, the present `960`-pair calibration is demonstrably too shallow to recover `N rho ~= k` for `k>1` because the maximum resolvable duplicate overlap is only about `0.732`;
5. a frozen background with roughly `6k` null pairs is enough to raise that censored duplicate overlap above `0.95`, while roughly `30k` pairs raises it above `0.99`.

Still open:

1. run the same k-duplicate test with a >=6k null corpus once API quota permits;
2. repeat the clean k-duplicate probe on `gemini-embedding-2`, whose earlier 960-null experiment already placed same-direction pairs beyond the observed null tail but did not include the full duplicate clique;
3. test several distinct same-direction anchors rather than one symmetry anchor;
4. validate that the frozen null corpus itself is semantically defensible and not dominated by template artifacts;
5. do not promote this pairwise occupancy kernel to MMD/RKHS use unless positive semidefiniteness is separately established.
