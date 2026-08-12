# Synthetic manifold density-normalization probe — 2026-08-12

## Question

Can a dense, smooth embedding manifold have a reliable local cosine metric while still being sampled very non-uniformly, and can that sampling-density distortion be removed without fitting a global null-CDF tail?

This probe is deliberately synthetic and API-free. It tests only the geometry idea, not the DTE controller.

## Experiment A: 1D circle with severe non-uniform sampling

Ground truth manifold: a unit circle. Cosine/ambient angle is exact on this manifold.

Sampling: 500 points, with 65–90% of points drawn from a concentrated von-Mises component and the remainder uniformly around the circle. Thus the manifold geometry is unchanged, but the observed point density is strongly non-uniform.

We compare the standard diffusion-map normalization with `alpha=0` against density-corrected `alpha=1` (Coifman–Lafon style). Eight independently randomized density profiles were tested.

Results averaged over the eight trials:

- recovery of the true first non-trivial circle eigenspace (`cos(theta), sin(theta)`), R^2:
  - alpha=0: `0.676`
  - alpha=1: `0.964`
- correlation between unsupervised recovered pairwise circular distances and true pairwise circular distances:
  - alpha=0: `0.898`
  - alpha=1: `0.990`

So the density-corrected operator recovered the intrinsic circle geometry far more faithfully even though the sample density was heavily distorted.

A separate volume-weight test used local kernel-density weights proportional to the inverse observed neighborhood density. Uniformity error was measured by total-variation distance from the true uniform angular volume measure over 24 equal bins, across 20 randomized trials:

- raw sampled points: mean TV error `0.597`
- inverse-local-density volume weights: mean TV error `0.113`
- alpha=1 diffusion stationary-volume weights: mean TV error `0.139`

Thus most of the apparent density distortion can be removed using only pairwise geometric information; no labelled null pairs or extreme-tail fit is required.

## Experiment B: 2D sphere

Ground truth manifold: the unit 2-sphere, again with cosine giving the exact ambient angular metric.

Sampling: 1,200 points; 80% from a strong north-pole concentration and 20% uniform. Equal-area bins were made from uniform `z=cos(theta)` bins and azimuth bins.

Across eight randomized runs with a small local kernel scale:

- raw sample mean TV distance from uniform spherical volume: `0.607`
- inverse-local-density corrected mean TV: `0.115`
- alpha=1 diffusion-volume corrected mean TV: `0.133`

The same effect therefore survives in a 2D manifold and is not just a 1D-circle artifact.

## Important limitation found

The density-corrected diffusion operator recovers the *global intrinsic geometry / volume measure* very well in these probes, but a naive attempt to turn only the first two diffusion coordinates into a local duplicate-occupancy kernel was not robust enough at finite sample size. Small-scale duplicate distances were still somewhat distorted even when global pairwise-distance correlation was near 0.99.

Therefore the current positive result supports **intrinsic-volume recovery**, not yet the full DTE occupancy construction.

In particular, do not yet replace the existing occupancy kernel with a two-coordinate diffusion embedding.

## Interpretation

These synthetic results support a different picture from global null-CDF calibration:

1. cosine can be a good local metric even when the observed embedding point cloud is extremely non-uniform;
2. much of that non-uniformity can be interpreted as a sampling/measure distortion rather than a failure of the local metric;
3. density-normalized manifold operators can recover an approximately uniform intrinsic volume measure from pairwise similarities alone;
4. this is a plausible route to a model-independent calibration layer that does not require thousands of labelled unrelated pairs;
5. the remaining DTE problem is how to define frontier occupancy *relative to the recovered intrinsic volume*, without destroying genuine frontier clustering.

## Current best theoretical target

Treat the embedding as a dense sample from an unknown smooth metric manifold whose local geometry is supplied by cosine. Recover a density-independent intrinsic volume measure from the similarity graph. Then define DTE search occupancy as frontier search mass per unit intrinsic semantic volume.

This is currently better supported than assuming the residual embedding geometry is a globally isotropic sphere.
