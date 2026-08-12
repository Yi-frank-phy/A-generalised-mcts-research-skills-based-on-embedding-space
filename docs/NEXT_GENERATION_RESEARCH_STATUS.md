# Next-generation DTE research status

> **Status note only.** This document is not an implementation specification and does not change the public v1 controller.

The public `main` branch remains the stable compatibility baseline for the current Judge -> controller -> Executor workflow. The next-generation research line is mirrored separately so that executable TDD can advance without silently changing `src/dte_backend/**`.

## What remains true in public v1

- DTE is the only outer controller.
- Model-produced role outputs are bounded and validated before state mutation.
- The current compatibility controller retains its existing Judge-derived value, geometry-derived exploration, Boltzmann allocation, hard budget caps, Relation/provenance machinery, and App-native lifecycle.
- Passing CI establishes protocol behaviour, not research effectiveness.
- `src/dte_backend/**` remains the production Skill-bundle identity and is intentionally untouched by the isolated next-generation prototype.

## Current next-generation authority

The authoritative next-generation microscopic search coordinate is a completed retrospective **method ↔ epistemic-change transition**.

A completed research move may retain source context/tension `Q`, method/intervention description `m`, and epistemic change `ΔU`, but the canonical embedded coordinate is exactly

```text
(m, ΔU)
```

`Q` is context only and never enters canonical embedding text.

`ΔU` supports three canonical change kinds:

- `new_understanding`;
- `sharper_unknown`;
- `no_material_change`.

Prospective thoughts remain pre-execution intervention proposals only. They are not the authoritative KDE/UCB/entropy/MMD search-space coordinate.

## Closed frontier lifecycle

The active frontier contains completed transition nodes that have **not yet been continued**.

For a normal one-parent -> one-child continuation:

```text
Z- = (..., z_i, ...)
```

becomes

```text
Z_i+ = (..., z_i', ...)
```

The used parent leaves the active frontier permanently and remains only in full tree/history provenance. The completed child replaces that frontier slot. Normal continuation is replacement, never append.

This keeps archive mass out of the live geometric observable.

## Closed controller semantics

For the current active transition frontier `Z={z_1,...,z_N}`, the runnable baseline computes a nonnegative overlap field `rho_i` and uses

```text
SD_i = 1 / sqrt(N * rho_i)
U_i = V_i + SD_i
H_geom = -mean(log rho_i)
```

The intended semantics are now explicit:

- `SD` is directional underoccupation / inertia from the **current live frontier geometry**;
- repeated nearby live transition nodes reduce `SD` automatically;
- no return variance, second moments, historical-evidence KDE, effective-pull counter, novelty prior, or manual reheating belongs in this primitive;
- `V_i` is expected normalized displacement of the **whole active transition frontier** when direction `i` is continued.

One realized propulsion return is the direct replacement-frontier displacement

```text
r_i = MMD^2(Z-, Z_i+) / 2
```

using the frozen transition metric. The `/2` only normalizes biased RBF-MMD² from `[0,2]` to `[0,1]`.

No matched-null subtraction is part of the authoritative controller. `null_adjusted_geometric_return(...)` remains legacy/optional analysis only.

## Geometry realization under audit

The **occupancy semantics** of `rho_i`, the resulting directional-inertia interpretation of `SD_i`, and the microstate-Shannon interpretation of `H_geom` are current controller baselines. The exact overlap function/resolution that should instantiate those semantics is still under theoretical audit.

The current runnable implementation uses RBF overlap with

```text
h(Z) = median_pair_distance(Z) / sqrt(2)
```

This is a useful high-dimensional numerical scale normalization: a median pair has Gaussian overlap of order `exp(-1)`. It is **not** a derivation of the semantic resolution required by microstate occupancy.

In particular, an approximately equidistant frontier can make most off-diagonal overlaps stay near `e^-1` independently of frontier size, so the effective breadth can saturate instead of scaling like the number of genuinely distinct directions. The RBF median-bandwidth realization therefore remains runnable but must be treated as a calibration heuristic under audit.

## Experimental frozen angular calibration

The isolated next-generation package also exposes an experimental geometry layer:

```text
src/dte_nextgen/thought_space/angular.py
```

For nonzero embeddings it first removes magnitude by rowwise L2 normalization and evaluates pairwise cosine. A frozen empirical background random-pair cosine distribution `F_0` then defines the signed calibrated angular coordinate

```text
C_0(c) = 2 F_0(c) - 1
```

The reference distribution is frozen externally rather than fit from the current frontier, so the live frontier cannot redefine its own angular resolution each round.

The public tests verify, among other things:

- invariance to positive row rescaling after L2 normalization;
- exact simple cosine geometry;
- empirical recovery of ordinary cosine under a synthetic 3D-uniform reference;
- expansion of narrow high-dimensional-like cosine concentration;
- frozen background ordering and input validation.

This layer is **not wired into the authoritative controller**. `C_0(c)` is signed, while `rho_i` requires a nonnegative overlap. No mapping from the signed calibrated angular relation to the required occupancy overlap has been frozen yet.

## Executable public mirror

The isolated next-generation implementation lives under `src/dte_nextgen/**` and does not alter stable `src/dte_backend/**`.

Its public TDD coverage locks:

1. canonical `(m,ΔU)` transition serialization and `Q` exclusion;
2. prospective-thought authority quarantine;
3. current-live-frontier `SD=1/sqrt(N rho)` and entropy use of the same `rho`;
4. completed-transition scoring and one-action scheduling;
5. frozen transition metric identity;
6. used-parent retirement and replacement-only frontier construction;
7. direct replacement return equal to `MMD²/2`;
8. small replacement jitter scoring below a large frontier move;
9. the isolated frozen angular-calibration contract.

The transition-pair core and angular calibration were merged from Draft PR #42 after public CI run #253 passed Ubuntu Python 3.10-3.14, Windows Python 3.12/3.14, backend-wheel verification, and repository Skill-bundle verification.

## Superseded experimental assumptions

The following should no longer be treated as current next-generation theory authority:

- prospective-thought embeddings as the microscopic search coordinate;
- historical-density / return-variance / effective-pull interpretations of `SD`;
- appending executed children while keeping used parents in the active frontier;
- null-adjusted MMD as authoritative `V` return;
- the old prospective-thought + null-adjusted 12-pair preregistration.

The old real-trajectory protocol must not be run. A new v2 preregistration is required before labelled trajectory returns are inspected.

## Open but nonblocking

1. derive and validate the nonnegative occupancy overlap corresponding to the calibrated angular relation, then decide whether it replaces the RBF baseline;
2. derive a principled `Phi(H_geom,N)` / scheduler-breadth closure;
3. calibrate stopping thresholds, persistence, repeated-testing treatment, temporal statistics, and semantic frontier coverage in transition-pair space;
4. handle history non-stationarity for `V` and proposal reuse.

## Experiment-gated

5. test whether direct whole-active-frontier displacement in `(m,ΔU)` space actually predicts productive research strongly enough for the controller to exploit;
6. freeze a new v2 preregistration and only then run matched real trajectories.

## Migration policy

Public `main` should remain a stable, testable compatibility baseline while next-generation work stays isolated in `dte_nextgen` until the relevant theory and evidence gates are satisfied.

A next-generation component may migrate toward production only when its mathematical/statistical meaning is explicit, its architectural boundary is stable, and any effectiveness claim that requires empirical evidence has survived a predeclared test. Synthetic/unit tests establish implementation contracts; they do not by themselves establish research effectiveness.
