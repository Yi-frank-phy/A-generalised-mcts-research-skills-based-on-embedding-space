# Next-generation DTE research status

> **Status note only.** This document is not an implementation specification and does not change the public v1 controller.

The public `main` branch remains the stable compatibility baseline for the current Judge -> controller -> Executor workflow. It is useful for controlled evaluation, maintenance, and reproducible public runs.

However, it is no longer accurate to describe the research architecture itself as permanently frozen. A separate next-generation prototype is actively re-examining the search object and controller theory. That work is intentionally not migrated into public `main` until its theory and evidence gates are strong enough.

## What remains true in public v1

- DTE is the only outer controller.
- Model-produced role outputs are bounded and validated before state mutation.
- Embedding geometry uses the recovered adaptive RBF soft-count construction.
- The current public controller retains Judge-derived value, geometry-derived exploration, Boltzmann allocation, hard budget caps, Relation/provenance machinery, and the App-native lifecycle.
- Passing CI establishes protocol behaviour, not research effectiveness.

## Important semantic qualification

The current public compatibility controller still uses the legacy notation

```text
n_eff,i = N * rho_i
SD_i = 1 / sqrt(n_eff,i)
U_i = V_i + SD_i
```

for its geometry-derived exploration channel.

`SD_i` in this public formula should be read as a **legacy compatibility uncertainty / exploration heuristic**. It is not an ordinary empirical standard deviation of observed returns, and it is not a calibrated standard error or confidence radius. The public runtime behaviour is unchanged; this note only narrows the statistical claim attached to the name.

Likewise, the current public closure

```text
H_B(T) = H_geom
```

is a controller hypothesis that maps current-frontier geometric breadth into scheduler breadth. It is not part of the original derivation of the soft-discrete entropy and should not be read as a theorem of statistical mechanics.

## Next-generation direction under theory audit

The experimental architecture currently studies a different search object: an explicit **prospective structural thought** produced before expensive execution, rather than relying on a retrospective Judge score as the primary value object.

The most stable architectural ideas under study are:

1. generate compact prospective structural notices before execution;
2. embed those notices to obtain local thought-space geometry;
3. preserve node identity and temporal evidence instead of collapsing siblings into one value;
4. authorize a small amount of compute, observe the result, then recompute before further allocation;
5. measure post-execution reorganization with one frozen temporal embedding/kernel metric rather than using round-local adaptive entropy as a convergence coordinate;
6. keep statistical objects separate: observed return variability, ignorance/novelty priors, scheduler breadth, and stopping evidence should not silently share one symbol merely because they all influence exploration.

This direction is still a **falsifiable candidate theory**, not a replacement public contract.

## Still unresolved before migration

The next-generation controller is deliberately holding several questions open rather than encoding attractive formulas prematurely:

- whether observed return variability should receive any positive scheduling bonus;
- the correct ignorance/novelty prior and whether proposal multiplicity is signal or redundant sampling;
- the geometry-to-scheduler breadth map;
- transfer/decay of historical evidence under changing research state;
- finite-sample calibration of projected relaxation, recurrence, and frontier coverage;
- whether null-adjusted geometric reorganization is actually predictive of productive research trajectories.

No public v1 behaviour should be rewritten solely to mirror an unresolved experimental formula.

## Migration policy

Public `main` should remain a stable, testable baseline. Next-generation changes should migrate only when they satisfy both:

1. **theory audit:** the mathematical/statistical meaning of each controller quantity is explicit and internally consistent;
2. **evidence gate:** the relevant empirical hypotheses survive predeclared real-trajectory falsification rather than only synthetic tests.

Until then, public v1 and the experimental next-generation architecture should be compared as distinct systems rather than partially mixed.