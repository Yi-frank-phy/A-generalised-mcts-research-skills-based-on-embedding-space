# Strict Boltzmann Entropy–Temperature Controller Design

## Goal

Restore the DTE temperature controller to a strict Boltzmann entropy–temperature correspondence while preserving the existing DTE search equations and current spatial-entropy observable.

This change is deliberately narrow. It does **not** redesign KDE, uncertainty, Judge value, UCB, allocation mass, stopping policy, Relation, or the search graph.

## Preserved equations

The UCB objective remains

\[
U_i = V_i + c\,\tau\,u_i.
\]

Boltzmann allocation remains

\[
p_i(T)=\frac{\exp(A_i/T)}{\sum_j \exp(A_j/T)},
\]

with the prototype default \(A_i=U_i\).

The current spatial entropy observable remains, for this change only,

\[
S_{\mathrm{spatial}}=-\frac1N\sum_i \log \rho_i,
\]

where \(\rho_i\) is the current KDE-derived density proxy. This design does not endorse that observable as the final research-state entropy; it only prevents the temperature controller from adding a second heuristic distortion.

## Strict Boltzmann correspondence

For a fixed non-degenerate allocation-value vector \(A=(A_1,\ldots,A_N)\), define

\[
p_i(T)=\frac{e^{A_i/T}}{Z(T)},\qquad
Z(T)=\sum_j e^{A_j/T},
\]

and

\[
S_B(T)=-\sum_i p_i(T)\log p_i(T).
\]

For \(T>0\),

\[
\frac{dS_B}{dT}=\frac{\operatorname{Var}_{p(T)}(A)}{T^3}\ge 0.
\]

If not all \(A_i\) are equal, the variance is positive and \(S_B(T)\) is strictly increasing for finite \(T>0\). Therefore the mapping between entropy and temperature is unique on the interior of the attainable entropy interval.

Let \(k_{\max}\) be the multiplicity of the maximum allocation value. Then

- \(T\to0^+\): probability concentrates uniformly over the \(k_{\max}\) maximizing values and \(S_B\to\log k_{\max}\);
- \(T\to\infty\): \(p_i\to1/N\) and \(S_B\to\log N\).

Hence the attainable Boltzmann entropy interval for fixed \(A\) is

\[
[\log k_{\max},\log N].
\]

The controller must obtain temperature from this strict relation, never from an empirical formula involving \(|\Delta S|\).

## Requested entropy and normalized UCB coordinate

For the present implementation, the requested entropy is the existing spatial entropy clipped only to its generic probability-entropy range:

\[
S_{\mathrm{requested}}=\operatorname{clip}(S_{\mathrm{spatial}},0,\log N).
\]

The bounded quantity used in the unchanged UCB equation is

\[
\tau=
\begin{cases}
S_{\mathrm{requested}}/\log N,&N>1,\\
0,&N\le1.
\end{cases}
\]

Thus \(\tau\in[0,1]\) remains a bounded monotone coordinate of the research-state entropy while the actual Boltzmann allocation uses the temperature implied by the fixed allocation values.

This preserves the scale expected by the existing UCB formula without inventing a second temperature schedule.

## Attainable entropy for the fixed allocation values

After \(\tau\) is known, compute the unchanged allocation values

\[
A_i=V_i+c\tau u_i.
\]

These fixed \(A_i\) determine the actual attainable Boltzmann interval. Define

\[
S_{\min}(A)=\log k_{\max},\qquad S_{\max}(A)=\log N,
\]

and project the requested entropy only when exact score symmetry makes it unattainable:

\[
S_{\mathrm{target}}
=
\operatorname{clip}
\left(
S_{\mathrm{requested}},
S_{\min}(A),
S_{\max}(A)
\right).
\]

This projection is not a tunable heuristic. It is the nearest entropy that the unchanged Boltzmann family can realize for the already-fixed \(A_i\).

For a unique maximum, \(S_{\min}=0\), so no lower-end projection is needed.

## Controller data flow

Each controller iteration becomes:

1. Compute the current frontier embeddings/KDE and \(S_{\mathrm{spatial}}\) exactly as today.
2. Compute \(S_{\mathrm{requested}}\) and \(\tau=S_{\mathrm{requested}}/\log N\).
3. Compute the unchanged UCB/allocation values \(A_i=V_i+c\tau u_i\).
4. Determine the attainable interval \([\log k_{\max},\log N]\) and project only if the requested entropy lies outside it.
5. Holding those \(A_i\) fixed, solve the unique Boltzmann temperature \(T\) satisfying
   \[
   S_B(T)=S_{\mathrm{target}}.
   \]
6. Allocate with the existing Boltzmann rule using that \(T\).
7. Compute \(\Delta S\) only for plateau/continuation telemetry and legacy stopping compatibility. \(\Delta S\) no longer determines \(\tau\) or \(T\).

The present hard floors

```text
max(normalized_temperature, 0.05)
max(effective_temperature, 0.05)
```

must be removed from both the ordinary runner and App driver, because they break the strict mapping at low entropy.

## Numerical inversion

Use deterministic bisection because monotonicity is guaranteed for non-degenerate \(A\).

Required behavior:

- If \(S_{\mathrm{target}}=\log k_{\max}\), return the zero-temperature endpoint \(T=0\); the resulting limiting distribution is uniform over the maximizing allocation values.
- If \(S_{\mathrm{target}}\approx\log N\), return \(T=+\infty\), which yields an exactly uniform soft allocation.
- Otherwise bracket by geometrically increasing the upper temperature until \(S_B(T_{\mathrm{hi}})\ge S_{\mathrm{target}}\), then bisect to a fixed deterministic tolerance.
- Use stable shifted logits when evaluating \(p_i(T)\).

Existing Boltzmann code may retain a tiny internal positive denominator only as a numerical implementation of the \(T=0\) limiting distribution; this numerical guard must not be exposed as the controller temperature.

No stochastic root finder and no learned calibration are permitted.

## Degenerate allocation values

If all \(A_i\) are equal, then \(k_{\max}=N\) and the attainable interval collapses to the single point \(\log N\). Temperature is not identifiable: the Boltzmann distribution is uniform for every \(T\).

The controller must not fabricate a unique root. It shall:

- allocate uniformly;
- expose the effective temperature as \(+\infty\) by convention for this iteration;
- leave the bounded UCB coordinate \(\tau\) determined by \(S_{\mathrm{requested}}\) exactly as above;
- record that the realized/target Boltzmann entropy is \(\log N\);
- preserve \(S_{\mathrm{requested}}\), the existing spatial entropy, and \(\Delta S\) telemetry;
- make the unattainable requested entropy observable if \(S_{\mathrm{requested}}<\log N\).

This is a true symmetry/identifiability limit, not an implementation error.

For \(N\le1\), allocation is trivial; use \(\tau=0\) and treat temperature as non-identifiable without changing the only available branch.

## Entropy delta and stopping

The existing relative entropy delta remains

\[
\Delta S_t=\frac{|S_t-S_{t-1}|}{\max(|S_{t-1}|,1)}.
\]

For this change, it retains only its present plateau/continuation role. The existing confirmation count and legacy `entropy_plateau` semantics are preserved.

The forbidden relationship after this change is

\[
T\propto\Delta S
\]

or any equivalent hand-tuned temperature schedule.

## State and telemetry compatibility

`EntropyState` keeps the current public concepts:

- `spatial_entropy`
- `entropy_delta`
- `effective_temperature`
- `normalized_temperature`
- plateau fields

but their meanings become:

- `effective_temperature`: the strict Boltzmann temperature \(T\), including \(0\) and \(+\infty\) endpoint semantics;
- `normalized_temperature`: the bounded entropy coordinate \(\tau=S_{\mathrm{requested}}/\log N\), not an entropy-delta heuristic.

Implementation may add explicit requested/realized entropy telemetry if needed to make symmetry projection auditable, but it must not introduce another controller schedule or reward.

Persistent App controller records currently store `normalized_temperature` but not `effective_temperature`; no state-schema migration is required solely for this change unless requested/realized entropy is persisted. Existing historical records remain historical facts produced by the old controller and must not be silently recomputed.

## Files expected to change during implementation

Architecture-level behavior is documented before code, per `AGENTS.md`.

Expected implementation scope:

- `SPEC.md`: replace the heuristic temperature description with the strict correspondence and clarify `normalized_temperature` semantics.
- `src/dte_backend/entropy.py`: entropy normalization, Boltzmann entropy evaluation, strict inversion, endpoint handling, plateau bookkeeping.
- `src/dte_backend/math_engine.py`: expose/reuse deterministic Boltzmann probability/entropy logic as needed without changing the allocation equation.
- `src/dte_backend/runner.py`: compute the strict temperature from current allocation values and remove the `0.05` floors.
- `src/dte_backend/app_driver.py`: same deterministic controller path and removal of the `0.05` floors.
- `tests/test_entropy.py`: mathematical monotonicity, inversion, tied maxima, endpoints, degeneracy, and plateau separation.
- `tests/test_math_engine.py` and controller tests as needed: confirm unchanged UCB/Boltzmann equations and strict-temperature integration.

No unrelated refactor is in scope.

## Test requirements

Implementation must be test-driven and must lock down at least these properties:

1. For fixed non-equal \(A\), \(S_B(T)\) increases with \(T\).
2. Round-trip: for several finite \(T\), compute \(S_B(T)\), invert it, and recover \(T\) within tolerance.
3. With a unique maximizing \(A_i\), requested entropy 0 gives the zero-temperature endpoint.
4. With \(k>1\) tied maxima, the zero-temperature entropy is \(\log k\), and any lower requested entropy projects to this attainable endpoint rather than producing a fake root.
5. Requested entropy \(\log N\) gives uniform allocation / infinite-temperature endpoint.
6. Equal allocation values are recognized as non-identifiable and allocate uniformly rather than inventing a root.
7. Changing only `previous_entropy` changes plateau telemetry but does not change the current \(\tau\), fixed \(A_i\), or strict \(T\).
8. Runner and App driver no longer impose an artificial `0.05` minimum.
9. Existing UCB equation and Boltzmann allocation equation remain unchanged.
10. Existing hard child/node budgets and deterministic discretization remain unchanged.

## Non-goals

This change does not decide whether KDE node-text entropy is the correct research entropy. It deliberately leaves that research question for the subsequent DTE work on method–understanding transitions, interpretation divergence, and annealing signals.

It also does not add a new verifier, reward, history mechanism, method vector, or convergence metric.
