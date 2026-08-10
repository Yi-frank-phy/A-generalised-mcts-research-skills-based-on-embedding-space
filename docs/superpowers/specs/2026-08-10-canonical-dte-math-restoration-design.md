# Canonical DTE mathematics restoration design

## Status

Approved design for restoring the DTE controller's mathematical layer before further research on method→understanding observables.

This document supersedes `docs/superpowers/specs/2026-08-10-strict-boltzmann-temperature-design.md` as the implementation authority for the current restoration work.

The purpose is not to reproduce one historical commit verbatim. Repository archaeology shows that DTE's formulas drifted across several generations. This design recovers the simplest mathematical separation supported by the earliest design intent and by the later statistical derivation of the UCB uncertainty term.

## 1. Archaeological result

Three distinct controller structures existed historically.

### 1.1 Early design: local branch score and global annealing were separate

At commit `208db3daaf4616bf82c77f74c944d4a65b88724e`, the design described a local branch score of the form

\[
\mathrm{FinalScore}=V+C\,\mathrm{ExplorationBonus},
\]

while temperature belonged to a separate simulated-annealing/Metropolis layer. The same design used a normalized population entropy to set temperature:

\[
T=T_{\max}\frac{H}{H_{\max}}.
\]

The important structural fact is separation: temperature was a population-level control variable, not part of the local uncertainty estimate.

### 1.2 First drift: temperature was inserted into UCB

Commit `098c597a4a5f6edd3d7cf3fa21a467f2d2d04149` explicitly changed the controller description so that temperature scaled the UCB exploration bonus:

\[
\mathrm{UCB}=V+T\,\mathrm{ExplorationBonus}.
\]

This is therefore a later design choice, not an unavoidable consequence of the original UCB/statistical argument.

### 1.3 Later statistical derivation: KDE estimated standard-error-like uncertainty

The legacy proof in `deep-think-evolving/docs/math_proof_ucb.md` starts from

\[
\operatorname{Var}(\bar X_n)=\frac{\sigma^2}{n},
\qquad
\operatorname{SE}(\bar X_n)=\frac{\sigma}{\sqrt n},
\]

and extends the effective sample count to continuous strategy space with kernel regression/KDE, yielding the local relationship

\[
\mathrm{SD}(x)\propto\frac{1}{\sqrt{\hat f(x)}}.
\]

This establishes the conceptual role of KDE in the local UCB layer: it is an estimator of standard-deviation/standard-error-like uncertainty, not temperature itself.

### 1.4 Second drift: inverse-canonical state probe plus split allocation semantics

By the later legacy implementation, temperature was estimated independently through

\[
\ln p(v)\approx\frac{V}{T}+C,
\qquad
T_{\mathrm{eff}}
=
\left|\frac{\operatorname{Var}(V)}{\operatorname{Cov}(V,\ln p)}\right|.
\]

At the same time, the implementation computed a temperature-scaled UCB mainly for ranking/display while Boltzmann child allocation used raw Judge value `V`. This left the local uncertainty layer and actual resource-allocation layer only partially coupled.

### 1.5 Current drift

The current backend uses both

\[
U_i=V_i+c\tau u_i
\]

and a heuristic temperature generated from entropy change,

\[
\tau\sim |\Delta H|,
\]

then normally allocates using UCB values. This combines two later modifications and no longer matches the clean early separation.

## 2. Canonical mathematical layers

The restored controller has three independent mathematical layers.

### 2.1 Local uncertainty / UCB

For each live frontier branch `i`, define

\[
\boxed{U_i=V_i+\mathrm{SD}_i}.
\]

Interpretation:

- `V_i` is the Judge's current estimated branch value;
- `SD_i` is the current standard-deviation/standard-error-like uncertainty estimate for that branch.

There is no global temperature factor in this formula.

There is no independent `c_explore` multiplier in the canonical formula.

For this restoration patch, the existing KDE-derived `uncertainty` field remains the provisional estimator supplied as `SD_i`. This patch does **not** claim that the present raw semantic KDE geometry is the final correct estimator of epistemic uncertainty. Issue #27 / subsequent method→understanding research owns that estimator question.

Therefore this patch restores the controller structure without prematurely redesigning the SD observable.

### 2.2 Global diversity state / temperature

Let `H_t` be the current population-level diversity observable used by the controller, and let `N_t` be the number of live frontier states from the same batch. `N_t` is an explicit controller input; it must not be inferred from the entropy value or reconstructed through a fitted normalization.

The canonical normalized temperature coordinate is

\[
\boxed{
\tau_t=
\begin{cases}
H_t/\log N_t,&N_t>1,\\
0,&N_t\le1,
\end{cases}}
\]

and the effective system temperature is

\[
\boxed{T_t=T_{\max}\tau_t}.
\]

The current backend's self-including kernel quantity satisfies

\[
0\le H_t\le\log N_t
\]

because each KDE-like mean kernel density contains the self-kernel contribution and hence lies in `[1/N,1]`.

For this compatibility input, finite values outside that analytically guaranteed range indicate a metric/implementation defect and must not be silently repaired with a new controller schedule. Normal floating-point tolerance may be handled explicitly in tests/validation.

However, issue #26 has already established that this quantity is only a batch-relative kernel surprisal proxy and must not be presented as an absolute thermodynamic entropy or as proof of convergence. Therefore:

- the structural mapping `population diversity observable -> normalized temperature -> T` is restored now;
- the current legacy proxy may feed that mapping temporarily for compatibility;
- code/docs must retain the semantic quarantine and must not claim that the proxy is the final research-state entropy;
- replacing the global observable with method→understanding / interpretation-state structure remains a later research task.

No clipping schedule, entropy-delta scaling, learned calibration, or manual annealing curve is added.

### 2.3 Resource allocation

After local branch UCB values and global temperature are determined, allocate search mass by

\[
\boxed{
p_i(T)=
\frac{\exp(U_i/T)}{\sum_j\exp(U_j/T)}
}
\]

with the existing deterministic discretization and hard child/node budgets applied afterward.

The allocation energy/attractiveness variable is `U_i`, not raw `V_i`.

Reason: the local UCB uncertainty term must affect actual search-budget allocation rather than remain display-only. Using raw `V_i` here would reproduce the legacy split where the UCB exploration calculation did not control the branch expansion it was intended to guide.

Temperature controls how concentrated the budget is across the already-computed UCB values:

- low `T`: concentrate on the highest-`U` branches;
- high `T`: flatten allocation across branches.

This preserves the conceptual division:

\[
\boxed{
\begin{aligned}
V_i+\mathrm{SD}_i &\longrightarrow U_i &&\text{local branch uncertainty},\\
\text{global diversity state} &\longrightarrow T &&\text{global annealing state},\\
(U_i,T) &\longrightarrow p_i &&\text{resource allocation}.
\end{aligned}}
\]

## 3. Entropy delta and convergence telemetry

The existing cross-iteration change metric may remain for compatibility:

\[
\Delta H_t=
\frac{|H_t-H_{t-1}|}{\max(|H_{t-1}|,1)}.
\]

Its role is restricted to plateau/continuation telemetry and any separately governed convergence logic.

It must **not** enter either

\[
U_i
\]

or

\[
T_t.
\]

In particular, the following current relationship is removed:

\[
T\propto |\Delta H|.
\]

Issue #26/#28 semantic protections remain authoritative: the legacy batch-relative proxy and its delta are not sufficient evidence of epistemic convergence.

## 4. Boundary behaviour

### 4.1 Empty and singleton frontier

For `N=0`, allocation is empty.

For `N=1`:

\[
\tau=0,\qquad T=0,
\]

and the only live branch receives all available allocation mass subject to existing hard budgets.

### 4.2 Zero diversity proxy

For `N>1` and `H=0`:

\[
T=0.
\]

Boltzmann allocation is interpreted in the zero-temperature limit: allocation mass concentrates on the branch or tied branches with maximal `U`.

Implementation may use a tiny positive denominator internally for numerical stability, but the controller/telemetry temperature remains exactly zero.

### 4.3 Maximum proxy value

When

\[
H=\log N,
\]

then

\[
\tau=1,\qquad T=T_{\max}.
\]

This is the restored early normalized-temperature rule. It does **not** assert that `T_max` is an infinite-temperature thermodynamic endpoint.

### 4.4 Ties

Exact ties in maximal `U` at zero temperature are treated symmetrically. No stable-ID preference may be introduced before the existing deterministic discretization/hard-cap stage requires a tie-break.

## 5. Public semantic changes

### 5.1 `normalized_temperature`

After this patch:

\[
\texttt{normalized_temperature}=H/\log N
\]

for `N>1`, and `0` for `N<=1`.

It no longer means `entropy_delta / threshold`.

### 5.2 `effective_temperature`

After this patch:

\[
\texttt{effective_temperature}=T_{\max}\times\texttt{normalized_temperature}.
\]

It no longer means a heuristic function of entropy change.

### 5.3 UCB score

After this patch:

\[
\texttt{ucb_score}=\texttt{score}+\texttt{uncertainty}.
\]

`tau` and `c_explore` must not alter the value.

Existing configuration fields may be retained temporarily for backward-compatible parsing if other interfaces still send them, but they must not affect the canonical UCB calculation. Their deprecation should be explicit rather than silently repurposed.

## 6. Implementation scope

Expected files:

- `SPEC.md`
  - replace the temperature-scaled UCB equation with `U = V + SD`;
  - separate local uncertainty, global temperature, and Boltzmann allocation;
  - document the legacy diversity proxy as provisional/quarantined.

- `src/dte_backend/math_engine.py`
  - make `calculate_ucb()` implement `score + uncertainty` only;
  - remove temperature/exploration-coefficient dependence from UCB computation;
  - keep Boltzmann allocation driven by UCB by default;
  - preserve deterministic discretization and hard caps;
  - explicitly handle the `T=0` limiting allocation without exposing an artificial controller floor.

- `src/dte_backend/entropy.py`
  - accept the current live `frontier_size` explicitly alongside the diversity observable;
  - derive `normalized_temperature` from current `H/log N`, not entropy delta;
  - derive `effective_temperature = t_max * normalized_temperature`;
  - leave entropy delta and plateau bookkeeping separate;
  - retain issue #26's semantic quarantine of the legacy input metric.

- `src/dte_backend/runner.py`
  - pass the current live frontier size into the entropy/temperature controller;
  - stop passing temperature into UCB;
  - use the restored effective temperature for allocation;
  - remove any artificial minimum temperature that changes controller semantics.

- `src/dte_backend/app_driver.py`
  - mirror runner semantics exactly, including explicit frontier size;
  - remove the same artificial floors/temperature-in-UCB path.

- `src/dte_backend/__main__.py` and compatibility call sites
  - adapt API calls if `tau`/`c_explore` are removed from the canonical calculation signature.

- tests
  - add mathematical characterization tests before production changes;
  - update integration/controller tests only after red tests demonstrate the existing drift.

No unrelated agent/orchestration/hook/history/method-vector refactor is in scope.

## 7. TDD acceptance properties

Implementation must be driven by failing tests that lock down at least the following.

### 7.1 UCB independence from global temperature

For fixed `V` and `SD`, changing `H`, `tau`, `T`, previous entropy, or entropy threshold cannot change `U`:

\[
U=V+SD.
\]

### 7.2 Exact local primitive

Representative values satisfy exact arithmetic, e.g.

\[
V=0.6,\quad SD=0.25
\quad\Rightarrow\quad
U=0.85.
\]

### 7.3 Temperature from current population state only

For `N>1`:

\[
\tau=H/\log N,
\qquad
T=T_{\max}\tau.
\]

Changing only `previous_entropy` may change plateau telemetry but cannot change current `tau` or `T`.

### 7.4 Endpoints

- `H=0 -> tau=0 -> T=0`;
- `H=log N -> tau=1 -> T=T_max`;
- `N<=1 -> tau=0 -> T=0`.

### 7.5 Zero-temperature allocation

At `T=0`, allocation concentrates on maximal UCB branch(es) without relying on a public nonzero floor.

### 7.6 UCB actually drives allocation

Construct two frontiers with equal Judge values but different uncertainties. The branch with larger `SD` must obtain the larger UCB and, at finite nonzero temperature, weakly greater Boltzmann probability/allocation support.

This guards against regression to the legacy `Boltzmann(V/T)` split.

### 7.7 Delta separation

Two states with the same current `H` and frontier but different previous `H` must have identical UCB values and temperature. Only delta/plateau telemetry may differ.

### 7.8 Frontier-size dependency is explicit

Two controller evaluations with the same scalar `H` but different valid frontier sizes may have different normalized temperatures because the normalization denominator is `log N`. Tests must pass `N` explicitly and must reject/inhibit any implementation that guesses `N` from `H`.

### 7.9 Budget invariants

Existing allocation mass semantics, deterministic piecewise discretization, stable tie handling, and hard per-iteration/node caps remain unchanged.

### 7.10 Metric-semantic compatibility

Issue #26's counterexamples and metric-identity protections must continue to pass. This restoration must not relabel the current batch-relative kernel surprisal as validated thermodynamic entropy.

### 7.11 Full regression

All focused mathematical/controller tests and the full repository test suite must pass before completion is claimed.

## 8. Non-goals

This patch does not solve the new research question of how to estimate `SD` from method→understanding transitions, interpretation instability, explanatory compression, or structured unexplained residuals.

It does not define the final research-state entropy.

It does not add history retrieval, method vectors, new Judge rewards, new stopping authority, or model training.

It does not implement the previously proposed strict inversion

\[
S_B(T)\leftrightarrow T
\]

because repository archaeology showed that this would be a new controller design rather than a faithful restoration of the earlier DTE structure.

## 9. Success criterion

After the patch, the current DTE backend must express one unambiguous controller chain:

\[
\boxed{
\mathrm{SD}_i\;\xrightarrow{\;V_i+\mathrm{SD}_i\;}\;U_i,
\qquad
H_t\;\xrightarrow{\;/\log N\;}\;\tau_t\;\xrightarrow{\;T_{\max}\;}\;T_t,
\qquad
(U_i,T_t)\;\xrightarrow{\;\mathrm{Boltzmann}\;}\;p_i.
}
\]

No temperature multiplier remains in UCB, and no entropy-change heuristic remains in temperature.