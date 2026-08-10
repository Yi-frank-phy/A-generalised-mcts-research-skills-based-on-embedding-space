# Issue #26 design: quarantine batch-relative KDE semantics

## Goal

Fix the confirmed architectural defect without prematurely choosing the next research-state entropy theory.

The current self-including, per-batch median-bandwidth KDE quantity is useful as a legacy batch diagnostic, but it is not a cross-iteration thermodynamic entropy and must not be presented as proof of convergence.

## Approaches considered

### A. Semantic quarantine first — selected

Keep the legacy calculation for compatibility, but rename its semantics to `batch_relative_kernel_surprisal`, make its metric identity explicit, add exact regression counterexamples, and prevent documentation/telemetry from presenting it as authoritative convergence evidence.

Pros: minimal authority change; preserves compatibility; directly fixes the known false semantic claim; leaves future entropy/temperature theory open.

Cons: does not yet provide the replacement research-state metrics.

### B. Implement the report's three shadow metrics immediately

Add fixed-kernel conditional variance, weighted spectral entropy/effective rank, and weighted MMD in one change.

Pros: richer observability immediately.

Cons: risks hard-coding provisional choices before the recovered `U = V + SD` and entropy/temperature theory is re-derived; larger surface and more coupled tests.

### C. Replace the old metric with one new fixed-kernel entropy

Pros: superficially simple API.

Cons: collapses coverage, within-frontier diversity, and cross-window movement into one number again; repeats the architectural mistake identified by the review.

## Selected design

Implement Approach A as the first migration step.

### 1. Preserve numerical compatibility

`compute_kde_state()` keeps computing the current quantity so existing persisted runs and controller code do not silently change behavior in this issue.

The legacy numeric field may remain as a compatibility alias, but new code and docs must call the quantity `batch_relative_kernel_surprisal` rather than `spatial_entropy`.

### 2. Make metric identity explicit

Introduce a frozen/versioned identity for the legacy diagnostic containing at least:

- metric name/version;
- embedding normalization rule;
- kernel family;
- bandwidth rule;
- self-kernel inclusion policy.

The identity is telemetry/provenance, not a score.

Cross-window comparison is valid only when identities match exactly. A version mismatch starts a new sequence rather than silently computing a delta.

### 3. Add mathematical regression tests

Tests must encode the review's exact counterexamples:

- any two distinct normalized points under the per-batch median bandwidth rule produce the same batch-relative surprisal regardless of their absolute pair distance before normalization/scaling;
- an equidistant support produces the closed-form value and approaches 1/2 as support grows;
- these facts are documented as reasons the quantity cannot be interpreted as absolute search-space entropy.

These tests intentionally preserve the legacy behavior while preventing future code from accidentally claiming stronger semantics.

### 4. Entropy-state compatibility

Do not redesign global temperature in issue #26. `EntropyState` compatibility fields can remain temporarily, but documentation/docstrings must mark the input as the legacy batch-relative proxy. Any cross-iteration delta must be guarded by metric identity in new APIs.

Issue #28 owns authoritative stopping decoupling. Issue #27 owns the re-derivation of the uncertainty/SD estimator. Future research-state entropy/temperature work remains separate.

### 5. Error handling

- Reject malformed/non-finite embeddings as before or more strictly if existing validators allow it.
- A metric identity mismatch returns `delta=None` / sequence reset semantics in the new comparison helper; it must not fabricate a numerical delta.
- Degenerate empty/singleton batches remain explicitly specified compatibility cases.

## Testing strategy

1. Write failing tests for the counterexamples and identity mismatch behavior.
2. Add the minimum compatibility API/aliases needed to pass them.
3. Run focused KDE/entropy tests.
4. Run the full repository test suite before any completion claim.

## Non-goals

- no new method/history representation;
- no new UCB equation;
- no replacement SD estimator;
- no new thermodynamic entropy definition;
- no authority change to stopping/allocation beyond semantic quarantine;
- no conditional-variance/spectral-entropy/MMD promotion in this patch.

## Follow-up

After this lands, issue #28 should remove any remaining authoritative stop/readiness path that can be triggered by the legacy proxy. Issue #27 can then re-derive the local SD estimator without depending on batch min-max KDE semantics.