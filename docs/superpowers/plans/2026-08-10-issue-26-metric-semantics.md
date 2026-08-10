# Issue #26 Metric Semantic Quarantine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the legacy KDE calculation while preventing it from being treated as a cross-iteration research entropy, with versioned provenance and regression tests for the known mathematical counterexamples.

**Architecture:** Keep `compute_kde_state()` numerically compatible. Add an explicit immutable identity and a new semantic alias `batch_relative_kernel_surprisal`; legacy `spatial_entropy` remains compatibility-only. Add a small comparison object/helper that refuses cross-version deltas. Do not redesign UCB, temperature, allocation, or stopping in this patch.

**Tech Stack:** Python 3, dataclasses, NumPy, pytest.

## Global Constraints

- Preserve current controller behavior in issue #26.
- Do not introduce a replacement research-state entropy.
- Do not modify the theoretical `U = V + SD` line.
- Do not promote conditional variance, spectral entropy, or MMD to controller authority.
- Metric identity mismatches must never silently produce a cross-window delta.

---

### Task 1: Pin the legacy KDE counterexamples in tests

**Files:**
- Modify: `tests/test_kde.py`

**Interfaces:**
- Consumes: `compute_kde_state(embeddings: list[list[float]]) -> KDEState`
- Produces: regression coverage for the known two-point and equidistant-support behavior.

- [ ] **Step 1: Write failing semantic regression tests**

Add imports for `math` and `numpy` if needed, then add tests equivalent to:

```python
import math

import numpy as np


def test_two_distinct_points_have_batch_relative_closed_form():
    expected = -math.log((1.0 + math.exp(-0.5)) / 2.0)
    near = compute_kde_state([[1.0, 0.0], [0.99, 0.1]])
    far = compute_kde_state([[1.0, 0.0], [-1.0, 0.0]])
    assert near.spatial_entropy == pytest.approx(expected)
    assert far.spatial_entropy == pytest.approx(expected)


def test_equidistant_support_matches_closed_form_and_tends_to_half():
    # Use a regular simplex Gram construction so normalized pairwise distances are equal.
    for n in (3, 4, 8):
        gram = np.full((n, n), -1.0 / (n - 1))
        np.fill_diagonal(gram, 1.0)
        vals, vecs = np.linalg.eigh(gram)
        coords = vecs[:, vals > 1e-10] @ np.diag(np.sqrt(vals[vals > 1e-10]))
        state = compute_kde_state(coords.tolist())
        expected = -math.log((1.0 + (n - 1) * math.exp(-0.5)) / n)
        assert state.spatial_entropy == pytest.approx(expected)
    assert expected < 0.5
```

Also import `pytest` explicitly.

- [ ] **Step 2: Run the focused tests**

Run:

```bash
pytest tests/test_kde.py -v
```

Expected before implementation: existing numeric tests pass; any test referencing the new semantic alias/identity in Task 2 is not added yet.

- [ ] **Step 3: Commit the regression-only tests**

```bash
git add tests/test_kde.py
git commit -m "test: pin legacy KDE surprisal counterexamples"
```

---

### Task 2: Add explicit legacy metric identity and semantic alias

**Files:**
- Modify: `src/dte_backend/kde.py`
- Modify: `tests/test_kde.py`

**Interfaces:**
- Produces: `KDEMetricIdentity`, `LEGACY_KDE_METRIC_IDENTITY`, `KDEState.batch_relative_kernel_surprisal`.
- Compatibility: `KDEState.spatial_entropy` remains available.

- [ ] **Step 1: Write failing identity/alias tests**

Add:

```python
from dte_backend.kde import LEGACY_KDE_METRIC_IDENTITY


def test_legacy_kde_metric_identity_is_explicit_and_versioned():
    identity = LEGACY_KDE_METRIC_IDENTITY
    assert identity.name == "batch_relative_kernel_surprisal"
    assert identity.version == 1
    assert identity.embedding_normalization == "l2_per_vector"
    assert identity.kernel == "gaussian"
    assert identity.bandwidth_rule == "median_nonzero_pairwise_squared_distance_per_batch"
    assert identity.self_kernel_included is True


def test_batch_relative_kernel_surprisal_alias_matches_legacy_field():
    state = compute_kde_state([[1, 0], [0, 1], [-1, 0]])
    assert state.batch_relative_kernel_surprisal == state.spatial_entropy
```

- [ ] **Step 2: Verify the new tests fail**

Run:

```bash
pytest tests/test_kde.py::test_legacy_kde_metric_identity_is_explicit_and_versioned tests/test_kde.py::test_batch_relative_kernel_surprisal_alias_matches_legacy_field -v
```

Expected: FAIL because the identity/property do not exist.

- [ ] **Step 3: Implement the minimum compatibility API**

In `src/dte_backend/kde.py`, add:

```python
@dataclass(frozen=True)
class KDEMetricIdentity:
    name: str
    version: int
    embedding_normalization: str
    kernel: str
    bandwidth_rule: str
    self_kernel_included: bool


LEGACY_KDE_METRIC_IDENTITY = KDEMetricIdentity(
    name="batch_relative_kernel_surprisal",
    version=1,
    embedding_normalization="l2_per_vector",
    kernel="gaussian",
    bandwidth_rule="median_nonzero_pairwise_squared_distance_per_batch",
    self_kernel_included=True,
)
```

Add to `KDEState`:

```python
@property
def batch_relative_kernel_surprisal(self) -> float:
    """Compatibility-safe name for the legacy batch-relative KDE diagnostic."""
    return self.spatial_entropy
```

Update module/class/function docstrings so `spatial_entropy` is described as a legacy compatibility field and never as physical/research entropy.

- [ ] **Step 4: Run focused tests**

```bash
pytest tests/test_kde.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dte_backend/kde.py tests/test_kde.py
git commit -m "refactor: name legacy KDE metric explicitly"
```

---

### Task 3: Add version-safe comparison semantics

**Files:**
- Modify: `src/dte_backend/entropy.py`
- Modify: `tests/test_entropy.py`

**Interfaces:**
- Consumes: `KDEMetricIdentity`, `LEGACY_KDE_METRIC_IDENTITY`.
- Produces: `MetricObservation`, `relative_metric_delta(current, previous) -> float | None`.

- [ ] **Step 1: Write failing comparison tests**

Add tests equivalent to:

```python
from dataclasses import replace

from dte_backend.entropy import MetricObservation, relative_metric_delta
from dte_backend.kde import LEGACY_KDE_METRIC_IDENTITY


def test_metric_delta_requires_identical_metric_identity():
    current = MetricObservation(1.1, LEGACY_KDE_METRIC_IDENTITY)
    previous = MetricObservation(1.0, LEGACY_KDE_METRIC_IDENTITY)
    assert relative_metric_delta(current, previous) == pytest.approx(0.1)

    incompatible = MetricObservation(
        1.0,
        replace(LEGACY_KDE_METRIC_IDENTITY, version=2),
    )
    assert relative_metric_delta(current, incompatible) is None
```

Import `pytest` explicitly in `tests/test_entropy.py`.

- [ ] **Step 2: Verify failure**

```bash
pytest tests/test_entropy.py::test_metric_delta_requires_identical_metric_identity -v
```

Expected: FAIL because the new types/helpers do not exist.

- [ ] **Step 3: Implement the helper**

In `src/dte_backend/entropy.py`:

```python
from .kde import KDEMetricIdentity, LEGACY_KDE_METRIC_IDENTITY, compute_kde_state


@dataclass(frozen=True)
class MetricObservation:
    value: float
    identity: KDEMetricIdentity = LEGACY_KDE_METRIC_IDENTITY


def relative_metric_delta(
    current: MetricObservation,
    previous: MetricObservation | None,
) -> float | None:
    if previous is None or current.identity != previous.identity:
        return None
    return abs(current.value - previous.value) / max(abs(previous.value), 1.0)
```

Do not route the existing controller through this helper yet; issue #26 only establishes safe semantics and provenance without an authority change.

- [ ] **Step 4: Run entropy tests**

```bash
pytest tests/test_entropy.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dte_backend/entropy.py tests/test_entropy.py
git commit -m "feat: version KDE metric comparisons"
```

---

### Task 4: Align public documentation with the compatibility boundary

**Files:**
- Modify: `SPEC.md`
- Modify: `ARCHITECTURE.md`

**Interfaces:**
- No runtime API changes.

- [ ] **Step 1: Locate statements that call the legacy KDE quantity spatial/research entropy or convergence evidence**

Run:

```bash
rg -n "spatial entropy|spatial_entropy|entropy plateau|KDE.*entropy|convergence" SPEC.md ARCHITECTURE.md
```

- [ ] **Step 2: Replace only the semantic claims affected by #26**

Required wording principles:

- call the current quantity `batch-relative kernel surprisal proxy`;
- state that its batch-relative scale is not comparable across arbitrary metric versions/bandwidth rules;
- state that it is not proof of research convergence;
- preserve current behavior descriptions as compatibility behavior;
- point authoritative stopping changes to issue #28 rather than specifying a new entropy formula here.

- [ ] **Step 3: Run documentation-sensitive tests plus focused unit tests**

```bash
pytest tests/test_kde.py tests/test_entropy.py -v
```

- [ ] **Step 4: Commit**

```bash
git add SPEC.md ARCHITECTURE.md
git commit -m "docs: quarantine legacy KDE entropy semantics"
```

---

### Task 5: Full verification

**Files:**
- No new files.

- [ ] **Step 1: Run the full test suite**

```bash
pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Inspect diff for accidental controller-authority changes**

```bash
git diff main...HEAD -- src/dte_backend tests SPEC.md ARCHITECTURE.md
```

Confirm no allocation/stopping/temperature behavior was changed.

- [ ] **Step 3: Record issue progress**

Comment on #26 with the branch/commit summary and test result. Keep #26 open unless all acceptance criteria are met; replacement metrics remain follow-up work.
