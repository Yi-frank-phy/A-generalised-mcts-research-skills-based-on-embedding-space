# Canonical DTE Mathematics Restoration Implementation Plan

> **SUPERSEDED FOR GEOMETRY/CONTROLLER MATH (2026-08-11):** Use `docs/superpowers/specs/2026-08-11-p0-geometry-controller-recovery.md`. This file is retained only as historical implementation archaeology; its `H/log(N) -> T` law is not authoritative.


> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the canonical DTE controller chain `SD -> U=V+SD`, `H -> tau=H/log N -> T`, and `(U,T) -> Boltzmann allocation` without redesigning the current provisional KDE observables.

**Architecture:** Keep local uncertainty, global temperature, and resource allocation as separate deterministic layers. `math_engine.py` owns the local UCB primitive and zero-temperature Boltzmann limit; `entropy.py` owns the current-population diversity-to-temperature mapping plus independent plateau telemetry; runner/App driver only wire those values together. The current KDE observable remains semantically quarantined as a provisional batch-relative proxy.

**Tech Stack:** Python 3, NumPy, Pydantic models, pytest, GitHub Actions.

## Global Constraints

- Canonical UCB is exactly `U_i = V_i + SD_i`; temperature and `c_explore` do not scale it.
- `normalized_temperature = H/log(N)` for `N > 1`, else `0`.
- `effective_temperature = t_max * normalized_temperature`.
- `entropy_delta` remains plateau/continuation telemetry only.
- Boltzmann allocation uses UCB values, not raw Judge values.
- `T=0` is a real controller endpoint; implementation may use limiting logic internally but must not expose a positive temperature floor.
- Existing deterministic discretization and hard budget caps remain unchanged.
- The current KDE quantity remains a provisional batch-relative kernel surprisal proxy; this patch does not validate it as thermodynamic entropy.
- No unrelated agent, hook, history, method-vector, Judge, or stopping refactor.

---

### Task 1: Lock the canonical local UCB and zero-temperature Boltzmann behavior

**Files:**
- Modify: `tests/test_math_engine.py`
- Modify: `src/dte_backend/math_engine.py`

**Interfaces:**
- Consumes: `SearchNode.score`, `SearchNode.uncertainty`, existing deterministic `discretize_allocation()`.
- Produces: `calculate_ucb(score: float, uncertainty: float) -> float`; `allocate_frontier(...)` with legacy `tau`/`c_explore` arguments accepted only if needed for compatibility but mathematically ignored; `boltzmann_allocation(..., temperature=0.0)` implementing the exact zero-temperature limit.

- [ ] **Step 1: Write failing characterization tests**

Add tests equivalent to:

```python
def test_ucb_is_exact_value_plus_sd_and_ignores_temperature_controls():
    assert calculate_ucb(0.6, 0.25) == 0.85
    assert calculate_ucb(0.6, 0.25, tau=0.0, c_explore=0.0) == 0.85
    assert calculate_ucb(0.6, 0.25, tau=99.0, c_explore=99.0) == 0.85


def test_zero_temperature_boltzmann_concentrates_on_unique_maximum():
    assert boltzmann_allocation(
        [0.2, 0.9, 0.4],
        allocation_mass_per_iteration=3,
        max_children_per_iteration=5,
        node_ids=["a", "b", "c"],
        temperature=0.0,
    ) == [0, 3, 0]


def test_zero_temperature_tied_maxima_are_symmetric_before_cap_trimming():
    allocation = boltzmann_allocation(
        [0.9, 0.9, 0.2],
        allocation_mass_per_iteration=2,
        max_children_per_iteration=5,
        node_ids=["a", "b", "c"],
        temperature=0.0,
    )
    assert allocation == [1, 1, 0]


def test_higher_sd_changes_actual_ucb_allocation_support():
    nodes = [
        SearchNode(node_id="a", claim="A", score=0.5, uncertainty=0.1),
        SearchNode(node_id="b", claim="B", score=0.5, uncertainty=0.4),
    ]
    result = allocate_frontier(
        nodes,
        allocation_mass_per_iteration=4,
        max_children_per_iteration=5,
        temperature=0.25,
    )
    by_id = {item.node_id: item for item in result}
    assert by_id["b"].ucb_score > by_id["a"].ucb_score
    assert by_id["b"].expansion_budget >= by_id["a"].expansion_budget
```

- [ ] **Step 2: Run focused tests and verify RED**

Run through CI or an equivalent repository test runner:

```bash
pytest tests/test_math_engine.py -q
```

Expected before production changes: at least the UCB-independence test fails because current code multiplies uncertainty by `c_explore * tau`; zero-temperature semantics are not explicit.

- [ ] **Step 3: Implement the minimal local-math change**

Make `calculate_ucb()` return:

```python
return float(score + uncertainty)
```

Keep compatibility parameters only if removing them would force unrelated call-site churn; document that they are ignored/deprecated. In `allocate_frontier()`, compute each UCB from score and uncertainty only.

For `boltzmann_allocation()` add an explicit `temperature <= 0` branch:

```python
max_value = float(np.max(values))
winners = np.isclose(values, max_value, rtol=0.0, atol=1e-12)
probs = winners.astype(float) / float(np.sum(winners))
```

Then feed the resulting quotas through the existing `discretize_allocation()` so hard-cap behavior stays unchanged.

- [ ] **Step 4: Re-run focused tests and verify GREEN**

```bash
pytest tests/test_math_engine.py -q
```

Expected: all math-engine tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_math_engine.py src/dte_backend/math_engine.py
git commit -m "fix: restore canonical DTE UCB primitive"
```

---

### Task 2: Restore current-state entropy-to-temperature mapping and separate plateau telemetry

**Files:**
- Modify: `tests/test_entropy.py`
- Modify: `src/dte_backend/entropy.py`

**Interfaces:**
- Consumes: `spatial_entropy: float`, explicit `frontier_size: int`, `t_max: float`, previous entropy only for delta/plateau telemetry.
- Produces: `EntropyState.normalized_temperature = H/log(N)` and `effective_temperature = t_max * normalized_temperature` independent of previous entropy.

- [ ] **Step 1: Write failing tests**

Add tests equivalent to:

```python
import math


def test_temperature_comes_from_current_entropy_and_frontier_size():
    state = evaluate_entropy_state(
        spatial_entropy=0.5 * math.log(4),
        frontier_size=4,
        previous_entropy=None,
        iteration=1,
        min_iterations=2,
        entropy_change_threshold=0.05,
        t_max=2.0,
    )
    assert state.normalized_temperature == 0.5
    assert state.effective_temperature == 1.0


def test_temperature_endpoints():
    cold = evaluate_entropy_state(
        spatial_entropy=0.0,
        frontier_size=4,
        previous_entropy=0.9,
        iteration=2,
        min_iterations=2,
        entropy_change_threshold=0.05,
        t_max=2.0,
    )
    hot = evaluate_entropy_state(
        spatial_entropy=math.log(4),
        frontier_size=4,
        previous_entropy=0.9,
        iteration=2,
        min_iterations=2,
        entropy_change_threshold=0.05,
        t_max=2.0,
    )
    assert (cold.normalized_temperature, cold.effective_temperature) == (0.0, 0.0)
    assert (hot.normalized_temperature, hot.effective_temperature) == (1.0, 2.0)


def test_previous_entropy_changes_delta_not_temperature():
    kwargs = dict(
        spatial_entropy=0.5 * math.log(4),
        frontier_size=4,
        iteration=3,
        min_iterations=2,
        entropy_change_threshold=0.05,
        t_max=2.0,
    )
    first = evaluate_entropy_state(previous_entropy=0.1, **kwargs)
    second = evaluate_entropy_state(previous_entropy=0.69, **kwargs)
    assert first.normalized_temperature == second.normalized_temperature
    assert first.effective_temperature == second.effective_temperature
    assert first.entropy_delta != second.entropy_delta


def test_singleton_frontier_has_zero_controller_temperature():
    state = evaluate_entropy_state(
        spatial_entropy=0.0,
        frontier_size=1,
        previous_entropy=None,
        iteration=1,
        min_iterations=2,
        entropy_change_threshold=0.05,
        t_max=2.0,
    )
    assert state.normalized_temperature == 0.0
    assert state.effective_temperature == 0.0
```

- [ ] **Step 2: Run focused tests and verify RED**

```bash
pytest tests/test_entropy.py -q
```

Expected before production changes: signature/temperature tests fail because `frontier_size` is absent and current temperature is derived from entropy delta.

- [ ] **Step 3: Implement minimal entropy-controller change**

Change `evaluate_entropy_state()` to require `frontier_size`. Compute:

```python
if frontier_size <= 1:
    normalized_temperature = 0.0
else:
    max_entropy = math.log(frontier_size)
    normalized_temperature = spatial_entropy / max_entropy
    if normalized_temperature < -1e-12 or normalized_temperature > 1.0 + 1e-12:
        raise ValueError("spatial_entropy outside the current proxy's [0, log N] range")
    normalized_temperature = min(1.0, max(0.0, normalized_temperature))
effective_temperature = float(t_max * normalized_temperature)
```

Compute `entropy_delta`, plateau counts, and plateau signal separately. On the first iteration, delta remains `None` and plateau count remains zero, but temperature still comes from current entropy.

- [ ] **Step 4: Re-run focused tests and verify GREEN**

```bash
pytest tests/test_entropy.py -q
```

Expected: all entropy tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_entropy.py src/dte_backend/entropy.py
git commit -m "fix: restore entropy driven DTE temperature"
```

---

### Task 3: Wire runner and App driver to the canonical chain

**Files:**
- Modify: `src/dte_backend/runner.py`
- Modify: `src/dte_backend/app_driver.py`
- Modify: controller/integration tests discovered by search for `evaluate_entropy_state(`, `allocate_frontier(`, and `0.05` temperature floors.

**Interfaces:**
- Consumes: Task 1 UCB/allocator behavior and Task 2 `evaluate_entropy_state(..., frontier_size=len(frontier), ...)`.
- Produces: both controller paths use identical `H -> T` semantics and actual allocation from UCB, with no artificial temperature floor.

- [ ] **Step 1: Add failing integration tests**

Add the smallest existing-controller tests that prove:

```python
# Same current frontier/H, different previous H => same temperature and UCB/allocation inputs.
# A zero-current-entropy state reaches allocation with temperature == 0.0, not 0.05.
# Both runner and persistent App-driver records expose H/log(N) as normalized_temperature.
```

Prefer extending existing runner/App-driver tests rather than creating a new integration harness.

- [ ] **Step 2: Run the focused integration tests and verify RED**

Run only the touched controller test modules first. Expected failures: missing `frontier_size` argument and/or observed `0.05` floors.

- [ ] **Step 3: Update runner**

Change its entropy call to include:

```python
frontier_size=len(frontier)
```

Change allocation wiring from:

```python
tau=max(entropy_state.normalized_temperature, 0.05),
c_explore=1.0,
temperature=max(entropy_state.effective_temperature, 0.05),
```

to canonical wiring with no floor. If compatibility parameters remain in `allocate_frontier()`, do not use them as mathematical controls:

```python
temperature=entropy_state.effective_temperature
```

- [ ] **Step 4: Update App driver identically**

Find the deterministic controller transition that calls `evaluate_entropy_state()` and `allocate_frontier()`. Pass `frontier_size=len(frontier)` and remove `max(..., 0.05)` or equivalent floors. Persist the existing normalized temperature field with the new `H/log N` meaning.

- [ ] **Step 5: Re-run focused controller tests and verify GREEN**

Run the touched integration/controller modules. Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/dte_backend/runner.py src/dte_backend/app_driver.py tests/
git commit -m "fix: wire canonical DTE temperature allocation"
```

---

### Task 4: Align specification and compatibility surfaces

**Files:**
- Modify: `SPEC.md`
- Modify: `src/dte_backend/__main__.py` only if call signatures require it.
- Modify: other compatibility call sites found by exact search.

**Interfaces:**
- Consumes: implemented canonical functions.
- Produces: one authoritative mathematical description across docs and code.

- [ ] **Step 1: Search for stale formulas and call sites**

Search repository for exact concepts:

```text
c * tau * uncertainty
entropy_delta / threshold
normalized_temperature, 0.05
effective_temperature, 0.05
calculate_ucb(
evaluate_entropy_state(
allocate_frontier(
```

Classify each hit as production, test, historical/superseded spec, or telemetry compatibility.

- [ ] **Step 2: Update `SPEC.md`**

Document exactly:

```text
U_i = V_i + SD_i
normalized_temperature = H / log(N), N > 1; else 0
T = T_max * normalized_temperature
p_i = exp(U_i/T) / sum_j exp(U_j/T)
```

State explicitly that the current `H` input is the semantically quarantined batch-relative kernel surprisal proxy pending replacement, and that `delta H` is not temperature.

- [ ] **Step 3: Update necessary compatibility calls only**

If public/internal signatures still accept deprecated `tau`/`c_explore`, retain them only where backward compatibility materially reduces unrelated churn; mark them ignored/deprecated. Do not silently repurpose configuration fields.

- [ ] **Step 4: Run specification/static checks**

Use the repository's existing spec checker if configured, plus focused tests.

- [ ] **Step 5: Commit**

```bash
git add SPEC.md src/dte_backend/__main__.py src/dte_backend tests/
git commit -m "docs: align DTE controller mathematics"
```

---

### Task 5: Full regression, CI, and branch completion

**Files:**
- No new production scope unless a regression reveals a direct compatibility defect caused by Tasks 1-4.

**Interfaces:**
- Produces: verified branch ready to integrate.

- [ ] **Step 1: Run focused mathematical suite**

```bash
pytest tests/test_math_engine.py tests/test_entropy.py -q
```

Expected: PASS.

- [ ] **Step 2: Run full repository test suite**

```bash
pytest -q
```

Expected: all tests PASS.

- [ ] **Step 3: Run repository spec/static checks**

Run the existing repository check command(s), including `scripts/check_specs.py` if applicable.

- [ ] **Step 4: Inspect diff against the approved spec**

Verify no temperature multiplier remains in UCB; no entropy-delta temperature mapping remains; no public `0.05` controller floor remains; allocation still uses UCB; issue #26 semantic quarantine remains intact.

- [ ] **Step 5: Verify GitHub Actions on the branch/PR**

All required checks must pass before completion is claimed.

- [ ] **Step 6: Complete the development branch**

Use `superpowers:finishing-a-development-branch` and integrate according to the already-approved user workflow without introducing unrelated changes.
