# Compression–Necessity Grant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an experimentally measured compression–necessity gate that can collapse a redundant active research cluster into a shorter macro-state without information loss and, when the compact representation is also empirically necessary, grant that method one full controller-round node budget without a generic Judge episode.

**Architecture:** Keep ordinary DTE search value (`V`, `SD`, `U = V + SD`) unchanged. A high-weight/high-allocation method may trigger a separate bounded representation experiment: reconstruct a frozen cluster from a compact method-conditioned context, score recoverability against hidden structured probes, compare representation length, then repeat the same reconstruction with the method withheld. Passing compression, fidelity, and necessity thresholds creates a durable `CompressionNecessityGrant`; the original nodes remain immutable provenance but leave the active representation in favor of a compact macro-node, and the granted method receives one full round of expansion budget. This is an event-triggered budget grant, not a scalar reward term.

**Tech Stack:** Python 3, Pydantic models, existing DTE `EpisodeRequest -> EpisodeResult` protocol, pytest, GitHub Actions.

**Spec:** `docs/PHYSICS.md`, `docs/DESIGN.md`, `SPEC.md`; design rationale and experiment notes are tracked in repository issue `#138`.

## Global Constraints

- Do not reintroduce Judge score as controller value or as a gate input.
- Do not add compression or necessity as additive terms to `U = V + SD`.
- Compression and necessity must be measured from reconstruction/ablation behavior, not from an LLM self-rating such as “compression score 0–10”.
- The original committed nodes and epistemic ledger records are immutable provenance and are never deleted from history.
- Only the active/search representation may collapse to a compact macro-node.
- A grant is single-use and non-renewable for the triggering compact representation; no recursive self-granting loop is allowed.
- A full-round grant cannot exceed the run's remaining hard committed-node budget.
- The reconstruction arm and method-withheld ablation arm use the same model/runtime limits and the same hidden probe set.
- Probe answers must be hidden from the reconstruction model. Probe generation and scoring must be reproducible from frozen pre-test state.
- If fidelity cannot be established, the cluster is not compacted and no grant is issued.
- This feature changes controller allocation policy and therefore must update `docs/PHYSICS.md`, `docs/DESIGN.md`, `SPEC.md`, relevant theory-lock digests, and tests together.

---

## File Structure

**Create**
- `src/dte_backend/compression_models.py` — typed probe, reconstruction, ablation, and grant records.
- `src/dte_backend/compression_test.py` — deterministic cluster freezing, probe construction/scoring, length accounting, and pass/fail decision.
- `tests/test_compression_test.py` — unit tests for fidelity, compression, necessity, and single-use grant semantics.
- `tests/test_app_compression_grant.py` — App-driver integration tests for trigger → bounded episode(s) → compact state → full-round grant.

**Modify**
- `src/dte_backend/episode_models.py` — add a dedicated bounded `reconstruction` episode contract; it returns reconstruction content only and cannot mutate controller metrics.
- `src/dte_backend/episode_adapter.py` — build blinded method-present and method-withheld reconstruction requests from the same frozen test state.
- `src/dte_backend/app_driver.py` — schedule eligible compression tests, persist results, replace only active representation after a pass, and apply one full-round grant subject to hard remaining budget.
- `src/dte_backend/models.py` — add compact macro-node metadata that references absorbed historical node IDs without deleting them.
- `src/dte_backend/observability_models.py` and `src/dte_backend/observability.py` — report attempted/passed/failed compression tests and grants.
- `docs/PHYSICS.md` — define the compression–necessity grant as a separate allocation event, not a UCB term.
- `docs/DESIGN.md` — define active-representation compaction vs immutable history.
- `SPEC.md` — define persistence/replay requirements and one-shot budget authority.
- theory-lock tests/digests that pin `docs/PHYSICS.md` and `docs/DESIGN.md`.

---

### Task 1: Define the measured compression/necessity contract

**Files:**
- Create: `src/dte_backend/compression_models.py`
- Test: `tests/test_compression_test.py`

**Interfaces:**
- Produces `CompressionProbe`, `CompressionSnapshot`, `ReconstructionResult`, `CompressionNecessityDecision`, and `CompressionNecessityGrant`.
- `CompressionSnapshot` freezes the candidate method ID, absorbed node IDs, atomic epistemic record IDs, original representation length, probe set identity, and runtime/model identity before either reconstruction arm runs.
- `CompressionNecessityDecision` contains measured quantities only: fidelity, compact length, baseline length, compression ratio, method-withheld fidelity/length, necessity gap, and exact threshold outcomes.

- [ ] **Step 1: Write failing model-validation tests**

```python
def test_compression_snapshot_requires_frozen_nonempty_cluster():
    with pytest.raises(ValueError):
        CompressionSnapshot(
            method_node_id="m1",
            absorbed_node_ids=[],
            epistemic_record_ids=[],
            original_token_length=0,
            probe_set_hash="0" * 64,
            model_identity="test-model",
        )


def test_grant_is_single_use_and_bound_to_snapshot():
    grant = CompressionNecessityGrant(
        grant_id="g1",
        snapshot_hash="a" * 64,
        method_node_id="m1",
        full_round_budget=3,
        consumed=False,
    )
    assert grant.consumed is False
```

- [ ] **Step 2: Run the tests and confirm RED**

Run:

```bash
pytest tests/test_compression_test.py -q
```

Expected: import/model failures because `compression_models.py` and the types do not yet exist.

- [ ] **Step 3: Implement only the typed records and validators**

Use strict `DTEBaseModel` subclasses. Hash fields use the existing 64-hex pattern. Require at least two absorbed nodes for a cluster compaction event, positive original length, nonempty probe set, and `full_round_budget >= 1`.

- [ ] **Step 4: Re-run the focused tests and confirm GREEN**

```bash
pytest tests/test_compression_test.py -q
```

- [ ] **Step 5: Commit**

```bash
git add src/dte_backend/compression_models.py tests/test_compression_test.py
git commit -m "feat: define compression necessity grant records"
```

---

### Task 2: Freeze a cluster and build hidden reconstruction probes

**Files:**
- Create: `src/dte_backend/compression_test.py`
- Modify: `tests/test_compression_test.py`

**Interfaces:**
- Consumes committed `SearchNode` objects plus the durable epistemic ledger.
- Produces `CompressionSnapshot` and a probe bundle whose expected answers are stored only in backend state, never in the reconstruction request.
- Probe units correspond to durable atomic information already present before the test: distinct claims, assumptions, counterexamples/challenges, evidence dependencies, unresolved dependencies, and path dispositions.

- [ ] **Step 1: Write a failing test that proves probes are hidden from the model-facing payload**

```python
def test_reconstruction_payload_does_not_expose_expected_probe_answers(sample_cluster):
    snapshot, backend_probes, public_questions = freeze_compression_snapshot(sample_cluster)
    assert backend_probes
    assert public_questions
    for probe in backend_probes:
        assert probe.expected_answer not in json.dumps(public_questions, ensure_ascii=False)
```

- [ ] **Step 2: Write a failing test that the same frozen probe identity is reused for both arms**

```python
def test_method_present_and_ablation_share_probe_set(sample_cluster):
    snapshot, probes, _ = freeze_compression_snapshot(sample_cluster)
    with_method = build_reconstruction_trial(snapshot, probes, include_method=True)
    without_method = build_reconstruction_trial(snapshot, probes, include_method=False)
    assert with_method.probe_set_hash == without_method.probe_set_hash == snapshot.probe_set_hash
```

- [ ] **Step 3: Run focused tests and verify RED**

```bash
pytest tests/test_compression_test.py -q
```

Expected: missing freeze/build functions.

- [ ] **Step 4: Implement deterministic freezing and probe construction**

Canonicalize the pre-test cluster and durable epistemic records; compute hashes from canonical JSON; generate question-only public probes and keep expected structured answers backend-side. Do not ask a model to rate fidelity.

- [ ] **Step 5: Re-run focused tests and confirm GREEN**

```bash
pytest tests/test_compression_test.py -q
```

- [ ] **Step 6: Commit**

```bash
git add src/dte_backend/compression_test.py tests/test_compression_test.py
git commit -m "feat: freeze hidden reconstruction probes"
```

---

### Task 3: Measure fidelity and compression without a Judge score

**Files:**
- Modify: `src/dte_backend/compression_test.py`
- Modify: `tests/test_compression_test.py`

**Interfaces:**
- `score_reconstruction(snapshot, probes, result) -> ReconstructionScore` computes probe recovery and representation length from frozen inputs.
- `evaluate_compression(...) -> CompressionNecessityDecision` cannot inspect controller UCB or mutate graph state.

- [ ] **Step 1: Write failing tests for perfect recovery, information loss, and fake verbosity**

```python
def test_short_full_reconstruction_passes_compression(snapshot, probes):
    result = reconstruction_answering_all_probes_compactly(probes)
    score = score_reconstruction(snapshot, probes, result)
    assert score.fidelity == 1.0
    assert score.compact_token_length < snapshot.original_token_length


def test_short_reconstruction_with_missing_counterexample_fails_fidelity(snapshot, probes):
    result = reconstruction_missing_one_material_probe(probes)
    score = score_reconstruction(snapshot, probes, result)
    assert score.fidelity < 1.0


def test_full_but_long_reconstruction_does_not_count_as_compression(snapshot, probes):
    result = verbose_reconstruction_answering_all_probes(probes)
    decision = evaluate_compression(snapshot, probes, result)
    assert decision.compression_passed is False
```

- [ ] **Step 2: Run focused tests and verify RED**

```bash
pytest tests/test_compression_test.py -q
```

- [ ] **Step 3: Implement deterministic fidelity accounting and token-length accounting**

Fidelity is recovered material probe mass divided by total material probe mass. The first implementation uses equal probe mass unless a durable pre-test epistemic type requires an explicitly specified weight in the spec; no post-hoc model-authored weights are allowed. Representation length uses the repository's fixed tokenizer/counting utility and includes method text, residual context, and reconstructed compact state.

- [ ] **Step 4: Re-run focused tests and confirm GREEN**

```bash
pytest tests/test_compression_test.py -q
```

- [ ] **Step 5: Commit**

```bash
git add src/dte_backend/compression_test.py tests/test_compression_test.py
git commit -m "feat: measure reconstruction fidelity and compression"
```

---

### Task 4: Add the paired necessity ablation

**Files:**
- Modify: `src/dte_backend/compression_test.py`
- Modify: `tests/test_compression_test.py`

**Interfaces:**
- Both arms use the same `CompressionSnapshot`, probe set, model identity, runtime limits, and maximum returned representation length.
- `necessity_gap` is measured at a fixed target fidelity `R0`: the extra compact representation cost required when the candidate method is forbidden, or a fidelity deficit when the ablation cannot reach `R0` within the same budget.

- [ ] **Step 1: Write a failing test where a replaceable summary is not necessary**

```python
def test_equally_short_ablation_blocks_necessity_grant(snapshot, probes):
    with_method = compact_full_reconstruction(probes)
    without_method = equally_compact_full_reconstruction(probes)
    decision = evaluate_paired_reconstruction(snapshot, probes, with_method, without_method)
    assert decision.compression_passed is True
    assert decision.necessity_passed is False
```

- [ ] **Step 2: Write a failing test where withholding the method forces a much longer representation**

```python
def test_method_with_large_ablation_cost_passes_necessity(snapshot, probes):
    with_method = compact_full_reconstruction(probes)
    without_method = long_full_reconstruction(probes)
    decision = evaluate_paired_reconstruction(snapshot, probes, with_method, without_method)
    assert decision.fidelity_passed is True
    assert decision.compression_passed is True
    assert decision.necessity_passed is True
```

- [ ] **Step 3: Run focused tests and verify RED**

```bash
pytest tests/test_compression_test.py -q
```

- [ ] **Step 4: Implement the paired decision rule**

Use predeclared thresholds in run/controller configuration. Necessity passes when the method-present arm reaches target fidelity compactly and the method-withheld arm either misses target fidelity under the same bound or needs at least the configured additional representation cost. Persist the raw measurements and thresholds so the decision can be replayed.

- [ ] **Step 5: Re-run focused tests and confirm GREEN**

```bash
pytest tests/test_compression_test.py -q
```

- [ ] **Step 6: Commit**

```bash
git add src/dte_backend/compression_test.py tests/test_compression_test.py
git commit -m "feat: add compression necessity ablation"
```

---

### Task 5: Add bounded reconstruction episodes to the transport-neutral protocol

**Files:**
- Modify: `src/dte_backend/episode_models.py`
- Modify: `src/dte_backend/episode_adapter.py`
- Test: `tests/test_app_compression_grant.py`

**Interfaces:**
- Add `role="reconstruction"` with a versioned payload containing candidate method text when `include_method=true`, residual context, hidden-probe questions only, maximum returned compact representation length, and snapshot/probe hashes.
- The result returns the compact representation plus structured answers keyed by opaque probe IDs. It returns no score, UCB, allocation, graph revision, or grant decision.

- [ ] **Step 1: Write a failing protocol test**

```python
def test_reconstruction_episode_cannot_pre_authorize_grant(app_state):
    request = build_reconstruction_episode_request(...)
    assert request.role == "reconstruction"
    dumped = request.model_dump(mode="json")
    assert "expected_answer" not in json.dumps(dumped)
    assert "grant" not in request.allowed_output_types
```

- [ ] **Step 2: Run the focused integration test and verify RED**

```bash
pytest tests/test_app_compression_grant.py -q
```

- [ ] **Step 3: Implement the minimal episode models and request builder**

Reuse existing role isolation and runtime diagnostics contracts. Bind both reconstruction arms to the same snapshot hash and probe set hash.

- [ ] **Step 4: Re-run focused tests and confirm GREEN**

```bash
pytest tests/test_app_compression_grant.py -q
```

- [ ] **Step 5: Commit**

```bash
git add src/dte_backend/episode_models.py src/dte_backend/episode_adapter.py tests/test_app_compression_grant.py
git commit -m "feat: add bounded reconstruction episodes"
```

---

### Task 6: Compact active state and issue one full-round budget grant

**Files:**
- Modify: `src/dte_backend/app_driver.py`
- Modify: `src/dte_backend/models.py`
- Modify: `tests/test_app_compression_grant.py`

**Interfaces:**
- `maybe_schedule_compression_test(state) -> EpisodeRequest | None` triggers only for a predeclared high-weight/high-allocation candidate cluster and never consumes ordinary node budget by itself.
- `apply_compression_decision(state, decision) -> CompressionNecessityGrant | None` creates a compact macro-node only after fidelity + compression + necessity pass.
- The compact macro-node stores references to all absorbed node IDs and epistemic record IDs; those historical objects remain unchanged.
- A valid unused grant sets the next iteration's available expansion mass for the method to one configured full round, clipped by remaining hard node budget; consuming it atomically marks the grant used.

- [ ] **Step 1: Write a failing test that history is preserved while active representation shrinks**

```python
def test_passing_compaction_replaces_active_cluster_but_preserves_history(state, passing_decision):
    before_ids = {node.node_id for node in state.nodes}
    apply_compression_decision(state, passing_decision)
    assert before_ids.issubset({node.node_id for node in state.nodes})
    assert sum(node.status == "frontier" for node in state.nodes) < len(before_ids)
    assert any(node.node_type == "merge" and node.absorbed_node_ids for node in state.nodes)
```

- [ ] **Step 2: Write a failing test that a passed grant bypasses generic review and receives one full round only**

```python
def test_passed_compression_grant_gets_exactly_one_full_round(state, passing_decision):
    grant = apply_compression_decision(state, passing_decision)
    first = allocate_next_iteration(state)
    assert first.allocations[grant.method_node_id] == min(
        state.spec.budget.allocation_mass_per_iteration,
        remaining_search_node_slots(state),
    )
    second = allocate_next_iteration(state)
    assert grant.consumed is True
    assert second.used_compression_grant_id is None
```

- [ ] **Step 3: Run focused integration tests and verify RED**

```bash
pytest tests/test_app_compression_grant.py -q
```

- [ ] **Step 4: Implement minimal active-state compaction and single-use grant consumption**

Do not alter historical nodes or ledger records. Do not add a Judge/reviewer gate. The only authority for the extra round is the durable paired reconstruction decision plus remaining hard budget.

- [ ] **Step 5: Re-run focused integration tests and confirm GREEN**

```bash
pytest tests/test_app_compression_grant.py -q
```

- [ ] **Step 6: Commit**

```bash
git add src/dte_backend/app_driver.py src/dte_backend/models.py tests/test_app_compression_grant.py
git commit -m "feat: grant full round after necessary compression"
```

---

### Task 7: Persist, replay, and expose compression events

**Files:**
- Modify: `src/dte_backend/app_driver.py`
- Modify: `src/dte_backend/observability_models.py`
- Modify: `src/dte_backend/observability.py`
- Modify: `tests/test_app_compression_grant.py`
- Modify: `tests/test_observability.py`

**Interfaces:**
- Persist frozen snapshot, both reconstruction results, decision, macro-node mapping, grant issuance, and grant consumption.
- Reload validation recomputes hashes and rejects any grant whose frozen snapshot, probe identity, or reconstruction results disagree with durable state.
- Observability reports attempted, passed, rejected, and consumed grants without presenting them as scientific truth verification.

- [ ] **Step 1: Write failing replay/tamper tests**

```python
def test_reloaded_grant_recomputes_same_decision(run_dir):
    state = load_app_run(run_dir)
    assert state.compression_grants[0].snapshot_hash == state.compression_snapshots[0].snapshot_hash


def test_tampered_ablation_result_invalidates_persisted_grant(run_dir):
    tamper_method_withheld_result(run_dir)
    with pytest.raises(ValueError):
        load_app_run(run_dir)
```

- [ ] **Step 2: Run focused tests and verify RED**

```bash
pytest tests/test_app_compression_grant.py tests/test_observability.py -q
```

- [ ] **Step 3: Implement persistence validation and observability**

Persist raw measured quantities, not only the final boolean. Include model/runtime identity and probe-set hash in the replay contract.

- [ ] **Step 4: Re-run focused tests and confirm GREEN**

```bash
pytest tests/test_app_compression_grant.py tests/test_observability.py -q
```

- [ ] **Step 5: Commit**

```bash
git add src/dte_backend/app_driver.py src/dte_backend/observability_models.py src/dte_backend/observability.py tests/test_app_compression_grant.py tests/test_observability.py
git commit -m "feat: persist compression necessity grants"
```

---

### Task 8: Update controller authority docs and theory locks

**Files:**
- Modify: `docs/PHYSICS.md`
- Modify: `docs/DESIGN.md`
- Modify: `SPEC.md`
- Modify: theory-lock test/digest files discovered by `rg "PHYSICS.md|DESIGN.md|sha256" tests scripts`

**Interfaces:**
- Documents define the grant as an event-triggered allocation override outside `U`, with immutable provenance and one-shot semantics.

- [ ] **Step 1: Add a failing theory-lock expectation for the new normative text**

The test must require all four properties: measured reconstruction fidelity, method-withheld necessity ablation, active-state-only compaction, and single-use full-round grant.

- [ ] **Step 2: Run the theory-lock test and verify RED**

```bash
pytest -q -k "theory_lock or physics or design"
```

- [ ] **Step 3: Update the normative documents and regenerate exact theory-lock digests**

Do not redefine `V`, `SD`, or `U`. State explicitly that the compression grant does not imply scientific truth and does not erase provenance.

- [ ] **Step 4: Re-run theory-lock tests and confirm GREEN**

```bash
pytest -q -k "theory_lock or physics or design"
```

- [ ] **Step 5: Commit**

```bash
git add docs/PHYSICS.md docs/DESIGN.md SPEC.md tests scripts
git commit -m "docs: specify compression necessity grant"
```

---

### Task 9: Full regression and public CI

**Files:**
- No new production behavior.

- [ ] **Step 1: Run the full local suite**

```bash
pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Run repository smoke/package checks used by CI**

```bash
python scripts/smoke_workflow.py
python scripts/generate_bundle_manifest.py
pytest -q
```

- [ ] **Step 3: Verify no generic Judge dependency was introduced by the compression feature**

```bash
rg -n "judge|Judge" src/dte_backend tests SKILL.md SPEC.md docs/PHYSICS.md docs/DESIGN.md
```

Expected: only explicitly preserved legacy/migration text, if any; no compression gate calls or requires a generic Judge role.

- [ ] **Step 4: Push the implementation branch and require public GitHub Actions CI to pass before merge**

- [ ] **Step 5: Final commit only if generated manifest or verification artifacts changed**

```bash
git add .
git commit -m "chore: refresh release verification artifacts"
```
