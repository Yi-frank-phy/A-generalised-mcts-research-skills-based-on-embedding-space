import numpy as np
import pytest

from dte_nextgen.thought_space.controller import score_proper_volume_embeddings
from dte_nextgen.thought_space.history import TransitionHistory
from dte_nextgen.thought_space.session import ProperVolumeTransitionSession
from dte_nextgen.thought_space.transition import MethodEpistemicTransition


def _transition(method: str, change: str = "structure") -> MethodEpistemicTransition:
    return MethodEpistemicTransition(
        retrospective_method=method,
        epistemic_change_kind="new_understanding",
        epistemic_change=change,
    )


def _references() -> tuple[MethodEpistemicTransition, ...]:
    return (
        _transition("group rewrite"),
        _transition("invariant scan"),
        _transition("duality map"),
        _transition("boundary audit"),
        _transition("counterexample search", "exception family"),
    )


def _fake_embed(text: str) -> list[float]:
    angles = {
        "group rewrite": 0.0,
        "invariant scan": 0.3,
        "duality map": 0.6,
        "boundary audit": 0.9,
        "counterexample search": 1.2,
    }
    for method, angle in angles.items():
        if method in text:
            return [float(np.cos(angle)), float(np.sin(angle))]
    raise AssertionError(f"unexpected canonical text: {text}")


def test_end_to_end_controller_uses_same_proper_volume_scale_for_value_and_sd() -> None:
    reference_embeddings = np.asarray([_fake_embed(t.canonical_text()) for t in _references()])
    live = np.asarray(
        [
            _fake_embed(_transition("group rewrite").canonical_text()),
            _fake_embed(_transition("group rewrite").canonical_text()),
            _fake_embed(_transition("counterexample search", "exception family").canonical_text()),
        ]
    )

    result = score_proper_volume_embeddings(
        live_embeddings=live,
        propulsion_values=np.asarray([0.4, 0.4, 0.2]),
        reference_embeddings=reference_embeddings,
        graph_k=1,
        volume_bandwidth=1.0,
    )

    assert result["reference_density_source"] == "uniform_frozen_reference_measure"
    assert result["sd_source"] == "proper_volume_boltzmann_reward"
    assert result["standard_deviations"][0] < result["standard_deviations"][2]
    assert np.allclose(
        result["ucb_scores"], result["values"] + result["standard_deviations"]
    )
    assert 0.0 <= result["allocation_target_entropy"] <= np.log(len(live))


def test_session_closes_select_execute_record_replace_rescore_loop() -> None:
    references = _references()
    parent = _transition("group rewrite")
    other = _transition("counterexample search", "exception family")
    child = _transition("duality map", "new representation")
    session = ProperVolumeTransitionSession(
        node_ids=("parent", "other"),
        frontier=(parent, other),
        reference_transitions=references,
        embed_fn=_fake_embed,
        graph_k=1,
        volume_bandwidth=1.0,
    )

    before = session.score()
    assert np.allclose(before["values"], 0.0)
    assert before["value_source"] == "zero_before_history"

    selected = session.select(selection_quantile=0.25)
    assert np.isclose(np.sum(selected["probabilities"]), 1.0)
    assert np.allclose(
        selected["ucb_scores"], selected["values"] + selected["standard_deviations"]
    )

    completed = session.complete(
        parent_index=0,
        child_node_id="child",
        child_transition=child,
    )

    assert completed["parent_node_id"] == "parent"
    assert completed["child_node_id"] == "child"
    assert completed["observed_return"] > 1.0
    assert session.node_ids == ("child", "other")
    assert session.frontier[0] == child
    assert session.reference_transitions == references

    history_embeddings, history_returns, history_counts = session.history.as_arrays()
    assert history_embeddings.shape == (1, 2)
    assert np.allclose(history_returns, [completed["observed_return"]])
    assert np.array_equal(history_counts, [1])

    after = session.score()
    assert after["value_source"] == "proper_volume_history_kernel"
    assert np.all(after["values"] > 0.0)


def test_history_value_regression_is_local_in_proper_volume_not_an_sd_estimator() -> None:
    left = _transition("group rewrite")
    right = _transition("counterexample search", "exception family")
    session = ProperVolumeTransitionSession(
        node_ids=("left", "right"),
        frontier=(left, right),
        reference_transitions=_references(),
        embed_fn=_fake_embed,
        graph_k=1,
        volume_bandwidth=1.0,
    )
    session.history.record("old-left", np.asarray(_fake_embed(left.canonical_text())), 1.0)
    session.history.record("old-right", np.asarray(_fake_embed(right.canonical_text())), 9.0)

    scored = session.score()

    assert scored["value_source"] == "proper_volume_history_kernel"
    assert scored["values"][0] < 5.0
    assert scored["values"][1] > 5.0
    assert scored["sd_source"] == "proper_volume_boltzmann_reward"


def test_session_rejects_pre_numeric_history_without_frozen_atlas_identity() -> None:
    history = TransitionHistory()
    history.record("old", np.asarray(_fake_embed(_transition("group rewrite").canonical_text())), 3.0)

    with pytest.raises(ValueError, match="pre-populated numeric history"):
        ProperVolumeTransitionSession(
            node_ids=("left", "right"),
            frontier=(_transition("group rewrite"), _transition("boundary audit")),
            reference_transitions=_references(),
            embed_fn=_fake_embed,
            graph_k=1,
            history=history,
        )
