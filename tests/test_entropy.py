import math

from dte_backend.entropy import evaluate_entropy_state, spatial_entropy_from_embeddings


def test_spatial_entropy_increases_for_spread_nodes():
    identical = spatial_entropy_from_embeddings([[1, 0], [1, 0], [1, 0]])
    spread = spatial_entropy_from_embeddings([[1, 0], [0, 1], [-1, 0]])
    assert spread >= identical


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


def test_temperature_endpoints_and_singleton():
    cold = evaluate_entropy_state(
        spatial_entropy=0.0,
        frontier_size=4,
        previous_entropy=None,
        iteration=1,
        min_iterations=2,
        entropy_change_threshold=0.05,
        t_max=2.0,
    )
    hot = evaluate_entropy_state(
        spatial_entropy=math.log(4),
        frontier_size=4,
        previous_entropy=None,
        iteration=1,
        min_iterations=2,
        entropy_change_threshold=0.05,
        t_max=2.0,
    )
    singleton = evaluate_entropy_state(
        spatial_entropy=0.0,
        frontier_size=1,
        previous_entropy=None,
        iteration=1,
        min_iterations=2,
        entropy_change_threshold=0.05,
        t_max=2.0,
    )

    assert (cold.normalized_temperature, cold.effective_temperature) == (0.0, 0.0)
    assert (hot.normalized_temperature, hot.effective_temperature) == (1.0, 2.0)
    assert (singleton.normalized_temperature, singleton.effective_temperature) == (0.0, 0.0)


def test_previous_entropy_changes_delta_but_not_current_temperature():
    kwargs = dict(
        spatial_entropy=0.5 * math.log(4),
        frontier_size=4,
        iteration=3,
        min_iterations=2,
        entropy_change_threshold=0.05,
        t_max=2.0,
    )
    far = evaluate_entropy_state(previous_entropy=0.1, **kwargs)
    near = evaluate_entropy_state(previous_entropy=0.69, **kwargs)

    assert far.normalized_temperature == near.normalized_temperature
    assert far.effective_temperature == near.effective_temperature
    assert far.entropy_delta != near.entropy_delta


def test_entropy_plateau_requires_configured_confirmations():
    first = evaluate_entropy_state(
        spatial_entropy=1.0,
        frontier_size=4,
        previous_entropy=1.01,
        iteration=2,
        min_iterations=2,
        entropy_change_threshold=0.05,
        previous_plateau_count=0,
        plateau_confirmations=2,
    )
    assert not first.plateau_signal
    assert first.consecutive_plateau_count == 1

    second = evaluate_entropy_state(
        spatial_entropy=1.0,
        frontier_size=4,
        previous_entropy=1.0,
        iteration=3,
        min_iterations=2,
        entropy_change_threshold=0.05,
        previous_plateau_count=first.consecutive_plateau_count,
        plateau_confirmations=2,
    )
    assert second.plateau_signal
    assert second.consecutive_plateau_count == 2
    assert second.stop_reason == "entropy_plateau"
