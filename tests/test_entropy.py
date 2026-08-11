import math

import numpy as np

from dte_backend.entropy import evaluate_entropy_state, spatial_entropy_from_embeddings


def _boltzmann_entropy(scores: list[float], temperature: float) -> float:
    values = np.asarray(scores, dtype=float)
    if temperature <= 0.0:
        winners = values == np.max(values)
        probabilities = winners.astype(float) / float(np.sum(winners))
    else:
        scaled = values / float(temperature)
        scaled -= np.max(scaled)
        weights = np.exp(scaled)
        probabilities = weights / np.sum(weights)
    positive = probabilities > 0.0
    return float(-np.sum(probabilities[positive] * np.log(probabilities[positive])))


def test_spatial_entropy_increases_for_spread_nodes():
    identical = spatial_entropy_from_embeddings([[1, 0], [1, 0], [1, 0]])
    spread = spatial_entropy_from_embeddings([[1, 0], [0, 1], [-1, 0]])
    assert spread >= identical


def test_effective_temperature_matches_current_geometry_entropy_for_current_ucbs():
    scores = [0.1, 0.3, 0.8, 1.2]
    target_entropy = 0.5 * math.log(4)
    state = evaluate_entropy_state(
        spatial_entropy=target_entropy,
        frontier_size=4,
        ucb_scores=scores,
        previous_entropy=None,
        iteration=1,
        min_iterations=2,
        entropy_change_threshold=0.05,
        t_max=2.0,
    )

    assert state.normalized_temperature == 0.5  # deprecated normalized-entropy telemetry
    assert math.isclose(
        _boltzmann_entropy(scores, state.effective_temperature),
        target_entropy,
        rel_tol=0.0,
        abs_tol=1e-8,
    )


def test_temperature_tracks_ucb_scale_instead_of_linear_h_over_log_n_rule():
    target_entropy = 0.7
    scores = [0.1, 0.3, 0.8]
    scaled_scores = [10.0 * value for value in scores]
    common = dict(
        spatial_entropy=target_entropy,
        frontier_size=3,
        previous_entropy=None,
        iteration=1,
        min_iterations=2,
        entropy_change_threshold=0.05,
        t_max=2.0,
    )

    base = evaluate_entropy_state(ucb_scores=scores, **common)
    scaled = evaluate_entropy_state(ucb_scores=scaled_scores, **common)

    assert math.isclose(
        _boltzmann_entropy(scores, base.effective_temperature),
        target_entropy,
        rel_tol=0.0,
        abs_tol=1e-8,
    )
    assert math.isclose(
        _boltzmann_entropy(scaled_scores, scaled.effective_temperature),
        target_entropy,
        rel_tol=0.0,
        abs_tol=1e-8,
    )
    assert math.isclose(
        scaled.effective_temperature,
        10.0 * base.effective_temperature,
        rel_tol=1e-6,
        abs_tol=1e-8,
    )


def test_zero_entropy_and_singleton_have_zero_temperature():
    cold = evaluate_entropy_state(
        spatial_entropy=0.0,
        frontier_size=4,
        ucb_scores=[0.0, 0.1, 0.2, 0.3],
        previous_entropy=None,
        iteration=1,
        min_iterations=2,
        entropy_change_threshold=0.05,
        t_max=2.0,
    )
    singleton = evaluate_entropy_state(
        spatial_entropy=0.0,
        frontier_size=1,
        ucb_scores=[0.3],
        previous_entropy=None,
        iteration=1,
        min_iterations=2,
        entropy_change_threshold=0.05,
        t_max=2.0,
    )

    assert (cold.normalized_temperature, cold.effective_temperature) == (0.0, 0.0)
    assert (singleton.normalized_temperature, singleton.effective_temperature) == (0.0, 0.0)


def test_previous_entropy_changes_delta_but_not_current_temperature():
    kwargs = dict(
        spatial_entropy=0.5 * math.log(4),
        frontier_size=4,
        ucb_scores=[0.1, 0.3, 0.8, 1.2],
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
        ucb_scores=[0.1, 0.3, 0.8, 1.2],
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
        ucb_scores=[0.1, 0.3, 0.8, 1.2],
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
