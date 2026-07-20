from dte_backend.entropy import evaluate_entropy_state, spatial_entropy_from_embeddings


def test_spatial_entropy_increases_for_spread_nodes():
    identical = spatial_entropy_from_embeddings([[1, 0], [1, 0], [1, 0]])
    spread = spatial_entropy_from_embeddings([[1, 0], [0, 1], [-1, 0]])
    assert spread >= identical


def test_entropy_plateau_requires_configured_confirmations():
    first = evaluate_entropy_state(
        spatial_entropy=1.0,
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
