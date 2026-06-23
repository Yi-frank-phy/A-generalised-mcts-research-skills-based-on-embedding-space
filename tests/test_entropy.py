from dte_backend.entropy import evaluate_entropy_state, spatial_entropy_from_embeddings


def test_spatial_entropy_increases_for_spread_nodes():
    identical = spatial_entropy_from_embeddings([[1, 0], [1, 0], [1, 0]])
    spread = spatial_entropy_from_embeddings([[1, 0], [0, 1], [-1, 0]])
    assert spread >= identical


def test_entropy_plateau_triggers_after_min_iteration():
    state = evaluate_entropy_state(
        spatial_entropy=1.0,
        previous_entropy=1.01,
        iteration=2,
        min_iterations=2,
        entropy_change_threshold=0.05,
    )
    assert state.should_synthesize
    assert state.stop_reason == "entropy_plateau"
