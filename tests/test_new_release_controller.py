import numpy as np
import pytest

from dte_backend.embedding import HashEmbeddingProvider
from dte_backend.episode_models import ExecutorNodeCandidate
from dte_backend.models import SearchNode
from dte_backend.new_controller import freeze_reference_atlas, score_frontier
from dte_backend.novelty import estimate_frontier_kde_state
from dte_backend.transition_state import canonical_transition_text


class StubEmbeddingProvider:
    name = "stub-transition"
    model = "stub-transition-v1"
    dim = 2

    _angles = {
        "direct construction": 0.0,
        "invariant rewrite": 0.35,
        "duality change": 0.75,
        "counterexample boundary": 1.2,
    }

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            matched = next((angle for key, angle in self._angles.items() if key in text), None)
            if matched is None:
                raise AssertionError(f"unexpected transition text: {text}")
            vectors.append([float(np.cos(matched)), float(np.sin(matched))])
        return vectors


def node(
    node_id: str,
    method: str,
    *,
    parent_ids: list[str] | None = None,
    score: float | None = None,
    status: str = "frontier",
    claim: str | None = None,
) -> SearchNode:
    return SearchNode(
        node_id=node_id,
        claim=claim or f"claim-{node_id}",
        parent_ids=parent_ids or [],
        score=score,
        status=status,
        retrospective_method=method,
        epistemic_change_kind="new_understanding",
        epistemic_change=f"change from {method}",
    )




def test_active_node_without_completed_transition_fails_closed() -> None:
    incomplete = SearchNode(node_id="x", claim="legacy node")

    with pytest.raises(ValueError, match="completed transition"):
        canonical_transition_text(incomplete)


def test_executor_candidate_requires_completed_transition_fields() -> None:
    with pytest.raises(ValueError):
        ExecutorNodeCandidate(node_id="child", claim="child claim", parent_ids=["parent"])


def test_single_initial_completed_transition_uses_packaged_reference_atlas() -> None:
    initial = [node("a", "direct construction")]

    frontier, state = estimate_frontier_kde_state(
        initial,
        provider=HashEmbeddingProvider(dim=8),
        expected_dimension=8,
        graph_k=1,
        volume_bandwidth=1.0,
    )

    assert [item.node_id for item in frontier] == ["a"]
    assert state.value_source == "proper_volume_history"
    assert state.sd_source == "proper_volume_boltzmann_reward"




def test_realized_parent_child_edge_updates_value_on_frozen_atlas() -> None:
    provider = StubEmbeddingProvider()
    initial = [
        node("a", "direct construction"),
        node("b", "invariant rewrite"),
        node("c", "duality change"),
        node("d", "counterexample boundary"),
    ]
    atlas = freeze_reference_atlas(initial, provider=provider, graph_k=1)
    retired = node("a", "direct construction", status="closed")
    child = node("child", "duality change", parent_ids=["a"])
    graph = [retired, initial[1], initial[2], initial[3], child]
    live = [initial[1], initial[2], initial[3], child]

    scored = score_frontier(
        graph_nodes=graph,
        live_nodes=live,
        atlas=atlas,
        provider=provider,
        volume_bandwidth=1.0,
    )

    assert scored.realized_returns.size == 1
    assert scored.realized_returns[0] > 1.0
    assert np.all(scored.values > 0.0)
    assert np.allclose(scored.ucb_scores, scored.values + scored.standard_deviations)


def test_frozen_atlas_identity_does_not_change_when_graph_grows() -> None:
    provider = StubEmbeddingProvider()
    initial = [
        node("a", "direct construction"),
        node("b", "invariant rewrite"),
        node("c", "duality change"),
        node("d", "counterexample boundary"),
    ]
    atlas = freeze_reference_atlas(initial, provider=provider, graph_k=1)
    identity = atlas.identity

    graph = initial + [node("child", "duality change", parent_ids=["a"])]
    score_frontier(
        graph_nodes=graph,
        live_nodes=initial[1:],
        atlas=atlas,
        provider=provider,
        volume_bandwidth=1.0,
    )

    assert atlas.identity == identity
    assert atlas.node_ids == ("a", "b", "c", "d")
