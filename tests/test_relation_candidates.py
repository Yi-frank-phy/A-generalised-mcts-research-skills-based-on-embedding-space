from dte_backend.models import SearchNode
from dte_backend.relation_candidates import select_relation_candidate_pairs


def test_duplicate_claim_selected_for_relation_oracle():
    nodes = [
        SearchNode(node_id="a", claim="Same claim", confidence=0.5),
        SearchNode(node_id="b", claim=" same   CLAIM ", confidence=0.4),
    ]
    pairs = select_relation_candidate_pairs(nodes)
    assert pairs
    assert pairs[0].node_ids == ("a", "b")
    assert "duplicate" in pairs[0].reason


def test_near_tied_branches_selected():
    nodes = [
        SearchNode(node_id="a", claim="A", ucb_score=0.50),
        SearchNode(node_id="b", claim="B", ucb_score=0.53),
        SearchNode(node_id="c", claim="C", ucb_score=0.10),
    ]
    pairs = select_relation_candidate_pairs(nodes, tie_threshold=0.05)
    assert any(pair.node_ids == ("a", "b") for pair in pairs)


def test_embedding_close_branches_selected():
    nodes = [
        SearchNode(node_id="a", claim="A", local_embedding=[1.0, 0.0]),
        SearchNode(node_id="b", claim="B", local_embedding=[0.99, 0.01]),
        SearchNode(node_id="c", claim="C", local_embedding=[0.0, 1.0]),
    ]
    pairs = select_relation_candidate_pairs(nodes, semantic_distance_threshold=0.05)
    assert any(pair.node_ids == ("a", "b") for pair in pairs)


def test_no_frontier_pair_returns_empty():
    nodes = [SearchNode(node_id="a", claim="A", status="closed")]
    assert select_relation_candidate_pairs(nodes) == []
