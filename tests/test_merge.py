from dte_backend.merge import apply_equivalent_merges, propose_equivalent_merges
from dte_backend.models import SearchNode


def test_equivalent_merge_marks_lower_ranked_duplicate():
    nodes = [
        SearchNode(node_id="a", claim="Same claim", confidence=0.9),
        SearchNode(node_id="b", claim=" same   CLAIM ", confidence=0.4),
        SearchNode(node_id="c", claim="different", confidence=0.5),
    ]
    proposals = propose_equivalent_merges(nodes)
    assert len(proposals) == 1
    assert proposals[0].target_node_id == "a"
    applied = apply_equivalent_merges(nodes)
    assert applied
    assert next(n for n in nodes if n.node_id == "b").status == "merged"
    assert next(n for n in nodes if n.node_id == "a").status == "frontier"
