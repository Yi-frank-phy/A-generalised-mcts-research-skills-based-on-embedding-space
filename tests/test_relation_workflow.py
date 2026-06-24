from dte_backend.models import SearchNode
from dte_backend.oracles import RelationOracleResult
from dte_backend.relation_workflow import relation_result_to_outputs


def test_equivalent_relation_creates_merge_proposal():
    nodes = [
        SearchNode(node_id="a", claim="A", confidence=0.7),
        SearchNode(node_id="b", claim="A", confidence=0.4),
    ]
    proposal, task = relation_result_to_outputs(
        RelationOracleResult(relation="equivalent", source_node_ids=["a", "b"], rationale="same route"),
        nodes,
    )
    assert task is None
    assert proposal is not None
    assert proposal.merge_type == "equivalent_merge"
    assert proposal.target_node_id == "a"
    assert proposal.absorbed_node_ids == ["b"]


def test_complementary_relation_creates_frontier_merge_node():
    nodes = [
        SearchNode(node_id="a", claim="A", assumptions=["x"], evidence=["e1"], confidence=0.6),
        SearchNode(node_id="b", claim="B", assumptions=["y"], evidence=["e2"], confidence=0.5),
    ]
    proposal, task = relation_result_to_outputs(
        RelationOracleResult(relation="complementary", source_node_ids=["a", "b"], rationale="routes support each other"),
        nodes,
    )
    assert task is None
    assert proposal is not None
    assert proposal.merge_type == "complementary_merge"
    assert proposal.merged_node is not None
    assert proposal.merged_node.status == "frontier"
    assert set(proposal.merged_node.parent_ids) == {"a", "b"}


def test_conflict_relation_creates_discriminator_task():
    nodes = [SearchNode(node_id="a", claim="A"), SearchNode(node_id="b", claim="B")]
    proposal, task = relation_result_to_outputs(
        RelationOracleResult(
            relation="conflict",
            source_node_ids=["a", "b"],
            rationale="assumptions clash",
            discriminator_question="Which assumption is necessary?",
        ),
        nodes,
    )
    assert proposal is not None
    assert proposal.merge_type == "conflict_merge"
    assert task is not None
    assert task.node_type == "counterexample"
    assert set(task.parent_ids) == {"a", "b"}


def test_independent_relation_is_noop():
    nodes = [SearchNode(node_id="a", claim="A"), SearchNode(node_id="b", claim="B")]
    proposal, task = relation_result_to_outputs(
        RelationOracleResult(relation="independent", source_node_ids=["a", "b"], rationale="separate"),
        nodes,
    )
    assert proposal is None
    assert task is None
