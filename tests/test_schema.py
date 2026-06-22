from dte_backend.models import DTERunSpec, SearchNode


def test_run_spec_validates():
    spec = DTERunSpec(
        problem="p",
        goal="g",
        budget={"max_iterations": 2, "total_child_budget": 3, "max_research_iterations": 1},
    )
    assert spec.mode == "mandatory_frontier"


def test_search_node_validates():
    node = SearchNode(node_id="n", claim="claim")
    assert node.status == "frontier"
