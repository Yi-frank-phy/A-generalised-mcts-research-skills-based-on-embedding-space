from dte_backend.models import DTERunSpec
from dte_backend.role_pipeline import seed_frontier_from_roles
from dte_backend.runner import run_frontier_search


def test_seed_pipeline_creates_nodes():
    spec = DTERunSpec(problem="p", goal="g", constraints=["c1"])
    nodes, audit = seed_frontier_from_roles(spec)
    assert len(nodes) >= 4
    assert "decomposition" in audit
    assert len({node.node_id for node in nodes}) == len(nodes)


def test_runner_seeds_when_no_nodes_are_given():
    spec = DTERunSpec(problem="p", goal="g")
    result = run_frontier_search(spec, initial_nodes=None)
    assert result.role_audit
    assert len(result.nodes) >= 4
