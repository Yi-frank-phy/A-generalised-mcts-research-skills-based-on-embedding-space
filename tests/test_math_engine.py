from dte_backend.math_engine import allocate_frontier, boltzmann_allocation, calculate_ucb
from dte_backend.models import SearchNode


def test_ucb_default_not_cost_aware():
    assert calculate_ucb(score=0.5, uncertainty=0.2, tau=1.0, c_explore=1.0) == 0.7


def test_boltzmann_allocation_nonempty_budget():
    allocation = boltzmann_allocation([0.2, 0.8], total_budget=3, temperature=1.0)
    assert sum(allocation) >= 1
    assert len(allocation) == 2


def test_allocate_frontier():
    nodes = [
        SearchNode(node_id="a", claim="A", score=0.7, uncertainty=0.2),
        SearchNode(node_id="b", claim="B", score=0.4, uncertainty=0.8),
    ]
    result = allocate_frontier(nodes, total_budget=3)
    assert len(result) == 2
    assert all(r.expansion_budget >= 0 for r in result)
