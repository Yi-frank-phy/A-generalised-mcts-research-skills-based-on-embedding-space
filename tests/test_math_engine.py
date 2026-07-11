from dte_backend.math_engine import (
    allocate_frontier,
    boltzmann_allocation,
    calculate_ucb,
    discretize_allocation,
)
from dte_backend.models import SearchNode


def test_ucb_default_not_cost_aware():
    assert calculate_ucb(score=0.5, uncertainty=0.2, tau=1.0, c_explore=1.0) == 0.7


def test_boltzmann_allocation_nonempty_budget():
    allocation = boltzmann_allocation(
        [0.2, 0.8],
        allocation_mass_per_iteration=3,
        max_children_per_iteration=5,
        node_ids=["a", "b"],
        temperature=1.0,
    )
    assert sum(allocation) >= 1
    assert len(allocation) == 2


def test_discretize_allocation_uses_round_half_up_below_one():
    assert discretize_allocation([0.5], [1.0], ["a"], max_children_per_iteration=5) == [1]


def test_discretize_allocation_matches_normative_example():
    allocation = discretize_allocation(
        [0.7, 0.6, 0.3, 1.4],
        [0.7, 0.6, 0.3, 1.4],
        ["a", "b", "c", "d"],
        max_children_per_iteration=5,
    )
    assert allocation == [1, 1, 0, 2]
    assert sum(allocation) == 4


def test_hard_cap_trims_six_equal_tentative_children_to_five():
    allocation = discretize_allocation(
        [0.5] * 6,
        [1.0] * 6,
        ["f", "e", "d", "c", "b", "a"],
        max_children_per_iteration=5,
    )
    assert sum(allocation) == 5
    assert dict(zip(["f", "e", "d", "c", "b", "a"], allocation)) == {
        "a": 1,
        "b": 1,
        "c": 1,
        "d": 1,
        "e": 1,
        "f": 0,
    }


def test_allocate_frontier_is_invariant_to_input_order_by_node_id():
    nodes = [SearchNode(node_id=node_id, claim=node_id, score=0.5) for node_id in "fedcba"]
    forward = allocate_frontier(
        nodes,
        allocation_mass_per_iteration=3,
        max_children_per_iteration=5,
    )
    reverse = allocate_frontier(
        list(reversed(nodes)),
        allocation_mass_per_iteration=3,
        max_children_per_iteration=5,
    )
    assert {item.node_id: item.expansion_budget for item in forward} == {
        item.node_id: item.expansion_budget for item in reverse
    }
    assert sum(item.expansion_budget for item in forward) <= 5


def test_allocate_frontier():
    nodes = [
        SearchNode(node_id="a", claim="A", score=0.7, uncertainty=0.2),
        SearchNode(node_id="b", claim="B", score=0.4, uncertainty=0.8),
    ]
    result = allocate_frontier(
        nodes,
        allocation_mass_per_iteration=3,
        max_children_per_iteration=5,
    )
    assert len(result) == 2
    assert all(r.expansion_budget >= 0 for r in result)
