from dte_backend.math_engine import (
    allocate_frontier,
    boltzmann_allocation,
    calculate_ucb,
    discretize_allocation,
)
from dte_backend.models import SearchNode


def test_ucb_is_exact_value_plus_sd_and_ignores_temperature_controls():
    assert calculate_ucb(score=0.6, uncertainty=0.25) == 0.85
    assert calculate_ucb(score=0.6, uncertainty=0.25, tau=0.0, c_explore=0.0) == 0.85
    assert calculate_ucb(score=0.6, uncertainty=0.25, tau=99.0, c_explore=99.0) == 0.85


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


def test_zero_temperature_boltzmann_concentrates_on_unique_maximum():
    assert boltzmann_allocation(
        [0.2, 0.9, 0.4],
        allocation_mass_per_iteration=3,
        max_children_per_iteration=5,
        node_ids=["a", "b", "c"],
        temperature=0.0,
    ) == [0, 3, 0]


def test_zero_temperature_tied_maxima_are_symmetric():
    assert boltzmann_allocation(
        [0.9, 0.9, 0.2],
        allocation_mass_per_iteration=2,
        max_children_per_iteration=5,
        node_ids=["a", "b", "c"],
        temperature=0.0,
    ) == [1, 1, 0]


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


def test_higher_sd_changes_actual_ucb_allocation_support():
    nodes = [
        SearchNode(node_id="a", claim="A", score=0.5, uncertainty=0.1),
        SearchNode(node_id="b", claim="B", score=0.5, uncertainty=0.4),
    ]
    result = allocate_frontier(
        nodes,
        allocation_mass_per_iteration=4,
        max_children_per_iteration=5,
        temperature=0.25,
    )
    by_id = {item.node_id: item for item in result}
    assert by_id["b"].ucb_score > by_id["a"].ucb_score
    assert by_id["b"].expansion_budget >= by_id["a"].expansion_budget


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
