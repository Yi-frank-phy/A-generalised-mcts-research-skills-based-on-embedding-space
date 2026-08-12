import numpy as np

from dte_nextgen.thought_space.return_metric import (
    frontier_after_replacement,
    replacement_frontier_return,
    null_adjusted_geometric_return,
)


def test_used_parent_leaves_frontier_and_child_replaces_it() -> None:
    before = np.asarray([[0.0], [1.0], [2.0]])
    child = np.asarray([10.0])

    after = frontier_after_replacement(before, parent_index=1, child_embedding=child)

    assert after.shape == before.shape
    assert np.allclose(after, [[0.0], [10.0], [2.0]])
    assert not np.any(np.all(np.isclose(after, [1.0]), axis=1))


def test_frontier_replacement_does_not_mutate_before_cloud() -> None:
    before = np.asarray([[0.0], [1.0], [2.0]])
    original = before.copy()

    frontier_after_replacement(before, parent_index=0, child_embedding=np.asarray([5.0]))

    assert np.array_equal(before, original)


def test_replacement_return_is_generic_null_adjusted_mmd_on_replaced_frontier() -> None:
    before = np.asarray([[0.0], [1.0], [2.0]])
    child = np.asarray([4.0])
    null_a = np.asarray([[0.0], [1.0], [2.0]])
    null_b = np.asarray([[0.0], [1.1], [2.0]])
    bandwidth = 0.7

    after = frontier_after_replacement(before, parent_index=1, child_embedding=child)
    expected = null_adjusted_geometric_return(
        before,
        after,
        null_a,
        null_b,
        bandwidth,
    )
    actual = replacement_frontier_return(
        before,
        parent_index=1,
        child_embedding=child,
        null_a=null_a,
        null_b=null_b,
        bandwidth=bandwidth,
    )

    assert np.isclose(actual, expected)


def test_replacement_requires_one_existing_active_parent() -> None:
    before = np.asarray([[0.0], [1.0]])

    try:
        frontier_after_replacement(before, parent_index=2, child_embedding=np.asarray([3.0]))
    except IndexError as exc:
        assert "parent_index" in str(exc)
    else:
        raise AssertionError("out-of-range parent_index must fail")
