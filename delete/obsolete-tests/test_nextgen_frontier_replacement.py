import numpy as np

from dte_nextgen.thought_space.return_metric import (
    frontier_after_replacement,
    replacement_frontier_return,
    rbf_mmd2,
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


def test_replacement_return_is_direct_normalized_mmd_displacement() -> None:
    before = np.asarray([[0.0], [1.0], [2.0]])
    child = np.asarray([4.0])
    bandwidth = 0.7

    after = frontier_after_replacement(before, parent_index=1, child_embedding=child)
    expected = rbf_mmd2(before, after, bandwidth) / 2.0
    actual = replacement_frontier_return(
        before,
        parent_index=1,
        child_embedding=child,
        bandwidth=bandwidth,
    )

    assert np.isclose(actual, expected)
    assert 0.0 <= actual <= 1.0


def test_small_frontier_jitter_naturally_has_small_value() -> None:
    before = np.asarray([[0.0], [1.0], [2.0]])
    small_jitter = replacement_frontier_return(
        before,
        parent_index=1,
        child_embedding=np.asarray([1.01]),
        bandwidth=0.7,
    )
    large_move = replacement_frontier_return(
        before,
        parent_index=1,
        child_embedding=np.asarray([4.0]),
        bandwidth=0.7,
    )

    assert small_jitter < large_move


def test_replacement_requires_one_existing_active_parent() -> None:
    before = np.asarray([[0.0], [1.0]])

    try:
        frontier_after_replacement(before, parent_index=2, child_embedding=np.asarray([3.0]))
    except IndexError as exc:
        assert "parent_index" in str(exc)
    else:
        raise AssertionError("out-of-range parent_index must fail")
