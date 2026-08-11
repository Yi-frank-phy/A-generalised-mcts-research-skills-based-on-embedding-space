import math

import numpy as np
import pytest

from dte_backend.thought_space import (
    CANONICALIZATION_VERSION,
    FrozenThoughtMetric,
    MetricIdentity,
    ProspectiveThought,
    adaptive_bandwidth,
    configurational_entropy,
    null_adjusted_geometric_return,
    rbf_mmd2,
)


def test_prospective_thought_canonicalization_is_stable() -> None:
    thought = ProspectiveThought(
        observation="  pair   counts match  ",
        possible_structure=" maybe   a cycle ",
        discriminating_test=" compute   the boundary ",
    )
    assert thought.canonical_text() == (
        "OBSERVATION:\npair counts match\n\n"
        "POSSIBLE_STRUCTURE:\nmaybe a cycle\n\n"
        "DISCRIMINATING_TEST:\ncompute the boundary"
    )


def test_adaptive_soft_count_entropy_is_bounded() -> None:
    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    h = adaptive_bandwidth(points)
    entropy = configurational_entropy(points, h)
    assert h > 0
    assert 0.0 <= entropy <= math.log(len(points))


def test_frozen_metric_rejects_dimension_drift() -> None:
    identity = MetricIdentity(
        embedding_provider="test",
        embedding_model="test-model",
        embedding_dimension=2,
        canonicalization_version=CANONICALIZATION_VERSION,
        normalization_policy="none",
        kernel_family="rbf",
        bandwidth=0.5,
    )
    metric = FrozenThoughtMetric(identity=identity, embed_fn=lambda _: [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="embedding dimension"):
        metric.embed_cloud([ProspectiveThought("a", "b", "c")])


def test_rbf_mmd_is_zero_for_identical_clouds() -> None:
    cloud = np.asarray([[0.0, 0.0], [1.0, 0.0]])
    assert rbf_mmd2(cloud, cloud, bandwidth=0.5) == pytest.approx(0.0)


def test_null_adjusted_geometric_return_is_bounded() -> None:
    before = np.asarray([[0.0], [0.0]])
    after = np.asarray([[10.0], [10.0]])
    null_a = np.asarray([[0.0], [0.0]])
    null_b = np.asarray([[0.0], [0.0]])
    value = null_adjusted_geometric_return(before, after, null_a, null_b, bandwidth=1.0)
    assert 0.0 <= value <= 1.0
