import numpy as np
import pytest

from dte_nextgen.thought_space.metric import (
    FrozenTransitionMetric,
    FrozenThoughtMetric,
    MetricIdentity,
)
from dte_nextgen.thought_space.prospective import (
    CANONICALIZATION_VERSION,
    ProspectiveThought,
)
from dte_nextgen.thought_space.transition import (
    METHOD_EPISTEMIC_TRANSITION_VERSION,
    MethodEpistemicTransition,
)


def test_frozen_transition_metric_accepts_transition_serializer_identity() -> None:
    identity = MetricIdentity(
        embedding_provider="test",
        embedding_model="test-model",
        embedding_dimension=2,
        canonicalization_version=METHOD_EPISTEMIC_TRANSITION_VERSION,
        normalization_policy="none",
        kernel_family="rbf",
        bandwidth=0.5,
    )
    metric = FrozenTransitionMetric(identity=identity, embed_fn=lambda _: [1.0, 2.0])
    transitions = [
        MethodEpistemicTransition(
            retrospective_method="group rewrite",
            epistemic_change_kind="new_understanding",
            epistemic_change="iff structure",
            context_q="context must not enter embedding",
        )
    ]

    cloud = metric.embed_cloud(transitions)

    assert np.allclose(cloud, [[1.0, 2.0]])


def test_frozen_transition_metric_rejects_prospective_serializer_identity() -> None:
    identity = MetricIdentity(
        embedding_provider="test",
        embedding_model="test-model",
        embedding_dimension=2,
        canonicalization_version=CANONICALIZATION_VERSION,
        normalization_policy="none",
        kernel_family="rbf",
        bandwidth=0.5,
    )
    with pytest.raises(ValueError, match="transition serializer"):
        FrozenTransitionMetric(identity=identity, embed_fn=lambda _: [1.0, 2.0])


def test_legacy_frozen_thought_metric_remains_compatible() -> None:
    identity = MetricIdentity(
        embedding_provider="test",
        embedding_model="test-model",
        embedding_dimension=2,
        canonicalization_version=CANONICALIZATION_VERSION,
        normalization_policy="none",
        kernel_family="rbf",
        bandwidth=0.5,
    )
    metric = FrozenThoughtMetric(identity=identity, embed_fn=lambda _: [1.0, 2.0])
    cloud = metric.embed_cloud([ProspectiveThought("a", "b", "c")])
    assert np.allclose(cloud, [[1.0, 2.0]])
