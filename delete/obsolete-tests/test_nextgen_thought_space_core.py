import math

import numpy as np
import pytest

from dte_nextgen.thought_space import (
    CANONICALIZATION_VERSION,
    FrozenThoughtMetric,
    MetricIdentity,
    ProspectiveThought,
    adaptive_bandwidth,
    configurational_entropy,
    null_adjusted_geometric_return,
    rbf_mmd2,
)
from dte_nextgen.thought_space.transition import (
    METHOD_EPISTEMIC_TRANSITION_VERSION,
    MethodEpistemicTransition,
    embed_method_epistemic_transitions,
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


def test_q_is_context_only_and_not_embedded() -> None:
    first = MethodEpistemicTransition(
        retrospective_method="rewrite in the shared group",
        epistemic_change_kind="new_understanding",
        epistemic_change="necessity and sufficiency are the same group constraint",
        context_q="why does the condition run backwards?",
    )
    second = MethodEpistemicTransition(
        retrospective_method=first.retrospective_method,
        epistemic_change_kind=first.epistemic_change_kind,
        epistemic_change=first.epistemic_change,
        context_q="a completely different source problem",
    )
    assert METHOD_EPISTEMIC_TRANSITION_VERSION == "method-epistemic-transition-v1"
    assert first.canonical_text() == second.canonical_text()
    assert "why does the condition" not in first.canonical_text()
    assert "different source problem" not in second.canonical_text()


def test_transition_pair_canonicalization_and_embedding_are_stable() -> None:
    transition = MethodEpistemicTransition(
        retrospective_method="  compare   commuting receivers ",
        epistemic_change_kind="sharper_unknown",
        epistemic_change="  geometric asymmetry still gives operational equivalence ",
        context_q="context only",
    )
    assert transition.canonical_text() == (
        "METHOD_EPISTEMIC_TRANSITION_V1\n"
        "RETROSPECTIVE_METHOD:\ncompare commuting receivers\n\n"
        "EPISTEMIC_CHANGE_KIND:\nsharper_unknown\n\n"
        "EPISTEMIC_CHANGE:\ngeometric asymmetry still gives operational equivalence"
    )
    seen: list[str] = []

    def embed_fn(text: str) -> list[float]:
        seen.append(text)
        return [1.0, 2.0]

    embedded = embed_method_epistemic_transitions([transition], embed_fn)
    assert embedded.shape == (1, 2)
    assert seen == [transition.canonical_text()]


def test_transition_pair_rejects_unknown_change_kind() -> None:
    with pytest.raises(ValueError, match="epistemic_change_kind"):
        MethodEpistemicTransition(
            retrospective_method="method",
            epistemic_change_kind="generic_novelty",
            epistemic_change="something changed",
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
