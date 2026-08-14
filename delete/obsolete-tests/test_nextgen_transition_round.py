import numpy as np

from dte_nextgen.thought_space.round import (
    score_transition_directions,
    select_transition_direction,
)
from dte_nextgen.thought_space.transition import MethodEpistemicTransition


def _fake_embed(text: str) -> list[float]:
    if "counterexample search" in text:
        return [10.0]
    return [0.0]


def _transitions() -> list[MethodEpistemicTransition]:
    return [
        MethodEpistemicTransition(
            retrospective_method="group rewrite",
            epistemic_change_kind="new_understanding",
            epistemic_change="iff structure",
            context_q="first source question",
        ),
        MethodEpistemicTransition(
            retrospective_method="group rewrite",
            epistemic_change_kind="new_understanding",
            epistemic_change="iff structure",
            context_q="different source question",
        ),
        MethodEpistemicTransition(
            retrospective_method="counterexample search",
            epistemic_change_kind="sharper_unknown",
            epistemic_change="exception family",
            context_q="third source question",
        ),
    ]


def test_scoreable_geometry_is_completed_transition_pairs() -> None:
    result = score_transition_directions(
        _transitions(),
        propulsion_values=np.asarray([0.4, 0.4, 0.2]),
        embed_fn=_fake_embed,
        bandwidth=0.1,
    )

    assert result["embeddings"].shape == (3, 1)
    assert result["embeddings"][0, 0] == result["embeddings"][1, 0]
    assert np.allclose(
        result["ucb_scores"],
        result["values"] + result["standard_deviations"],
    )
    assert result["standard_deviations"][0] < result["standard_deviations"][2]


def test_context_q_never_changes_scoring_embedding() -> None:
    first = MethodEpistemicTransition(
        retrospective_method="group rewrite",
        epistemic_change_kind="new_understanding",
        epistemic_change="iff structure",
        context_q="Q one",
    )
    second = MethodEpistemicTransition(
        retrospective_method="group rewrite",
        epistemic_change_kind="new_understanding",
        epistemic_change="iff structure",
        context_q="Q two",
    )
    seen: list[str] = []

    def embed_fn(text: str) -> list[float]:
        seen.append(text)
        return [1.0]

    result = score_transition_directions(
        [first, second],
        propulsion_values=np.asarray([0.2, 0.2]),
        embed_fn=embed_fn,
        bandwidth=0.1,
    )

    assert seen[0] == seen[1]
    assert "Q one" not in seen[0]
    assert "Q two" not in seen[1]
    assert np.allclose(result["embeddings"], [[1.0], [1.0]])


def test_select_transition_direction_is_one_action_from_entropy_matched_scheduler() -> None:
    result = select_transition_direction(
        _transitions(),
        propulsion_values=np.asarray([0.7, 0.4, 0.1]),
        embed_fn=_fake_embed,
        selection_quantile=0.2,
        bandwidth=0.1,
    )

    probabilities = result["probabilities"]
    assert np.isclose(np.sum(probabilities), 1.0)
    assert np.all(probabilities >= 0.0)
    assert 0 <= result["selected_index"] < len(_transitions())
    assert result["selected_transition"] == _transitions()[result["selected_index"]]
    assert result["temperature"] > 0.0
