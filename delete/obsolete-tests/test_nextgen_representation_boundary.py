import dte_nextgen.thought_space.round as transition_round
from dte_nextgen.thought_space.prospective import PROSPECTIVE_THOUGHT_ROLE


def test_prospective_thought_is_proposal_only() -> None:
    assert PROSPECTIVE_THOUGHT_ROLE == "intervention_proposal_only"


def test_transition_controller_does_not_depend_on_prospective_state() -> None:
    assert not hasattr(transition_round, "ProspectiveThought")
    assert not hasattr(transition_round, "embed_prospective_thoughts")
