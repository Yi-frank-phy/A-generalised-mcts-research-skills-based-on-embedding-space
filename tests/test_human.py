from dte_backend.human import maybe_create_human_question
from dte_backend.models import SearchNode


def test_human_question_on_plateau_tie():
    a = SearchNode(node_id="a", claim="A", ucb_score=0.5)
    b = SearchNode(node_id="b", claim="B", ucb_score=0.51)
    q = maybe_create_human_question([a, b], entropy_plateau=True)
    assert q is not None
    assert q.question_type == "branch_choice"


def test_no_human_question_when_not_plateau():
    a = SearchNode(node_id="a", claim="A", ucb_score=0.5)
    b = SearchNode(node_id="b", claim="B", ucb_score=0.51)
    assert maybe_create_human_question([a, b], entropy_plateau=False) is None
