from dte_backend.continuation import (
    count_committed_search_nodes,
    evaluate_continuation_gate,
    remaining_search_node_slots,
)
from dte_backend.epistemic_models import (
    EpistemicLedgerV1,
    EpistemicStatementRecordV1,
    PathDispositionRecordV1,
)
from dte_backend.models import SearchNode


HASH = "a" * 64


def node(node_id: str, *, status: str = "frontier") -> SearchNode:
    return SearchNode(node_id=node_id, claim=node_id, status=status)


def test_committed_node_budget_includes_merged_and_excludes_synthesis():
    nodes = [
        node("frontier"),
        node("closed", status="closed"),
        node("merged", status="merged"),
        SearchNode(node_id="summary", claim="summary", node_type="synthesis", status="synthesis"),
    ]
    assert count_committed_search_nodes(nodes) == 3
    assert remaining_search_node_slots(nodes, 5) == 2


def test_single_frontier_with_new_allocated_node_continues():
    record = evaluate_continuation_gate(
        iteration=1,
        graph_revision=1,
        nodes=[node("n1")],
        max_committed_search_nodes=20,
        entropy_delta=None,
        consecutive_plateau_count=0,
        plateau_confirmed=False,
        allocations={"n1": 1},
        previous_frontier_node_ids=set(),
        previous_positive_allocation_node_ids=set(),
        previous_provisional_synthesis_node_ids=set(),
        provisional_synthesis_node_ids=["n1"],
        ledger=EpistemicLedgerV1(),
        previously_considered_epistemic_ids=set(),
    )
    assert record.trigger_signals == ["single_canonical_frontier"]
    assert record.decision == "continue"
    assert record.continuation_target_node_ids == ["n1"]


def test_confirmed_plateau_without_new_yield_prepares_synthesis():
    record = evaluate_continuation_gate(
        iteration=3,
        graph_revision=3,
        nodes=[node("n1"), node("n2")],
        max_committed_search_nodes=20,
        entropy_delta=0.0,
        consecutive_plateau_count=2,
        plateau_confirmed=True,
        allocations={"n1": 1, "n2": 0},
        previous_frontier_node_ids={"n1", "n2"},
        previous_positive_allocation_node_ids={"n1"},
        previous_provisional_synthesis_node_ids={"n1"},
        provisional_synthesis_node_ids=["n1"],
        ledger=EpistemicLedgerV1(),
        previously_considered_epistemic_ids=set(),
    )
    assert record.decision == "prepare_synthesis"


def test_basis_backed_counterexample_is_consumed_once_for_continuation():
    disposition = PathDispositionRecordV1(
        disposition_id="disp-1",
        local_id="disp",
        target_node_id="n1",
        epistemic_disposition="counterexample_found",
        source_type="agent_reported",
        basis_refs=["node-claim:n1"],
        explanation="A concrete counterexample changes the branch obligation.",
        run_id="run",
        episode_id="episode",
        attempt_id="attempt",
        role="executor",
        output_hash=HASH,
        committed_at="2026-07-19T00:00:00+00:00",
    )
    ledger = EpistemicLedgerV1(path_dispositions=[disposition])
    kwargs = dict(
        iteration=3,
        graph_revision=3,
        nodes=[node("n1"), node("n2")],
        max_committed_search_nodes=20,
        entropy_delta=0.0,
        consecutive_plateau_count=2,
        plateau_confirmed=True,
        allocations={"n1": 1, "n2": 0},
        previous_frontier_node_ids={"n1", "n2"},
        previous_positive_allocation_node_ids={"n1"},
        previous_provisional_synthesis_node_ids={"n1"},
        provisional_synthesis_node_ids=["n1"],
        ledger=ledger,
    )
    first = evaluate_continuation_gate(
        **kwargs,
        previously_considered_epistemic_ids=set(),
    )
    assert first.decision == "continue"
    assert first.material_epistemic_record_ids == ["disp-1"]

    second = evaluate_continuation_gate(
        **kwargs,
        previously_considered_epistemic_ids={"disp-1"},
    )
    assert second.decision == "prepare_synthesis"


def test_open_question_does_not_count_as_material_yield():
    statement = EpistemicStatementRecordV1(
        statement_id="stmt-1",
        local_id="stmt",
        statement_type="open_question",
        text="Could there be another route?",
        target_node_id="n1",
        source_type="agent_reported",
        basis_refs=["node-claim:n1"],
        run_id="run",
        episode_id="episode",
        attempt_id="attempt",
        role="executor",
        output_hash=HASH,
        committed_at="2026-07-19T00:00:00+00:00",
    )
    record = evaluate_continuation_gate(
        iteration=3,
        graph_revision=3,
        nodes=[node("n1"), node("n2")],
        max_committed_search_nodes=20,
        entropy_delta=0.0,
        consecutive_plateau_count=2,
        plateau_confirmed=True,
        allocations={"n1": 1, "n2": 0},
        previous_frontier_node_ids={"n1", "n2"},
        previous_positive_allocation_node_ids={"n1"},
        previous_provisional_synthesis_node_ids={"n1"},
        provisional_synthesis_node_ids=["n1"],
        ledger=EpistemicLedgerV1(statements=[statement]),
        previously_considered_epistemic_ids=set(),
    )
    assert record.material_epistemic_record_ids == []
    assert record.decision == "prepare_synthesis"
