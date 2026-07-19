from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

import dte_backend.app_driver as app_driver
from dte_backend.app_driver import (
    AppRunState,
    app_run_status,
    create_app_run,
    fail_app_episode,
    next_app_episode,
    request_app_synthesis,
    retry_app_episode,
    submit_app_episode_result,
)
from dte_backend.embedding import HashEmbeddingProvider
from dte_backend.epistemic import (
    DuplicateLearningError,
    build_terminal_epistemic_handoff,
    read_researcher_learning_ledger,
    record_researcher_learning,
    render_epistemic_text,
)
from dte_backend.epistemic_commit import EpistemicReferenceContext
from dte_backend.epistemic_models import (
    EpistemicContributionBundle,
    EpistemicDependencyGraphV1,
    EpistemicEdgeContribution,
    EpistemicLedgerV1,
    EpistemicStatementContribution,
    PathDispositionContribution,
    ResearcherLearningRecordV1,
    TerminalEpistemicHandoffV1,
)
from dte_backend.episode_adapter import (
    build_executor_episode_request,
    build_judge_episode_request,
)
from dte_backend.episode_commit import EpisodeGraph, commit_episode_result
from dte_backend.episode_models import (
    EpisodeResult,
    ExecutorEpisodeOutput,
    ExecutorNodeCandidate,
    JudgeEpisodeOutput,
    JudgeObservation,
    RuntimeDiagnostics,
    compute_output_hash,
)
from dte_backend.models import (
    BudgetSpec,
    DTERunSpec,
    SearchNode,
    SynthesisControlRequest,
)
from dte_backend.relation_models import RelationEpisodeOutput, RelationObservation


def spec(*, node_cap: int = 2, max_iterations: int = 1) -> DTERunSpec:
    return DTERunSpec(
        problem="trace epistemic provenance",
        goal="preserve claim, assumption, evidence, and challenge dependencies",
        constraints=["do not infer edges from free text"],
        budget=BudgetSpec(
            max_iterations=max_iterations,
            allocation_mass_per_iteration=1,
            max_children_per_iteration=node_cap,
            max_relation_pairs_per_episode=node_cap,
            max_relation_enrichment_pairs=0,
            min_iterations_before_synthesis=2,
        ),
        embedding_provider="hash",
        embedding_dimension=8,
    )


def diagnostics(*, model: str | None = None, runtime_profile: str | None = None):
    return RuntimeDiagnostics(
        adapter_name="codex-app-main-agent",
        transport_name="current-app-runtime",
        profile="native-autonomous",
        runtime_profile=runtime_profile,
        model=model,
        usage_source="unavailable",
        diagnostics_source="unavailable",
    )


def result_for(request, output, *, model=None, runtime_profile=None) -> EpisodeResult:
    return EpisodeResult(
        episode_id=request.episode_id,
        attempt_id=request.attempt_id,
        run_id=request.run_id,
        role=request.role,
        input_graph_revision=request.input_graph_revision,
        selected_node_revisions=request.selected_node_revisions,
        status="completed",
        structured_output=output,
        runtime_diagnostics=diagnostics(model=model, runtime_profile=runtime_profile),
        output_hash=compute_output_hash(output, request.output_schema_version),
        schema_version=request.output_schema_version,
    )


def assumption_bundle(node_id: str = "parent") -> EpistemicContributionBundle:
    return EpistemicContributionBundle(
        statements=[
            EpistemicStatementContribution(
                local_id="assumption-1",
                statement_type="assumption",
                text="the regularity condition holds",
                target_node_id=node_id,
                source_type="agent_reported",
                basis_refs=[],
            )
        ],
        edges=[
            EpistemicEdgeContribution(
                local_id="requires-1",
                source_ref=f"node-claim:{node_id}",
                target_ref="local-statement:assumption-1",
                relation_type="requires",
                source_type="agent_reported",
                basis_refs=[],
                explanation="the claim is conditional on regularity",
            )
        ],
    )


def direct_executor_request(graph: EpisodeGraph, *, grant: int = 1):
    return build_executor_episode_request(
        graph,
        graph.node_by_id("parent"),
        run_id="run-epistemic",
        iteration=1,
        max_returned_children=grant,
        objective="expand one parent",
    )


def direct_judge_request(graph: EpisodeGraph):
    return build_judge_episode_request(
        graph,
        [graph.node_by_id("parent")],
        run_id="run-epistemic",
        problem="p",
        goal="g",
    )


def graph_snapshot(graph: EpisodeGraph):
    return graph.snapshot()


def test_executor_commits_structured_epistemic_contributions_atomically():
    graph = EpisodeGraph(nodes=[SearchNode(node_id="parent", claim="parent claim")])
    request = direct_executor_request(graph)
    output = ExecutorEpisodeOutput(
        nodes=[
            ExecutorNodeCandidate(
                node_id="child",
                claim="child claim",
                parent_ids=["parent"],
            )
        ],
        epistemic_contributions=EpistemicContributionBundle(
            statements=[
                EpistemicStatementContribution(
                    local_id="evidence-1",
                    statement_type="evidence",
                    text="a bounded calculation supports the child",
                    target_node_id="child",
                    source_type="agent_reported",
                    basis_refs=[],
                )
            ],
            edges=[
                EpistemicEdgeContribution(
                    local_id="supports-1",
                    source_ref="local-statement:evidence-1",
                    target_ref="node-claim:child",
                    relation_type="supports",
                    source_type="agent_reported",
                    basis_refs=[],
                    explanation="the calculation bears directly on the child claim",
                )
            ],
        ),
    )
    outcome = commit_episode_result(graph, request, result_for(request, output))

    assert outcome.accepted is True
    assert len(graph.epistemic_ledger.statements) == 1
    assert len(graph.epistemic_ledger.edges) == 1
    statement = graph.epistemic_ledger.statements[0]
    edge = graph.epistemic_ledger.edges[0]
    assert statement.target_node_id == "child"
    assert edge.source_ref == f"epistemic:{statement.statement_id}"
    assert edge.target_ref == "node-claim:child"
    assert statement.episode_id == request.episode_id
    assert statement.attempt_id == request.attempt_id


def test_judge_commits_epistemic_contributions_without_creating_nodes():
    graph = EpisodeGraph(nodes=[SearchNode(node_id="parent", claim="parent claim")])
    request = direct_judge_request(graph)
    output = JudgeEpisodeOutput(
        observations=[
            JudgeObservation(
                node_id="parent",
                score=0.7,
                reasoning="promising but conditional",
                risks=["regularity is unverified"],
            )
        ],
        epistemic_contributions=assumption_bundle(),
    )
    outcome = commit_episode_result(graph, request, result_for(request, output))

    assert outcome.accepted is True
    assert [node.node_id for node in graph.nodes] == ["parent"]
    assert graph.epistemic_ledger.statements[0].role == "judge"
    assert graph.epistemic_ledger.edges[0].relation_type == "requires"


@pytest.mark.parametrize("role", ["executor", "judge"])
def test_legacy_episode_output_without_epistemic_field_remains_compatible(role):
    graph = EpisodeGraph(nodes=[SearchNode(node_id="parent", claim="parent claim")])
    if role == "executor":
        request = direct_executor_request(graph, grant=0)
        output = ExecutorEpisodeOutput(nodes=[])
    else:
        request = direct_judge_request(graph)
        output = JudgeEpisodeOutput(
            observations=[
                JudgeObservation(
                    node_id="parent",
                    score=0.5,
                    reasoning="legacy observation",
                    risks=[],
                )
            ]
        )
    outcome = commit_episode_result(graph, request, result_for(request, output))
    assert outcome.accepted is True
    assert graph.epistemic_ledger == EpistemicLedgerV1()


def test_cross_episode_committed_epistemic_reference_resolves():
    graph = EpisodeGraph(nodes=[SearchNode(node_id="parent", claim="parent claim")])
    judge_request = direct_judge_request(graph)
    judge_output = JudgeEpisodeOutput(
        observations=[
            JudgeObservation(
                node_id="parent", score=0.6, reasoning="conditional", risks=[]
            )
        ],
        epistemic_contributions=assumption_bundle(),
    )
    assert commit_episode_result(
        graph, judge_request, result_for(judge_request, judge_output)
    ).accepted
    statement_id = graph.epistemic_ledger.statements[0].statement_id

    executor_request = direct_executor_request(graph, grant=0)
    executor_output = ExecutorEpisodeOutput(
        nodes=[],
        epistemic_contributions=EpistemicContributionBundle(
            edges=[
                EpistemicEdgeContribution(
                    local_id="qualifies-prior",
                    source_ref=f"epistemic:{statement_id}",
                    target_ref="node-claim:parent",
                    relation_type="qualifies",
                    source_type="agent_reported",
                    basis_refs=[f"epistemic:{statement_id}"],
                    explanation="the prior assumption bounds the claim",
                )
            ]
        ),
    )
    assert commit_episode_result(
        graph, executor_request, result_for(executor_request, executor_output)
    ).accepted
    assert graph.epistemic_ledger.edges[-1].source_ref == f"epistemic:{statement_id}"


@pytest.mark.parametrize(
    "bad_ref",
    [
        "node-claim:missing",
        "episode-result:missing-episode:missing-attempt",
        "relation:missing-relation",
        "merge:missing-merge",
        "epistemic:missing-record",
        "artifact:missing/proof.json",
    ],
)
def test_unknown_epistemic_reference_rejects_the_whole_commit(bad_ref):
    graph = EpisodeGraph(nodes=[SearchNode(node_id="parent", claim="parent claim")])
    request = direct_executor_request(graph, grant=0)
    output = ExecutorEpisodeOutput(
        nodes=[],
        epistemic_contributions=EpistemicContributionBundle(
            statements=[
                EpistemicStatementContribution(
                    local_id="bad-basis",
                    statement_type="evidence",
                    text="unsupported reference",
                    target_node_id="parent",
                    source_type="agent_reported",
                    basis_refs=[bad_ref],
                )
            ]
        ),
    )
    before = graph_snapshot(graph)
    outcome = commit_episode_result(
        graph,
        request,
        result_for(request, output),
        epistemic_context=EpistemicReferenceContext(),
    )
    assert outcome.accepted is False
    assert "epistemic reference" in (outcome.rejection_reason or "")
    assert graph_snapshot(graph) == before


def test_safe_existing_artifact_and_explicit_external_reference_are_accepted(tmp_path):
    artifact = tmp_path / "proof.json"
    artifact.write_text("{}", encoding="utf-8")
    graph = EpisodeGraph(nodes=[SearchNode(node_id="parent", claim="parent claim")])
    request = direct_executor_request(graph, grant=0)
    output = ExecutorEpisodeOutput(
        nodes=[],
        epistemic_contributions=EpistemicContributionBundle(
            statements=[
                EpistemicStatementContribution(
                    local_id="artifact-evidence",
                    statement_type="evidence",
                    text="the calculation artifact supports the claim",
                    target_node_id="parent",
                    source_type="external_artifact_backed",
                    basis_refs=["artifact:proof.json", "external:doi:10.1000/example"],
                )
            ]
        ),
    )
    context = EpistemicReferenceContext(artifact_paths={"proof.json"})
    outcome = commit_episode_result(
        graph, request, result_for(request, output), epistemic_context=context
    )
    assert outcome.accepted is True
    assert graph.epistemic_ledger.statements[0].source_type == (
        "external_artifact_backed"
    )


@pytest.mark.parametrize("source_type", ["human_confirmed", "backend_derived"])
def test_agent_episode_cannot_forge_human_or_backend_source(source_type):
    graph = EpisodeGraph(nodes=[SearchNode(node_id="parent", claim="parent claim")])
    request = direct_executor_request(graph, grant=0)
    output = ExecutorEpisodeOutput(
        nodes=[],
        epistemic_contributions=EpistemicContributionBundle(
            statements=[
                EpistemicStatementContribution(
                    local_id="forged",
                    statement_type="assumption",
                    text="forged authority",
                    target_node_id="parent",
                    source_type=source_type,
                    basis_refs=[],
                )
            ]
        ),
    )
    before = graph_snapshot(graph)
    outcome = commit_episode_result(graph, request, result_for(request, output))
    assert outcome.accepted is False
    assert "source_type" in (outcome.rejection_reason or "")
    assert graph_snapshot(graph) == before


def test_illegal_source_type_is_rejected_by_strict_schema():
    raw = assumption_bundle().model_dump(mode="json")
    raw["statements"][0]["source_type"] = "agent_verified"
    with pytest.raises(ValidationError, match="literal_error"):
        EpistemicContributionBundle.model_validate(raw)


def test_epistemic_contribution_hard_cap_is_enforced():
    with pytest.raises(ValidationError, match="too_long"):
        EpistemicContributionBundle(
            statements=[
                EpistemicStatementContribution(
                    local_id=f"statement-{index}",
                    statement_type="assumption",
                    text=f"bounded assumption {index}",
                    target_node_id="parent",
                    source_type="agent_reported",
                    basis_refs=[],
                )
                for index in range(25)
            ]
        )


def test_epistemic_basis_reference_length_is_bounded():
    with pytest.raises(ValidationError, match="at most 1024 characters"):
        EpistemicStatementContribution(
            local_id="oversized-ref",
            statement_type="evidence",
            text="bounded evidence",
            target_node_id="parent",
            source_type="agent_reported",
            basis_refs=[f"external:{'x' * 1024}"],
        )


@pytest.mark.parametrize("role", ["executor", "judge"])
def test_episode_cannot_target_an_existing_but_ungranted_node(role):
    graph = EpisodeGraph(
        nodes=[
            SearchNode(node_id="parent", claim="parent claim"),
            SearchNode(node_id="ungranted", claim="ungranted claim"),
        ]
    )
    bundle = EpistemicContributionBundle(
        statements=[
            EpistemicStatementContribution(
                local_id="out-of-scope",
                statement_type="assumption",
                text="an assumption about an ungranted node",
                target_node_id="ungranted",
                source_type="agent_reported",
                basis_refs=[],
            )
        ]
    )
    if role == "executor":
        request = direct_executor_request(graph, grant=0)
        output = ExecutorEpisodeOutput(
            nodes=[], epistemic_contributions=bundle
        )
    else:
        request = direct_judge_request(graph)
        output = JudgeEpisodeOutput(
            observations=[
                JudgeObservation(
                    node_id="parent",
                    score=0.5,
                    reasoning="only the granted node was judged",
                    risks=[],
                )
            ],
            epistemic_contributions=bundle,
        )
    before = graph_snapshot(graph)
    outcome = commit_episode_result(graph, request, result_for(request, output))

    assert outcome.accepted is False
    assert "target exceeds episode authority" in (outcome.rejection_reason or "")
    assert graph_snapshot(graph) == before


def test_external_artifact_backed_requires_an_external_or_artifact_basis():
    graph = EpisodeGraph(nodes=[SearchNode(node_id="parent", claim="parent claim")])
    request = direct_executor_request(graph, grant=0)
    output = ExecutorEpisodeOutput(
        nodes=[],
        epistemic_contributions=EpistemicContributionBundle(
            statements=[
                EpistemicStatementContribution(
                    local_id="unbacked",
                    statement_type="evidence",
                    text="claims external verification without a source",
                    target_node_id="parent",
                    source_type="external_artifact_backed",
                    basis_refs=[],
                )
            ]
        ),
    )
    outcome = commit_episode_result(graph, request, result_for(request, output))
    assert outcome.accepted is False
    assert "external_artifact_backed" in (outcome.rejection_reason or "")


@pytest.mark.parametrize("disposition", ["counterexample_found", "contradicted"])
def test_strong_negative_disposition_requires_basis(disposition):
    with pytest.raises(ValidationError, match="basis_refs"):
        PathDispositionContribution(
            local_id="negative",
            target_node_id="parent",
            epistemic_disposition=disposition,
            source_type="agent_reported",
            basis_refs=[],
            explanation="too strong without a basis",
        )


def test_stale_rejected_commit_adds_no_epistemic_records():
    graph = EpisodeGraph(nodes=[SearchNode(node_id="parent", claim="parent claim")])
    request = direct_executor_request(graph, grant=0)
    output = ExecutorEpisodeOutput(
        nodes=[], epistemic_contributions=assumption_bundle()
    )
    graph.revision += 1
    before = graph_snapshot(graph)
    outcome = commit_episode_result(graph, request, result_for(request, output))
    assert outcome.accepted is False
    assert graph_snapshot(graph) == before
    assert graph.epistemic_ledger == EpistemicLedgerV1()


def test_duplicate_stable_id_is_rejected_atomically():
    first_graph = EpisodeGraph(
        nodes=[SearchNode(node_id="parent", claim="parent claim")]
    )
    request = direct_executor_request(first_graph, grant=0)
    output = ExecutorEpisodeOutput(
        nodes=[], epistemic_contributions=assumption_bundle()
    )
    result = result_for(request, output)
    assert commit_episode_result(first_graph, request, result).accepted

    replay_graph = EpisodeGraph(
        nodes=[SearchNode(node_id="parent", claim="parent claim")],
        epistemic_ledger=first_graph.epistemic_ledger.model_copy(deep=True),
    )
    before = graph_snapshot(replay_graph)
    outcome = commit_episode_result(replay_graph, request, result)
    assert outcome.accepted is False
    assert "duplicate epistemic stable ID" in (outcome.rejection_reason or "")
    assert graph_snapshot(replay_graph) == before


def test_retry_only_commits_the_final_attempt_epistemic_records(tmp_path):
    run_dir = tmp_path / "retry"
    create_app_run(
        run_dir,
        spec(),
        [SearchNode(node_id="parent", claim="parent claim")],
        run_id="retry-run",
    )
    first = next_app_episode(run_dir).request
    assert first.role == "judge"
    fail_app_episode(
        run_dir, first.episode_id, first.attempt_id, "runtime unavailable"
    )
    retried = retry_app_episode(run_dir, first.episode_id).request
    output = JudgeEpisodeOutput(
        observations=[
            JudgeObservation(
                node_id="parent", score=0.8, reasoning="final attempt", risks=[]
            )
        ],
        epistemic_contributions=assumption_bundle(),
    )
    submit = submit_app_episode_result(run_dir, result_for(retried, output))
    assert submit.commit_outcome.accepted is True
    state = app_run_status(run_dir)
    assert len(state.epistemic_ledger.statements) == 1
    assert state.epistemic_ledger.statements[0].attempt_id == retried.attempt_id
    assert state.epistemic_ledger.statements[0].attempt_id != first.attempt_id


def test_late_superseded_attempt_cannot_write_epistemic_ledger(tmp_path):
    run_dir = tmp_path / "late"
    create_app_run(
        run_dir,
        spec(),
        [SearchNode(node_id="parent", claim="parent claim")],
        run_id="late-run",
    )
    first = next_app_episode(run_dir).request
    retry = retry_app_episode(run_dir, first.episode_id).request
    output = JudgeEpisodeOutput(
        observations=[
            JudgeObservation(
                node_id="parent", score=0.8, reasoning="late", risks=[]
            )
        ],
        epistemic_contributions=assumption_bundle(),
    )
    rejected = submit_app_episode_result(run_dir, result_for(first, output))
    assert rejected.commit_outcome.accepted is False
    assert app_run_status(run_dir).epistemic_ledger == EpistemicLedgerV1()
    final_output = output.model_copy(deep=True)
    assert submit_app_episode_result(
        run_dir, result_for(retry, final_output)
    ).commit_outcome.accepted


def test_app_run_state_legacy_migration_defaults_to_empty_epistemic_ledger(tmp_path):
    run_dir = tmp_path / "legacy"
    create_app_run(
        run_dir,
        spec(),
        [SearchNode(node_id="parent", claim="parent claim")],
        run_id="legacy-run",
    )
    path = run_dir / "app_run_state.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw.pop("epistemic_ledger")
    path.write_text(json.dumps(raw), encoding="utf-8")

    state = app_driver.load_app_run(run_dir)
    assert state.epistemic_ledger == EpistemicLedgerV1()
    assert build_terminal_epistemic_handoff(
        run_dir
    ).data_quality.epistemic_data_status == "unavailable"


def test_persisted_epistemic_payload_tampering_fails_restart_validation(tmp_path):
    run_dir = drive_terminal_run(tmp_path)
    path = run_dir / "app_run_state.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["epistemic_ledger"]["statements"][0]["text"] = "tampered text"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="epistemic statement disagrees"):
        app_run_status(run_dir)


def file_snapshot(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def drive_terminal_run(
    tmp_path: Path,
    *,
    external_support: bool = False,
    model: str | None = None,
) -> Path:
    run_dir = tmp_path / ("external" if external_support else "agent")
    create_app_run(
        run_dir,
        spec(node_cap=1),
        [SearchNode(node_id="parent", claim="the bound holds")],
        run_id=run_dir.name,
    )
    judge = next_app_episode(run_dir).request
    judge_output = JudgeEpisodeOutput(
        observations=[
            JudgeObservation(
                node_id="parent",
                score=0.8,
                reasoning="the route is useful but conditional",
                risks=["regularity remains unverified"],
            )
        ],
        epistemic_contributions=EpistemicContributionBundle(
            statements=[
                EpistemicStatementContribution(
                    local_id="regularity",
                    statement_type="assumption",
                    text="the regularity condition holds",
                    target_node_id="parent",
                    source_type="agent_reported",
                    basis_refs=[],
                ),
                EpistemicStatementContribution(
                    local_id="open-boundary",
                    statement_type="open_question",
                    text="does the boundary case preserve the bound?",
                    target_node_id="parent",
                    source_type="agent_reported",
                    basis_refs=[],
                ),
            ],
            edges=[
                EpistemicEdgeContribution(
                    local_id="requires-regularity",
                    source_ref="node-claim:parent",
                    target_ref="local-statement:regularity",
                    relation_type="requires",
                    source_type="agent_reported",
                    basis_refs=[],
                    explanation="regularity is necessary",
                ),
                EpistemicEdgeContribution(
                    local_id="requires-boundary-answer",
                    source_ref="node-claim:parent",
                    target_ref="local-statement:open-boundary",
                    relation_type="requires",
                    source_type="agent_reported",
                    basis_refs=[],
                    explanation="the boundary case remains open",
                ),
            ],
            path_dispositions=[
                PathDispositionContribution(
                    local_id="conditional-path",
                    target_node_id="parent",
                    epistemic_disposition="blocked_by_assumption",
                    source_type="agent_reported",
                    basis_refs=["local-statement:regularity"],
                    explanation="progress is conditional on regularity",
                )
            ],
        ),
    )
    assert submit_app_episode_result(
        run_dir,
        result_for(judge, judge_output, model=model, runtime_profile="high"),
    ).commit_outcome.accepted

    executor = next_app_episode(
        run_dir, embedding_provider=HashEmbeddingProvider(dim=8)
    ).request
    assert executor.role == "executor"
    basis_refs = []
    source_type = "agent_reported"
    if external_support:
        artifact = run_dir / "evidence" / "calculation.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("{}", encoding="utf-8")
        basis_refs = ["artifact:evidence/calculation.json"]
        source_type = "external_artifact_backed"
    executor_output = ExecutorEpisodeOutput(
        nodes=[],
        epistemic_contributions=EpistemicContributionBundle(
            statements=[
                EpistemicStatementContribution(
                    local_id="calculation",
                    statement_type="evidence",
                    text="the checked finite case satisfies the bound",
                    target_node_id="parent",
                    source_type=source_type,
                    basis_refs=basis_refs,
                ),
                EpistemicStatementContribution(
                    local_id="heuristic",
                    statement_type="heuristic",
                    text="separate generic and boundary cases early",
                    target_node_id="parent",
                    source_type="agent_reported",
                    basis_refs=["local-statement:calculation"],
                ),
            ],
            edges=[
                EpistemicEdgeContribution(
                    local_id="calculation-supports",
                    source_ref="local-statement:calculation",
                    target_ref="node-claim:parent",
                    relation_type="supports",
                    source_type=source_type,
                    basis_refs=basis_refs,
                    explanation="the finite calculation supplies structured support",
                )
            ],
        ),
    )
    assert submit_app_episode_result(
        run_dir,
        result_for(executor, executor_output, model=model, runtime_profile="high"),
    ).commit_outcome.accepted
    while True:
        outcome = next_app_episode(run_dir)
        if outcome.request is None:
            assert outcome.controller_action == "ready_for_synthesis"
            break
        if outcome.request.role == "executor":
            assert submit_app_episode_result(
                run_dir, result_for(outcome.request, ExecutorEpisodeOutput(nodes=[]))
            ).commit_outcome.accepted
            continue
        assert outcome.request.role == "relation"
        assert submit_app_episode_result(
            run_dir, relation_result(outcome.request, "independent")
        ).commit_outcome.accepted
    return run_dir


def test_terminal_handoff_is_deterministic_read_only_and_schema_round_trips(tmp_path):
    run_dir = drive_terminal_run(tmp_path)
    before = file_snapshot(run_dir)
    first = build_terminal_epistemic_handoff(run_dir)
    second = build_terminal_epistemic_handoff(run_dir)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert TerminalEpistemicHandoffV1.model_validate_json(
        first.model_dump_json()
    ).model_dump(mode="json") == first.model_dump(mode="json")
    assert EpistemicDependencyGraphV1.model_validate(
        first.dependency_graph.model_dump(mode="json")
    ) == first.dependency_graph
    assert file_snapshot(run_dir) == before
    assert render_epistemic_text(first)
    assert file_snapshot(run_dir) == before


def test_selected_claim_dependency_traversal_and_unresolved_assumptions(tmp_path):
    handoff = build_terminal_epistemic_handoff(drive_terminal_run(tmp_path))
    selected = handoff.selected_claims[0]
    statements = {
        item.statement_id: item for item in handoff.dependency_graph.statements
    }

    assert selected.claim_ref == "node-claim:parent"
    assert selected.required_assumption_refs
    assert selected.unresolved_dependency_refs
    unresolved_text = {
        statements[ref.removeprefix("epistemic:")].text
        for ref in selected.unresolved_dependency_refs
        if ref.startswith("epistemic:")
    }
    assert "the regularity condition holds" in unresolved_text
    assert "does the boundary case preserve the bound?" in unresolved_text
    assert selected.supporting_record_refs
    assert selected.producing_episode_ids
    assert selected.producing_attempt_ids


def test_external_and_agent_reported_support_are_counted_separately(tmp_path):
    agent = build_terminal_epistemic_handoff(
        drive_terminal_run(tmp_path / "one", external_support=False)
    )
    external = build_terminal_epistemic_handoff(
        drive_terminal_run(tmp_path / "two", external_support=True)
    )

    assert agent.independence_summary.agent_only_supported_selected_claim_count == 1
    assert agent.independence_summary.external_artifact_backed_selected_claim_count == 0
    assert external.independence_summary.agent_only_supported_selected_claim_count == 0
    assert external.independence_summary.external_artifact_backed_selected_claim_count == 1
    assert external.selected_claims[0].referenced_artifacts == [
        "evidence/calculation.json"
    ]


def test_missing_model_metadata_is_unavailable_not_guessed(tmp_path):
    handoff = build_terminal_epistemic_handoff(drive_terminal_run(tmp_path))
    independence = handoff.independence_summary
    assert independence.model_metadata_status == "unavailable"
    assert independence.same_model_cross_role_count is None
    assert independence.different_model_cross_role_count is None
    assert independence.same_model_support_challenge_count is None


def test_same_model_cross_role_is_only_a_correlated_error_risk_indicator(tmp_path):
    handoff = build_terminal_epistemic_handoff(
        drive_terminal_run(tmp_path, model="gpt-5.6-sol")
    )
    independence = handoff.independence_summary
    assert independence.model_metadata_status == "available"
    assert independence.same_model_cross_role_count == 1
    assert independence.runtime_profile_metadata_status == "available"
    assert independence.same_runtime_profile_cross_role_count == 1
    assert "same_model_cross_role_confirmation" in independence.risk_flags
    serialized = handoff.model_dump_json().casefold()
    assert "correctness score" not in serialized
    assert "scientific reliability score" not in serialized


def drive_targeted_selection_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "targeted"
    create_app_run(
        run_dir,
        spec(node_cap=2),
        [
            SearchNode(node_id="selected", claim="selected claim"),
            SearchNode(node_id="other", claim="low-score alternative"),
        ],
        run_id="targeted-run",
    )
    judge = next_app_episode(run_dir).request
    output = JudgeEpisodeOutput(
        observations=[
            JudgeObservation(
                node_id=node_id,
                score=0.9 if node_id == "selected" else 0.05,
                reasoning="bounded research-potential judgment",
                risks=[],
            )
            for node_id in judge.selected_node_revisions
        ]
    )
    assert submit_app_episode_result(run_dir, result_for(judge, output)).commit_outcome.accepted
    request_app_synthesis(
        run_dir,
        SynthesisControlRequest(
            action="force_synthesis_after_current_task",
            requested_by="main_agent",
            reason="test targeted handoff",
            scope="node_ids",
            node_ids=["selected"],
        ),
    )
    while True:
        outcome = next_app_episode(run_dir)
        if outcome.request is None:
            assert outcome.controller_action == "ready_for_synthesis"
            break
        if outcome.request.role == "executor":
            assert submit_app_episode_result(
                run_dir, result_for(outcome.request, ExecutorEpisodeOutput(nodes=[]))
            ).commit_outcome.accepted
            continue
        assert outcome.request.role == "relation"
        assert submit_app_episode_result(
            run_dir, relation_result(outcome.request, "independent")
        ).commit_outcome.accepted
    return run_dir


def test_search_and_epistemic_dispositions_are_strictly_separate(tmp_path):
    handoff = build_terminal_epistemic_handoff(drive_targeted_selection_run(tmp_path))
    by_id = {item.node_id: item for item in handoff.node_summaries}
    other = by_id["other"]

    assert "not_selected" in other.search_dispositions
    assert "contradicted" not in other.epistemic_dispositions
    assert "false" not in render_epistemic_text(handoff).casefold()


def relation_result(request, relation_type: str) -> EpisodeResult:
    observations = []
    for pair in request.relation_payload.candidate_pairs:
        values = dict(
            candidate_id=pair.candidate_id,
            left_node_id=pair.left.node_id,
            right_node_id=pair.right.node_id,
            relation_type=relation_type,
            confidence=0.8,
            rationale=f"agent classified the pair as {relation_type}",
            evidence_refs=(
                [pair.left.evidence[0].evidence_ref] if pair.left.evidence else []
            ),
            materiality_assessment="material",
        )
        if relation_type == "conflict":
            values.update(
                conflict_summary="the claims conflict",
                disclosure_required=True,
                conflicting_claims=[pair.left.claim, pair.right.claim],
            )
        elif relation_type == "equivalent":
            values.update(
                merge_recommended=True,
                canonicality_factors=["same normalized claim"],
            )
        observations.append(RelationObservation(**values))
    output = RelationEpisodeOutput(observations=observations)
    return result_for(request, output)


def drive_conflict_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "conflict"
    create_app_run(
        run_dir,
        spec(node_cap=2),
        [
            SearchNode(
                node_id="left",
                claim="the condition is sufficient",
                evidence=["shared-source"],
            ),
            SearchNode(
                node_id="right",
                claim="the condition is not sufficient",
                evidence=["shared-source"],
            ),
        ],
        run_id="conflict-run",
    )
    judge = next_app_episode(run_dir).request
    judge_output = JudgeEpisodeOutput(
        observations=[
            JudgeObservation(
                node_id=node_id, score=0.8, reasoning="material route", risks=[]
            )
            for node_id in judge.selected_node_revisions
        ]
    )
    submit_app_episode_result(run_dir, result_for(judge, judge_output))
    while True:
        outcome = next_app_episode(
            run_dir, embedding_provider=HashEmbeddingProvider(dim=8)
        )
        if outcome.request is None:
            assert outcome.controller_action == "ready_for_synthesis"
            break
        if outcome.request.role == "executor":
            submit_app_episode_result(
                run_dir,
                result_for(outcome.request, ExecutorEpisodeOutput(nodes=[])),
            )
            continue
        assert outcome.request.role == "relation"
        submit_app_episode_result(
            run_dir, relation_result(outcome.request, "conflict")
        )
    return run_dir


def test_relation_conflict_is_projected_from_existing_ledger_without_duplication(tmp_path):
    run_dir = drive_conflict_run(tmp_path)
    state = app_run_status(run_dir)
    handoff = build_terminal_epistemic_handoff(run_dir)

    assert [item.relation_record_id for item in handoff.material_conflicts] == [
        item.relation_record_id for item in state.relation_ledger
    ]
    assert handoff.dependency_graph.relation_projections[0].source_type == (
        "agent_reported"
    )
    assert state.epistemic_ledger.statements == []
    assert state.epistemic_ledger.edges == []


def test_learning_append_validation_and_controller_isolation(tmp_path):
    run_dir = drive_terminal_run(tmp_path)
    before = (run_dir / "app_run_state.json").read_bytes()
    record = record_researcher_learning(
        run_dir,
        source="main_agent",
        previous_view="the condition looked sufficient",
        updated_view="the condition may be only necessary",
        change_reason_refs=["node-claim:parent"],
        reusable_heuristic="separate necessary from sufficient conditions",
        recognized_failure_mode="prematurely collapsing logical directions",
        still_unclear="whether the boundary case closes the gap",
        learning_id="main-agent-learning",
    )
    records, diagnostics_record = read_researcher_learning_ledger(run_dir)

    assert record.user_confirmed is False
    assert [item.learning_id for item in records] == ["main-agent-learning"]
    assert diagnostics_record.valid_record_count == 1
    assert (run_dir / "app_run_state.json").read_bytes() == before


def test_learning_target_must_bind_to_existing_run_fact(tmp_path):
    run_dir = drive_terminal_run(tmp_path)
    with pytest.raises(ValueError, match="learning reason reference"):
        record_researcher_learning(
            run_dir,
            source="user",
            previous_view="old",
            updated_view="new",
            change_reason_refs=["node-claim:missing"],
        )


def test_main_agent_inferred_learning_cannot_be_user_confirmed():
    with pytest.raises(ValidationError, match="user_confirmed"):
        ResearcherLearningRecordV1(
            learning_id="forged",
            timestamp="2026-01-01T00:00:00+00:00",
            run_id="run",
            source="main_agent",
            previous_view="old",
            updated_view="new",
            change_reason_refs=["run:run"],
            user_confirmed=True,
        )


def test_user_confirmation_appends_without_modifying_prior_learning(tmp_path):
    run_dir = drive_terminal_run(tmp_path)
    first = record_researcher_learning(
        run_dir,
        source="main_agent",
        previous_view="old",
        updated_view="possible update",
        change_reason_refs=["node-claim:parent"],
        learning_id="inferred",
    )
    second = record_researcher_learning(
        run_dir,
        source="user",
        previous_view="old",
        updated_view="confirmed update",
        change_reason_refs=["learning:inferred"],
        learning_id="confirmed",
    )
    records, _ = read_researcher_learning_ledger(run_dir)

    assert first.user_confirmed is False
    assert second.user_confirmed is True
    assert [item.learning_id for item in records] == ["inferred", "confirmed"]
    assert records[0].updated_view == "possible update"


def test_damaged_learning_tail_is_quarantined_before_append(tmp_path):
    run_dir = drive_terminal_run(tmp_path)
    record_researcher_learning(
        run_dir,
        source="user",
        previous_view="old",
        updated_view="first",
        change_reason_refs=["node-claim:parent"],
        learning_id="first",
    )
    path = run_dir / "epistemic" / "researcher_learning.jsonl"
    with path.open("ab") as handle:
        handle.write(b'{"learning_id":')
    record_researcher_learning(
        run_dir,
        source="user",
        previous_view="first",
        updated_view="second",
        change_reason_refs=["node-claim:parent"],
        learning_id="second",
    )
    records, quality = read_researcher_learning_ledger(run_dir)

    assert [item.learning_id for item in records] == ["first", "second"]
    assert quality.corrupt_tail_repaired is True
    assert path.with_suffix(path.suffix + ".corrupt").exists()


def test_duplicate_learning_id_is_rejected(tmp_path):
    run_dir = drive_terminal_run(tmp_path)
    kwargs = dict(
        source="user",
        previous_view="old",
        updated_view="new",
        change_reason_refs=["node-claim:parent"],
        learning_id="same",
    )
    record_researcher_learning(run_dir, **kwargs)
    with pytest.raises(DuplicateLearningError):
        record_researcher_learning(run_dir, **kwargs)


def test_epistemic_cli_json_text_and_record_learning(tmp_path):
    run_dir = drive_terminal_run(tmp_path)
    root = Path(__file__).resolve().parents[1]
    json_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dte_backend",
            "epistemic-summary",
            "--run-dir",
            str(run_dir),
            "--format",
            "json",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    text_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dte_backend",
            "epistemic-summary",
            "--run-dir",
            str(run_dir),
            "--format",
            "text",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    learning_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dte_backend",
            "record-learning",
            "--run-dir",
            str(run_dir),
            "--source",
            "user",
            "--previous-view",
            "old",
            "--updated-view",
            "new",
            "--reason-ref",
            "node-claim:parent",
            "--reusable-heuristic",
            "check dependencies first",
            "--learning-id",
            "cli-learning",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert json_result.returncode == 0, json_result.stderr
    assert text_result.returncode == 0, text_result.stderr
    assert learning_result.returncode == 0, learning_result.stderr
    assert json.loads(json_result.stdout)["schema_version"] == (
        "dte-terminal-epistemic-handoff.v1"
    )
    assert "current most credible conclusions" in text_result.stdout.casefold()
    assert json.loads(learning_result.stdout)["user_confirmed"] is True


def test_epistemic_read_path_launches_no_codex_subprocess(monkeypatch, tmp_path):
    run_dir = drive_terminal_run(tmp_path)
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("epistemic read path launched a subprocess")
        ),
    )
    assert build_terminal_epistemic_handoff(run_dir).selected_claims


def test_skill_and_agents_require_both_terminal_summaries_and_explicit_learning():
    root = Path(__file__).resolve().parents[1]
    skill = (root / "SKILL.md").read_text(encoding="utf-8")
    agents = (root / "AGENTS.md").read_text(encoding="utf-8")
    combined = skill + "\n" + agents

    assert "observability-summary --run-dir <run-dir> --format json" in combined
    assert "epistemic-summary --run-dir <run-dir> --format json" in combined
    assert "record-learning --source user" in combined
    assert "most dangerous" in combined.casefold() or "最危险" in combined
    assert "silence" in combined.casefold()
