from __future__ import annotations
from tests.helpers import completed_node, completed_candidate

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
    _search_dispositions,
    build_terminal_epistemic_handoff,
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
    TerminalEpistemicHandoffV1,
)
from dte_backend.episode_adapter import build_executor_episode_request
from dte_backend.episode_commit import EpisodeGraph, commit_episode_result
from dte_backend.episode_models import EpisodeResult, ExecutorEpisodeOutput, ExecutorNodeCandidate, RuntimeDiagnostics, compute_output_hash
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




def graph_snapshot(graph: EpisodeGraph):
    return graph.snapshot()


def test_executor_commits_structured_epistemic_contributions_atomically():
    graph = EpisodeGraph(nodes=[completed_node(node_id="parent", claim="parent claim")])
    request = direct_executor_request(graph)
    output = ExecutorEpisodeOutput(
        nodes=[
            completed_candidate(
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
    graph = EpisodeGraph(nodes=[completed_node(node_id="parent", claim="parent claim")])
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
    graph = EpisodeGraph(nodes=[completed_node(node_id="parent", claim="parent claim")])
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
    graph = EpisodeGraph(nodes=[completed_node(node_id="parent", claim="parent claim")])
    request = direct_executor_request(graph, grant=0)
    output = ExecutorEpisodeOutput(
        nodes=[],
        epistemic_contributions=assumption_bundle(),
    )
    raw_result = result_for(request, output).model_dump(mode="json")
    raw_result["structured_output"]["epistemic_contributions"]["statements"][0][
        "source_type"
    ] = source_type
    before = graph_snapshot(graph)
    outcome = commit_episode_result(graph, request, raw_result)
    assert outcome.accepted is False
    assert "schema validation failed" in (outcome.rejection_reason or "")
    assert graph_snapshot(graph) == before


def test_illegal_source_type_is_rejected_by_strict_schema():
    raw = assumption_bundle().model_dump(mode="json")
    raw["statements"][0]["source_type"] = "agent_verified"
    with pytest.raises(ValidationError, match="literal_error"):
        EpistemicContributionBundle.model_validate(raw)


def test_current_epistemic_schemas_expose_no_human_confirmation_semantics():
    contribution_schema = json.dumps(
        EpistemicContributionBundle.model_json_schema(), sort_keys=True
    )
    handoff_schema = json.dumps(
        TerminalEpistemicHandoffV1.model_json_schema(), sort_keys=True
    )
    assert "human_confirmed" not in contribution_schema
    assert "human_confirmed" not in handoff_schema
    assert "human_confirmed_selected_claim_count" not in handoff_schema
    assert "human_confirmed_record_count" not in handoff_schema
    assert "researcher_learning" not in handoff_schema


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




def test_external_artifact_backed_requires_an_external_or_artifact_basis():
    graph = EpisodeGraph(nodes=[completed_node(node_id="parent", claim="parent claim")])
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
    graph = EpisodeGraph(nodes=[completed_node(node_id="parent", claim="parent claim")])
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
        nodes=[completed_node(node_id="parent", claim="parent claim")]
    )
    request = direct_executor_request(first_graph, grant=0)
    output = ExecutorEpisodeOutput(
        nodes=[], epistemic_contributions=assumption_bundle()
    )
    result = result_for(request, output)
    assert commit_episode_result(first_graph, request, result).accepted

    replay_graph = EpisodeGraph(
        nodes=[completed_node(node_id="parent", claim="parent claim")],
        epistemic_ledger=first_graph.epistemic_ledger.model_copy(deep=True),
    )
    before = graph_snapshot(replay_graph)
    outcome = commit_episode_result(replay_graph, request, result)
    assert outcome.accepted is False
    assert "duplicate epistemic stable ID" in (outcome.rejection_reason or "")
    assert graph_snapshot(replay_graph) == before








def test_app_run_state_legacy_migration_defaults_to_empty_epistemic_ledger(tmp_path):
    run_dir = tmp_path / "legacy"
    create_app_run(
        run_dir,
        spec(),
        [completed_node(node_id="parent", claim="parent claim")],
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




def file_snapshot(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
























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








def test_learning_reference_is_rejected_by_current_commit_contract():
    graph = EpisodeGraph(nodes=[completed_node(node_id="parent", claim="parent claim")])
    request = direct_executor_request(graph, grant=0)
    output = ExecutorEpisodeOutput(
        nodes=[],
        epistemic_contributions=EpistemicContributionBundle(
            statements=[
                EpistemicStatementContribution(
                    local_id="legacy-learning-ref",
                    statement_type="evidence",
                    text="retired learning reference",
                    target_node_id="parent",
                    source_type="agent_reported",
                    basis_refs=["learning:retired"],
                )
            ]
        ),
    )
    before = graph_snapshot(graph)
    outcome = commit_episode_result(graph, request, result_for(request, output))
    assert outcome.accepted is False
    assert "learning: references" in (outcome.rejection_reason or "")
    assert graph_snapshot(graph) == before






def test_skill_and_agents_require_both_terminal_summaries_without_learning_ledger():
    root = Path(__file__).resolve().parents[1]
    skill = (root / "SKILL.md").read_text(encoding="utf-8")
    agents = (root / "AGENTS.md").read_text(encoding="utf-8")
    combined = skill + "\n" + agents

    assert "hook-driver handoff" in combined
    assert "observability" in combined.casefold()
    assert "epistemic" in combined.casefold()
    assert "terminal-handoff" in combined.casefold()
    assert "record-learning" not in combined
    assert "record-feedback" in combined
    assert "does not verify scientific truth" in combined.casefold()
    assert "most dangerous" in combined.casefold() or "最危险" in combined


