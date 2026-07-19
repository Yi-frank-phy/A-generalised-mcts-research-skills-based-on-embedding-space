"""End-to-end App-native smoke for epistemic provenance and handoff."""

from __future__ import annotations

import tempfile
from pathlib import Path

from dte_backend.app_driver import (
    app_run_status,
    create_app_run,
    next_app_episode,
    submit_app_episode_result,
)
from dte_backend.embedding import HashEmbeddingProvider
from dte_backend.epistemic import (
    build_terminal_epistemic_handoff,
    record_researcher_learning,
)
from dte_backend.epistemic_models import (
    EpistemicContributionBundle,
    EpistemicEdgeContribution,
    EpistemicStatementContribution,
    PathDispositionContribution,
)
from dte_backend.episode_models import (
    EpisodeResult,
    ExecutorEpisodeOutput,
    ExecutorNodeCandidate,
    JudgeEpisodeOutput,
    JudgeObservation,
    RuntimeDiagnostics,
    compute_output_hash,
)
from dte_backend.models import BudgetSpec, DTERunSpec, SearchNode
from dte_backend.observability import build_run_observability_summary
from dte_backend.relation_models import RelationEpisodeOutput, RelationObservation


def _result(request, output) -> EpisodeResult:
    return EpisodeResult(
        episode_id=request.episode_id,
        attempt_id=request.attempt_id,
        run_id=request.run_id,
        role=request.role,
        input_graph_revision=request.input_graph_revision,
        selected_node_revisions=request.selected_node_revisions,
        status="completed",
        structured_output=output,
        runtime_diagnostics=RuntimeDiagnostics(
            adapter_name="codex-app-main-agent",
            transport_name="current-app-runtime",
            profile="native-autonomous",
            runtime_profile="high",
            model="smoke-model",
            usage_source="unavailable",
            diagnostics_source="unavailable",
        ),
        output_hash=compute_output_hash(output, request.output_schema_version),
        schema_version=request.output_schema_version,
    )


def _relation_output(request) -> RelationEpisodeOutput:
    observations = []
    assert request.relation_payload is not None
    for pair in request.relation_payload.candidate_pairs:
        observations.append(
            RelationObservation(
                candidate_id=pair.candidate_id,
                left_node_id=pair.left.node_id,
                right_node_id=pair.right.node_id,
                relation_type="conflict",
                confidence=0.8,
                rationale="the selected claims point in incompatible directions",
                evidence_refs=(
                    [pair.left.evidence[0].evidence_ref]
                    if pair.left.evidence
                    else []
                ),
                materiality_assessment="material",
                conflict_summary="the sufficient and not-sufficient claims conflict",
                conflicting_claims=[pair.left.claim, pair.right.claim],
                disclosure_required=True,
            )
        )
    return RelationEpisodeOutput(observations=observations)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="dte-epistemic-smoke-") as temporary:
        run_dir = Path(temporary) / "run"
        spec = DTERunSpec(
            problem="trace a disputed sufficient condition",
            goal="preserve assumptions, evidence, challenges, and conflict provenance",
            constraints=["do not infer formal edges from prose"],
            budget=BudgetSpec(
                max_iterations=1,
                allocation_mass_per_iteration=1,
                max_children_per_iteration=2,
                max_relation_pairs_per_episode=1,
                max_relation_enrichment_pairs=0,
                min_iterations_before_synthesis=2,
            ),
            embedding_provider="hash",
            embedding_dimension=8,
        )
        create_app_run(
            run_dir,
            spec,
            [
                SearchNode(
                    node_id="left",
                    claim="the condition is sufficient",
                    evidence=["shared premise"],
                ),
                SearchNode(
                    node_id="right",
                    claim="the condition is not sufficient",
                    evidence=["shared premise"],
                ),
            ],
            run_id="epistemic-smoke",
        )

        judge = next_app_episode(run_dir).request
        assert judge is not None and judge.role == "judge"
        judge_output = JudgeEpisodeOutput(
            observations=[
                JudgeObservation(
                    node_id=node_id,
                    score=0.8,
                    reasoning="material but conditional",
                    risks=["regularity remains unverified"],
                )
                for node_id in judge.selected_node_revisions
            ],
            epistemic_contributions=EpistemicContributionBundle(
                statements=[
                    EpistemicStatementContribution(
                        local_id="regularity",
                        statement_type="assumption",
                        text="the regularity condition holds",
                        target_node_id="left",
                        source_type="agent_reported",
                        basis_refs=[],
                    ),
                    EpistemicStatementContribution(
                        local_id="boundary-challenge",
                        statement_type="evidence",
                        text="the boundary case challenges sufficiency",
                        target_node_id="left",
                        source_type="agent_reported",
                        basis_refs=[],
                    ),
                ],
                edges=[
                    EpistemicEdgeContribution(
                        local_id="requires-regularity",
                        source_ref="node-claim:left",
                        target_ref="local-statement:regularity",
                        relation_type="requires",
                        source_type="agent_reported",
                        basis_refs=[],
                        explanation="sufficiency depends on regularity",
                    ),
                    EpistemicEdgeContribution(
                        local_id="challenge-left",
                        source_ref="local-statement:boundary-challenge",
                        target_ref="node-claim:left",
                        relation_type="challenges",
                        source_type="agent_reported",
                        basis_refs=[],
                        explanation="the boundary case weakens the selected claim",
                    ),
                ],
                path_dispositions=[
                    PathDispositionContribution(
                        local_id="left-challenged",
                        target_node_id="left",
                        epistemic_disposition="challenged",
                        source_type="agent_reported",
                        basis_refs=["local-statement:boundary-challenge"],
                        explanation="a material challenge remains",
                    )
                ],
            ),
        )
        assert submit_app_episode_result(
            run_dir, _result(judge, judge_output)
        ).commit_outcome.accepted

        executor = next_app_episode(
            run_dir, embedding_provider=HashEmbeddingProvider(dim=8)
        ).request
        assert executor is not None and executor.role == "executor"
        child_id = "bounded-child"
        executor_output = ExecutorEpisodeOutput(
            nodes=[
                ExecutorNodeCandidate(
                    node_id=child_id,
                    claim="the condition works away from the boundary",
                    parent_ids=[executor.parent_node_id],
                )
            ],
            epistemic_contributions=EpistemicContributionBundle(
                statements=[
                    EpistemicStatementContribution(
                        local_id="finite-check",
                        statement_type="evidence",
                        text="a bounded calculation supports the child claim",
                        target_node_id=child_id,
                        source_type="agent_reported",
                        basis_refs=[],
                    )
                ],
                edges=[
                    EpistemicEdgeContribution(
                        local_id="finite-support",
                        source_ref="local-statement:finite-check",
                        target_ref=f"node-claim:{child_id}",
                        relation_type="supports",
                        source_type="agent_reported",
                        basis_refs=[],
                        explanation="the calculation supplies bounded support",
                    ),
                    EpistemicEdgeContribution(
                        local_id="child-derived",
                        source_ref=f"node-claim:{child_id}",
                        target_ref=f"node-claim:{executor.parent_node_id}",
                        relation_type="derived_from",
                        source_type="agent_reported",
                        basis_refs=[],
                        explanation="the child refines its granted parent route",
                    ),
                ],
            ),
        )
        assert submit_app_episode_result(
            run_dir, _result(executor, executor_output)
        ).commit_outcome.accepted

        while True:
            outcome = next_app_episode(run_dir)
            if outcome.request is None:
                assert outcome.controller_action == "ready_for_synthesis"
                break
            if outcome.request.role == "judge":
                followup_judge = JudgeEpisodeOutput(
                    observations=[
                        JudgeObservation(
                            node_id=node_id,
                            score=0.7,
                            reasoning="the child is useful but remains provisional",
                            risks=["bounded evidence does not cover every case"],
                        )
                        for node_id in outcome.request.selected_node_revisions
                    ]
                )
                assert submit_app_episode_result(
                    run_dir, _result(outcome.request, followup_judge)
                ).commit_outcome.accepted
                continue
            if outcome.request.role == "executor":
                assert submit_app_episode_result(
                    run_dir,
                    _result(outcome.request, ExecutorEpisodeOutput(nodes=[])),
                ).commit_outcome.accepted
                continue
            assert outcome.request.role == "relation"
            assert submit_app_episode_result(
                run_dir,
                _result(outcome.request, _relation_output(outcome.request)),
            ).commit_outcome.accepted

        state_path = run_dir / "app_run_state.json"
        authoritative_before = state_path.read_bytes()
        operational_before = build_run_observability_summary(run_dir)
        epistemic_before = build_terminal_epistemic_handoff(run_dir)
        assert epistemic_before.selected_claims
        assert epistemic_before.material_conflicts
        child_handoff = next(
            item for item in epistemic_before.selected_claims if item.node_id == child_id
        )
        assert child_handoff.claim_origin == "executor_episode"
        assert child_handoff.claim_producing_episode_id == executor.episode_id
        assert child_handoff.claim_producing_attempt_id == executor.attempt_id

        record_researcher_learning(
            run_dir,
            source="user",
            previous_view="the condition looked sufficient without qualification",
            updated_view="the boundary must be checked separately",
            change_reason_refs=["node-claim:left"],
            reusable_heuristic="separate generic and boundary cases early",
            recognized_failure_mode="prematurely generalizing a bounded check",
            learning_id="smoke-learning",
        )
        operational_after = build_run_observability_summary(run_dir)
        epistemic_after = build_terminal_epistemic_handoff(run_dir)
        assert state_path.read_bytes() == authoritative_before
        assert operational_after == operational_before
        assert len(epistemic_after.researcher_learning) == 1

        # Restart through the normal state validator, then rebuild both views.
        app_run_status(run_dir)
        assert build_run_observability_summary(run_dir) == operational_after
        assert build_terminal_epistemic_handoff(run_dir) == epistemic_after
        assert state_path.read_bytes() == authoritative_before

    print("DTE epistemic provenance smoke ok")


if __name__ == "__main__":
    main()
