import math

import pytest

import dte_backend.app_driver as app_driver
from dte_backend.app_driver import create_app_run, next_app_episode, submit_app_episode_result
from dte_backend.embedding import HashEmbeddingProvider
from dte_backend.episode_models import (
    EpisodeResult,
    JudgeEpisodeOutput,
    JudgeObservation,
    RuntimeDiagnostics,
    compute_output_hash,
)
from dte_backend.models import BudgetSpec, DTERunSpec, SearchNode


def _judge_result(request: object) -> EpisodeResult:
    output = JudgeEpisodeOutput(
        observations=[
            JudgeObservation(
                node_id=node_id,
                score=0.8,
                reasoning="canonical controller test",
                risks=[],
            )
            for node_id in request.selected_node_revisions
        ]
    )
    return EpisodeResult(
        episode_id=request.episode_id,
        attempt_id=request.attempt_id,
        run_id=request.run_id,
        role="judge",
        input_graph_revision=request.input_graph_revision,
        selected_node_revisions=request.selected_node_revisions,
        status="completed",
        structured_output=output,
        runtime_diagnostics=RuntimeDiagnostics(
            adapter_name="test",
            transport_name="test",
            profile="native-autonomous",
            usage_source="unavailable",
        ),
        output_hash=compute_output_hash(output, request.output_schema_version),
        schema_version=request.output_schema_version,
    )


def test_app_controller_record_uses_current_frontier_diversity_for_temperature(tmp_path):
    spec = DTERunSpec(
        problem="canonical app controller",
        goal="record current-state temperature",
        budget=BudgetSpec(
            max_iterations=2,
            allocation_mass_per_iteration=2,
            max_children_per_iteration=2,
            max_relation_enrichment_pairs=0,
        ),
        embedding_provider="hash",
        embedding_dimension=8,
    )
    run_dir = tmp_path / "canonical-app"
    create_app_run(
        run_dir,
        spec,
        [
            SearchNode(node_id="a", claim="route A"),
            SearchNode(node_id="b", claim="route B"),
        ],
        run_id="canonical-app",
    )

    judge = next_app_episode(run_dir).request
    assert judge.role == "judge"
    submit_app_episode_result(run_dir, _judge_result(judge))

    state = app_driver.load_app_run(run_dir)
    action, _ = app_driver._progress_controller(
        run_dir,
        state,
        embedding_provider=HashEmbeddingProvider(dim=8),
    )

    assert action in {"continue_controller", "episode_required"}
    record = state.controller_iteration_records[-1]
    frontier_size = len(record.frontier_node_ids)
    expected_tau = 0.0 if frontier_size <= 1 else record.spatial_entropy / math.log(frontier_size)
    assert record.normalized_temperature == pytest.approx(expected_tau)
