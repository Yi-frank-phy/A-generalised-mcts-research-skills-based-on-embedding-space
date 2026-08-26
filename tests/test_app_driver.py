from tests.helpers import completed_node, completed_candidate
import json
import hashlib
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pytest

import dte_backend.app_driver as app_driver
from dte_backend.app_driver import (
    app_run_status,
    cancel_app_episode,
    cancel_app_run,
    create_app_run,
    fail_app_episode,
    next_app_episode,
    request_app_synthesis,
    retry_app_episode,
    submit_app_episode_result,
)
from dte_backend.episode_models import EpisodeResult, ExecutorEpisodeOutput, ExecutorNodeCandidate, RuntimeDiagnostics, RuntimeLimits, compute_output_hash, canonical_json_bytes
from dte_backend.control import OperatorAuthorizationError
from dte_backend.embedding import HashEmbeddingProvider
from dte_backend.models import BudgetSpec, DTERunSpec, SearchNode, SynthesisControlRequest
from dte_backend.telemetry import EpisodeEventLog


def spec() -> DTERunSpec:
    return DTERunSpec(
        problem="p",
        goal="g",
        budget=BudgetSpec(
            max_iterations=2,
            allocation_mass_per_iteration=1,
            max_children_per_iteration=2,
            max_relation_enrichment_pairs=0,
        ),
        embedding_provider="hash",
        embedding_dimension=8,
    )


def parent() -> SearchNode:
    return completed_node(node_id="parent", claim="committed parent")
















def test_app_create_rejects_initial_nodes_above_search_node_cap(tmp_path):
    bounded = spec().model_copy(
        update={
            "budget": spec().budget.model_copy(
                update={"max_committed_search_nodes": 1}
            )
        }
    )
    with pytest.raises(
        ValueError,
        match="initial committed search nodes exceed max_committed_search_nodes",
    ):
        create_app_run(
            tmp_path / "over-cap",
            bounded,
            [
                completed_node(node_id="one", claim="one"),
                completed_node(node_id="two", claim="two"),
            ],
        )








def result_for(request, *, children=1, node_id_prefix="child", status="completed"):
    output = None
    if status == "completed":
        output = ExecutorEpisodeOutput(
            nodes=[
                completed_candidate(
                    node_id=f"{node_id_prefix}-{index}",
                    claim=f"candidate {index}",
                    parent_ids=[request.parent_node_id],
                )
                for index in range(children)
            ],
            episode_summary="App-native episode completed",
        )
    return EpisodeResult(
        episode_id=request.episode_id,
        attempt_id=request.attempt_id,
        run_id=request.run_id,
        role=request.role,
        input_graph_revision=request.input_graph_revision,
        selected_node_revisions=request.selected_node_revisions,
        status=status,
        structured_output=output,
        runtime_diagnostics=RuntimeDiagnostics(
            adapter_name="codex-app-main-agent",
            transport_name="current-app-runtime",
            profile="native-autonomous",
            usage_source="unavailable",
        ),
        output_hash=compute_output_hash(output, request.output_schema_version),
        schema_version=request.output_schema_version,
    )


def graph_snapshot(run_dir):
    state = app_run_status(run_dir)
    return {
        "revision": state.graph_revision,
        "node_revisions": dict(state.node_revisions),
        "nodes": [node.model_dump(mode="json") for node in state.nodes],
    }


def lifecycle_for(state, episode_id):
    return next(episode for episode in state.episodes if episode.episode_id == episode_id)
































def test_skill_and_agents_define_current_app_loop_without_sdk_primary_path():
    root = Path(__file__).resolve().parents[1]
    skill = (root / "SKILL.md").read_text(encoding="utf-8")
    agents = (root / "AGENTS.md").read_text(encoding="utf-8")
    combined = skill + agents
    for command in (
        "hook-driver activate",
        "hook-driver init",
        "hook-driver step",
        "hook-driver submit",
        "hook-driver control",
        "hook-driver handoff",
    ):
        assert command in combined
    assert "direct `create-run`, `next-episode`, and `submit-episode-result`" in combined
    assert "current App main agent performs the episode" in combined
    assert "Do not launch another Codex process" in combined
    assert "CodexSdkEpisodeAdapter" not in combined
    assert "subagent count" in combined


