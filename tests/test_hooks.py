import json
import subprocess
import sys

from dte_backend.episode_adapter import build_relation_episode_request
from dte_backend.episode_commit import EpisodeGraph
from dte_backend.episode_models import EpisodeResult, RuntimeDiagnostics, compute_output_hash
from dte_backend.models import SearchNode
from dte_backend.relation_candidates import generate_relation_candidates
from dte_backend.relation_models import RelationEpisodeOutput, RelationObservation


def test_dte_guard_accepts_sample_artifacts():
    commands = [
        [sys.executable, "hooks/dte_guard.py", "spec", "examples/run_spec.json"],
        [
            sys.executable,
            "hooks/dte_guard.py",
            "executor",
            "--parent",
            "examples/executor_parent.json",
            "--output",
            "examples/executor_output.json",
            "--child-count",
            "1",
        ],
        [
            sys.executable,
            "hooks/dte_guard.py",
            "judge",
            "--nodes",
            "examples/frontier_nodes.json",
            "--output",
            "examples/judge_output.json",
        ],
        [
            sys.executable,
            "hooks/dte_guard.py",
            "relation",
            "--nodes",
            "examples/frontier_nodes.json",
            "--output",
            "examples/relation_output.json",
        ],
    ]

    for command in commands:
        completed = subprocess.run(command, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr


def test_prompt_guard_injects_readable_chinese_dte_reminder():
    payload = {
        "cwd": r"C:\Users\zhaoy\Downloads\dte-codex-skill-backend",
        "prompt": "请运行 dte-extreme-research",
    }

    completed = subprocess.run(
        [sys.executable, "hooks/dte_prompt_guard.py"],
        input=json.dumps(payload, ensure_ascii=False),
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    output = json.loads(completed.stdout)
    reminder = output["hookSpecificOutput"]["additionalContext"]
    assert "必须先使用已安装的 `dte-extreme-research` skill" in reminder
    assert "任何 guard 失败都必须停止消费该产物" in reminder
    assert chr(0xFFFD) not in reminder


def test_dte_guard_rejects_bad_gemini_dimension(tmp_path):
    bad_spec = json.loads(open("examples/run_spec.json", encoding="utf-8").read())
    bad_spec["embedding_provider"] = "gemini-embedding-2"
    bad_spec["embedding_dimension"] = 128
    spec_path = tmp_path / "bad_spec.json"
    spec_path.write_text(json.dumps(bad_spec), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, "hooks/dte_guard.py", "spec", str(spec_path)],
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "Gemini geometry must use embedding_dimension=3072" in completed.stderr


def test_relation_guard_validates_app_native_envelope_and_firewall(tmp_path):
    graph = EpisodeGraph(nodes=[SearchNode(node_id="a", claim="same"), SearchNode(node_id="b", claim=" SAME ")])
    candidates = generate_relation_candidates(
        graph.nodes,
        node_revisions=graph.node_revisions,
        graph_revision=graph.revision,
        provisional_synthesis_node_ids=["a", "b"],
    )
    request = build_relation_episode_request(
        graph,
        candidates[:1],
        run_id="guard-run",
        problem="p",
        goal="g",
        constraints=[],
        provisional_synthesis_node_ids=["a", "b"],
        max_relation_pairs_per_episode=1,
    )
    pair = request.relation_payload.candidate_pairs[0]
    output = RelationEpisodeOutput(
        observations=[
            RelationObservation(
                candidate_id=pair.candidate_id,
                left_node_id="a",
                right_node_id="b",
                relation_type="equivalent",
                confidence=0.9,
                rationale="same",
                evidence_refs=[],
                materiality_assessment="material",
                merge_recommended=True,
            )
        ]
    )
    result = EpisodeResult(
        episode_id=request.episode_id,
        attempt_id=request.attempt_id,
        run_id=request.run_id,
        role="relation",
        input_graph_revision=request.input_graph_revision,
        selected_node_revisions=request.selected_node_revisions,
        status="completed",
        structured_output=output,
        runtime_diagnostics=RuntimeDiagnostics(
            adapter_name="app",
            transport_name="app",
            profile="native-autonomous",
        ),
        output_hash=compute_output_hash(output, request.output_schema_version),
        schema_version=request.output_schema_version,
    )
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    request_path.write_text(request.model_dump_json(indent=2), encoding="utf-8")
    result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "hooks/dte_guard.py",
            "relation",
            "--request",
            str(request_path),
            "--output",
            str(result_path),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
