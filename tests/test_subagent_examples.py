import json
from pathlib import Path

from dte_backend.adapter import validate_adapter_output
from dte_backend.models import ExpansionRequest, SearchNode
from dte_backend.oracle_validation import validate_judge_output, validate_relation_output


ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPTS = ROOT / "examples" / "subagent_transcripts"


def load_transcript(name: str) -> dict:
    return json.loads((TRANSCRIPTS / name).read_text(encoding="utf-8"))


def test_codex_subagent_transcripts_preserve_prompt_order():
    for name, role_prompt in [
        ("judge_call.json", "prompts/judge_oracle.md"),
        ("executor_call.json", "prompts/executor_subagent.md"),
        ("relation_call.json", "prompts/relation_oracle.md"),
    ]:
        transcript = load_transcript(name)

        assert transcript["prompt_order"] == [
            "prompts/DTE_STATIC_PREFIX.md",
            role_prompt,
            "dynamic_json_payload",
        ]
        assert transcript["dynamic_payload_position"] == "last"
        assert transcript["subagent_response_format"] == "json_only"


def test_judge_transcript_response_passes_guard_contract():
    transcript = load_transcript("judge_call.json")
    nodes = [SearchNode.model_validate(item) for item in transcript["dynamic_payload"]["nodes"]]

    results = validate_judge_output(nodes, transcript["subagent_response"])

    assert [result.node_id for result in results] == ["n1", "n2"]


def test_executor_transcript_response_passes_guard_contract():
    transcript = load_transcript("executor_call.json")
    request = ExpansionRequest.model_validate(transcript["dynamic_payload"])

    children = validate_adapter_output(
        request.parent,
        request.child_count,
        transcript["subagent_response"],
    )

    assert children[0].parent_ids == [request.parent.node_id]


def test_relation_transcript_response_passes_guard_contract():
    transcript = load_transcript("relation_call.json")
    nodes = [SearchNode.model_validate(item) for item in transcript["dynamic_payload"]["nodes"]]

    result = validate_relation_output(nodes, transcript["subagent_response"])

    assert result.source_node_ids == ["n1", "n2"]


def test_mock_end_to_end_example_references_transcripts_and_artifacts():
    doc = (ROOT / "docs" / "MOCK_END_TO_END_EXAMPLE.md").read_text(encoding="utf-8")

    for required in [
        "examples/subagent_transcripts/judge_call.json",
        "examples/subagent_transcripts/executor_call.json",
        "examples/subagent_transcripts/relation_call.json",
        "artifacts/smoke-workflow/main_agent_status.md",
        "artifacts/smoke-workflow/frontier.md",
        "artifacts/smoke-workflow/entropy_trace.md",
        "artifacts/smoke-workflow/relation_candidates.md",
        "artifacts/smoke-workflow/human_questions.md",
        "artifacts/smoke-workflow/report.md",
    ]:
        assert required in doc
