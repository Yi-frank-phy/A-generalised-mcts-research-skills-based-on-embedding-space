import json
from pathlib import Path

from dte_backend.adapter import validate_adapter_output
from dte_backend.models import ExpansionRequest, SearchNode
from dte_backend.oracle_validation import validate_relation_output


ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPTS = ROOT / "examples" / "subagent_transcripts"


def load_transcript(name: str) -> dict:
    return json.loads((TRANSCRIPTS / name).read_text(encoding="utf-8"))






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


