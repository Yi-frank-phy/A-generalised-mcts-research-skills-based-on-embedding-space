from __future__ import annotations

import pytest
from pydantic import ValidationError

from dte_backend.relation_models import RelationObservation, RelationRecord


def conflict_observation(*, disclosure_required: bool = False) -> RelationObservation:
    return RelationObservation(
        candidate_id="candidate-a-b",
        left_node_id="a",
        right_node_id="b",
        relation_type="conflict",
        confidence=0.85,
        rationale="The claims require incompatible assumptions.",
        evidence_refs=["evidence-a", "evidence-b"],
        materiality_assessment="material",
        conflict_summary="Both claims cannot hold under the shared boundary condition.",
        disclosure_required=disclosure_required,
    )


def record_payload(observation: RelationObservation) -> dict[str, object]:
    return {
        "relation_record_id": "record-a-b",
        "candidate_id": observation.candidate_id,
        "left_node_id": observation.left_node_id,
        "right_node_id": observation.right_node_id,
        "relation_type": observation.relation_type,
        "scheduling_class": "blocking",
        "confidence": observation.confidence,
        "rationale": observation.rationale,
        "evidence_refs": list(observation.evidence_refs),
        "material_to_synthesis": True,
        "materiality_assessment": observation.materiality_assessment,
        "observation": observation,
        "disclosure_required": observation.disclosure_required,
        "episode_id": "episode-relation-a-b",
        "attempt_id": "attempt-relation-a-b",
        "input_graph_revision": 3,
        "selected_node_revisions": {"a": 1, "b": 2},
        "output_hash": "output-hash",
        "schema_version": "relation-output.v1",
        "committed_at": "2026-01-01T00:00:00+00:00",
    }


@pytest.mark.parametrize(
    ("field_name", "mismatched_value"),
    [
        ("candidate_id", "candidate-other"),
        ("left_node_id", "other-left"),
        ("right_node_id", "other-right"),
        ("relation_type", "independent"),
        ("confidence", 0.25),
        ("rationale", "Different rationale."),
        ("evidence_refs", ["evidence-a"]),
        ("materiality_assessment", "uncertain"),
    ],
)
def test_relation_record_rejects_fields_that_disagree_with_observation(
    field_name: str,
    mismatched_value: object,
) -> None:
    observation = conflict_observation()
    payload = record_payload(observation)
    payload[field_name] = mismatched_value

    with pytest.raises(ValidationError, match=field_name):
        RelationRecord.model_validate(payload)


def test_relation_record_allows_backend_to_promote_conflict_disclosure() -> None:
    observation = conflict_observation(disclosure_required=False)
    payload = record_payload(observation)
    payload["disclosure_required"] = True

    record = RelationRecord.model_validate(payload)

    assert record.observation.disclosure_required is False
    assert record.disclosure_required is True


def test_relation_record_cannot_drop_observation_disclosure_requirement() -> None:
    observation = conflict_observation(disclosure_required=True)
    payload = record_payload(observation)
    payload["disclosure_required"] = False

    with pytest.raises(ValidationError, match="cannot remove disclosure"):
        RelationRecord.model_validate(payload)


def test_relation_record_disclosure_is_only_valid_for_conflict() -> None:
    observation = RelationObservation(
        candidate_id="candidate-a-b",
        left_node_id="a",
        right_node_id="b",
        relation_type="independent",
        confidence=0.85,
        rationale="The claims address unrelated mechanisms.",
        evidence_refs=[],
        materiality_assessment="non_material",
        independence_summary="Neither claim changes the interpretation of the other.",
    )
    payload = record_payload(observation)
    payload["disclosure_required"] = True

    with pytest.raises(ValidationError, match="only valid for conflicts"):
        RelationRecord.model_validate(payload)
