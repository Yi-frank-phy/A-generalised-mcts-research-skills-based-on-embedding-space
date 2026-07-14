"""Append-only JSONL telemetry for DTE runs and bounded episodes."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EVENT_TYPES = {
    "run_created",
    "episode_granted",
    "episode_started",
    "episode_submitted",
    "episode_completed",
    "episode_failed",
    "episode_cancelled",
    "episode_expired",
    "episode_superseded",
    "output_rejected",
    "nodes_committed",
    "judge_observations_committed",
    "allocation_recorded",
    "relation_candidates_generated",
    "relation_episode_granted",
    "relation_observations_committed",
    "relation_result_rejected",
    "merge_proposed",
    "merge_applied",
    "conflict_recorded",
    "complementarity_recorded",
    "synthesis_readiness_evaluated",
    "synthesis_blocked_by_relation",
    "run_completed",
}


class EpisodeEventLog:
    """Small append-only event writer; telemetry never changes controller state."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def emit(self, event_type: str, **fields: Any) -> None:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"unsupported episode event type: {event_type}")
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "run_id": None,
            "episode_id": None,
            "attempt_id": None,
            "role": None,
            "adapter_name": None,
            "transport_name": None,
            "profile": None,
            "runtime_profile": None,
            "model": None,
            "wall_clock_ms": None,
            "queue_or_io_ms": None,
            "retry_count": 0,
            "status": None,
            "input_graph_revision": None,
            "returned_node_count": None,
            "accepted_node_count": None,
            "selected_node_count": None,
            "selected_pair_count": None,
            "returned_observation_count": None,
            "accepted_observation_count": None,
            "allocated_child_count": None,
            "spatial_entropy": None,
            "equivalent_count": None,
            "complementary_count": None,
            "conflict_count": None,
            "independent_count": None,
            "material_conflict_count": None,
            "merge_count": None,
            "blocking_candidate_count": None,
            "rejection_reason": None,
            "input_tokens": None,
            "output_tokens": None,
            "cached_tokens": None,
            "provider_reported_cost": None,
            "estimated_cost": None,
            "quota_delta": None,
            "usage_source": "unavailable",
            "schema_valid": None,
            "controller_field_violation_count": 0,
            "duplicate_within_result_count": 0,
            "later_judge_survival_count": None,
            "later_relation_outcome": None,
        }
        unknown = set(fields).difference(record)
        if unknown:
            raise ValueError(f"unsupported telemetry fields: {sorted(unknown)}")
        record.update(fields)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def read_events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line]
