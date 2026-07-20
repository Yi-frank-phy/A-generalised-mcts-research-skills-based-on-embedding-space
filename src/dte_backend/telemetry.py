"""Append-only JSONL telemetry for DTE runs and bounded episodes."""

from __future__ import annotations

import json
import os
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
    "continuation_gate_evaluated",
    "continuation_granted",
    "early_synthesis_selected",
    "search_node_budget_exhausted",
    "relation_candidates_generated",
    "relation_blocking_inventory_evaluated",
    "relation_blocking_inventory_completed",
    "relation_episode_granted",
    "relation_enrichment_granted",
    "relation_enrichment_committed",
    "relation_enrichment_budget_exhausted",
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

    @staticmethod
    def _decode_event_line(line: bytes) -> dict[str, Any] | None:
        """Decode one JSONL record without letting damaged bytes escape."""

        if line.endswith(b"\r"):
            line = line[:-1]
        if not line:
            return None
        try:
            event = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        return event if isinstance(event, dict) else None

    def _replace_contents(self, payload: bytes) -> None:
        """Durably stage repaired bytes before atomically installing them."""

        temporary = self.path.with_suffix(self.path.suffix + ".repair.tmp")
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self.path)

    def _repair_incomplete_tail(self) -> None:
        """Restore JSONL framing after a crashed/short append.

        A complete final JSON object which only lacks its newline is preserved.
        A malformed partial tail is quarantined and removed from the live log
        before durable outbox replay appends the complete event.
        """

        if not self.path.exists():
            return
        raw = self.path.read_bytes()
        if not raw:
            return

        # Ignore only empty physical lines at EOF while locating the final
        # record. A whitespace-only or non-object JSON line is damaged data,
        # not an empty JSONL separator.
        tail_end = len(raw)
        while tail_end > 0 and raw[tail_end - 1 : tail_end] == b"\n":
            tail_end -= 1
            if tail_end > 0 and raw[tail_end - 1 : tail_end] == b"\r":
                tail_end -= 1
        if tail_end == 0:
            return
        tail_start = raw.rfind(b"\n", 0, tail_end) + 1
        tail_line = raw[tail_start:tail_end]

        if self._decode_event_line(tail_line) is not None:
            if not raw.endswith(b"\n"):
                self._replace_contents(raw + b"\n")
            return

        # A short write can leave either an unterminated fragment or a
        # malformed line whose newline reached disk. Remove the entire final
        # physical record in both cases so every live non-empty line remains
        # valid JSON. Preserve the bytes separately for diagnosis.
        corrupt_tail = raw[tail_start:]
        quarantine = self.path.with_suffix(self.path.suffix + ".corrupt")
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        with quarantine.open("ab") as handle:
            handle.write(corrupt_tail)
            if not corrupt_tail.endswith(b"\n"):
                handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._replace_contents(raw[:tail_start])

    def emit(self, event_type: str, **fields: Any) -> None:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"unsupported episode event type: {event_type}")
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_id": None,
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
            "graph_revision": None,
            "controller_iteration": None,
            "returned_node_count": None,
            "accepted_node_count": None,
            "selected_node_count": None,
            "selected_pair_count": None,
            "returned_observation_count": None,
            "accepted_observation_count": None,
            "allocated_child_count": None,
            "spatial_entropy": None,
            "entropy_delta": None,
            "consecutive_plateau_count": None,
            "committed_search_node_count": None,
            "remaining_search_node_slots": None,
            "canonical_frontier_count": None,
            "trigger_signals": None,
            "material_yield_signals": None,
            "continuation_target_node_ids": None,
            "decision": None,
            "reason": None,
            "equivalent_count": None,
            "complementary_count": None,
            "conflict_count": None,
            "independent_count": None,
            "material_conflict_count": None,
            "merge_count": None,
            "blocking_candidate_count": None,
            "provisional_selected_node_count": None,
            "blocking_pair_count": None,
            "resolved_blocking_pair_count": None,
            "unresolved_blocking_pair_count": None,
            "blocking_inventory_complete": None,
            "enrichment_candidate_count": None,
            "enrichment_pairs_committed": None,
            "enrichment_pairs_remaining": None,
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
        self._repair_incomplete_tail()
        event_id = record["event_id"]
        if event_id is not None and self.path.exists():
            # AppRunState uses event IDs as a tiny durable outbox. If a
            # process dies after append but before clearing the outbox, replay
            # must not duplicate an already-published fact.
            for line in self.path.read_bytes().split(b"\n"):
                existing = self._decode_event_line(line)
                if existing is None:
                    continue
                if existing.get("event_id") == event_id:
                    return
        encoded = (
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        with self.path.open("ab") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())

    def read_events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        self._repair_incomplete_tail()
        events: list[dict[str, Any]] = []
        for line in self.path.read_bytes().split(b"\n"):
            event = self._decode_event_line(line)
            if event is not None:
                events.append(event)
        return events
