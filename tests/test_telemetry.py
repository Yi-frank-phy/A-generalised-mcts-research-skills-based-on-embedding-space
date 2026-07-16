from __future__ import annotations

import json
from pathlib import Path

import pytest

from dte_backend.telemetry import EpisodeEventLog


def _event(event_id: str, event_type: str = "run_created") -> bytes:
    return json.dumps(
        {"event_id": event_id, "event_type": event_type},
        sort_keys=True,
    ).encode("utf-8")


def _assert_live_log_is_valid_jsonl(path: Path) -> list[dict[str, object]]:
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    records = [json.loads(line.decode("utf-8")) for line in raw.splitlines() if line]
    assert all(isinstance(record, dict) for record in records)
    return records


def test_emit_preserves_complete_json_tail_missing_only_newline(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"
    path.write_bytes(_event("first"))
    log = EpisodeEventLog(path)

    log.emit("run_completed", event_id="second", run_id="run-1")

    records = _assert_live_log_is_valid_jsonl(path)
    assert [record["event_id"] for record in records] == ["first", "second"]
    assert not path.with_suffix(".jsonl.corrupt").exists()


def test_replay_deduplicates_complete_event_tail_missing_newline(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"
    path.write_bytes(_event("same"))
    log = EpisodeEventLog(path)

    log.emit("run_created", event_id="same")

    records = _assert_live_log_is_valid_jsonl(path)
    assert [record["event_id"] for record in records] == ["same"]


@pytest.mark.parametrize(
    "damaged_tail",
    [
        b'{"event_id":"cut-off"',
        b'{"event_id":\n',
        b"\xff\xfe\n",
    ],
)
def test_damaged_tail_is_quarantined_and_does_not_break_read_or_emit(
    tmp_path: Path,
    damaged_tail: bytes,
) -> None:
    path = tmp_path / "episodes.jsonl"
    path.write_bytes(_event("first") + b"\n" + damaged_tail)
    log = EpisodeEventLog(path)

    assert [event["event_id"] for event in log.read_events()] == ["first"]

    quarantine = path.with_suffix(".jsonl.corrupt")
    assert quarantine.exists()
    assert damaged_tail.rstrip(b"\n") in quarantine.read_bytes()
    assert [record["event_id"] for record in _assert_live_log_is_valid_jsonl(path)] == [
        "first"
    ]

    log.emit("run_completed", event_id="after-repair", run_id="run-1")

    records = _assert_live_log_is_valid_jsonl(path)
    assert [record["event_id"] for record in records] == ["first", "after-repair"]


def test_read_events_skips_malformed_interior_line(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"
    path.write_bytes(
        _event("first")
        + b"\n"
        + b"\xff malformed interior\n"
        + _event("last")
        + b"\n"
    )

    events = EpisodeEventLog(path).read_events()

    assert [event["event_id"] for event in events] == ["first", "last"]


def test_emit_directly_repairs_malformed_tail_before_append(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"
    path.write_bytes(_event("first") + b'\n{"event_id":')
    log = EpisodeEventLog(path)

    log.emit("run_completed", event_id="second", run_id="run-1")

    records = _assert_live_log_is_valid_jsonl(path)
    assert [record["event_id"] for record in records] == ["first", "second"]
    assert path.with_suffix(".jsonl.corrupt").read_bytes().startswith(b'{"event_id":')
