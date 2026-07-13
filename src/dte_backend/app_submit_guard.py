"""Fail-closed identity guard for App-native episode submissions.

The persistent App driver owns lifecycle validation, but an unknown or missing
``episode_id`` / ``attempt_id`` must still return a structured rejection rather
than leaking a raw ``KeyError`` from record lookup.  This wrapper is installed
on the public App-driver submission entrypoint by :mod:`dte_backend`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .episode_models import CommitOutcome


def _mapping_payload(raw_result: Any) -> dict[str, Any] | None:
    if isinstance(raw_result, Mapping):
        return dict(raw_result)
    model_dump = getattr(raw_result, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        if isinstance(dumped, Mapping):
            return dict(dumped)
    return None


def build_fail_closed_submit(
    *,
    original_submit: Callable[[str | Any, Any], Any],
    load_state: Callable[[str | Any], Any],
    event_log_factory: Callable[[str | Any], Any],
    submit_outcome_type: Callable[..., Any],
) -> Callable[[str | Any, Any], Any]:
    """Wrap App submission so identity lookup failures are normal rejections."""

    def reject(run_dir: str | Any, payload: Mapping[str, Any], reason: str) -> Any:
        state = load_state(run_dir)
        episode_value = payload.get("episode_id")
        attempt_value = payload.get("attempt_id")
        episode_id = episode_value if isinstance(episode_value, str) else ""
        attempt_id = attempt_value if isinstance(attempt_value, str) else ""
        outcome = CommitOutcome(
            accepted=False,
            episode_id=episode_id,
            graph_revision_before=state.graph_revision,
            graph_revision_after=state.graph_revision,
            rejection_reason=reason,
        )
        event_log_factory(run_dir).emit(
            "output_rejected",
            run_id=state.run_id,
            episode_id=episode_id or None,
            attempt_id=attempt_id or None,
            status="rejected",
            input_graph_revision=state.graph_revision,
            accepted_node_count=0,
            rejection_reason=reason,
            schema_valid=False,
            usage_source="unavailable",
        )
        return submit_outcome_type(
            run_id=state.run_id,
            episode_id=episode_id,
            attempt_id=attempt_id,
            commit_outcome=outcome,
            next_controller_action=state.controller_action,
        )

    def guarded_submit(run_dir: str | Any, raw_result: Any) -> Any:
        payload = _mapping_payload(raw_result)
        if payload is None:
            return reject(
                run_dir,
                {},
                "episode result must be a mapping or a model with model_dump()",
            )

        for field_name in ("episode_id", "attempt_id"):
            value = payload.get(field_name)
            if not isinstance(value, str) or not value.strip():
                return reject(
                    run_dir,
                    payload,
                    f"episode result is missing a non-empty {field_name}",
                )

        try:
            return original_submit(run_dir, raw_result)
        except KeyError as exc:
            detail = exc.args[0] if exc.args else str(exc)
            return reject(
                run_dir,
                payload,
                f"episode identity lookup failed: {detail}",
            )

    guarded_submit.__name__ = getattr(original_submit, "__name__", "submit_app_episode_result")
    guarded_submit.__doc__ = original_submit.__doc__
    return guarded_submit
