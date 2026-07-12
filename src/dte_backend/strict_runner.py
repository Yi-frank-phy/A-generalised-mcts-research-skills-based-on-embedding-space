"""Strict slash-command entrypoint for DTE skill runs.

`run` remains a flexible backend helper. `strict-run` is the locked workflow the
slash-command skill should use. It prevents Codex from freely mixing smoke tools,
heuristic fallbacks, and final synthesis.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Literal

from .adapter import ExecutorAdapter
from .artifacts import (
    render_checkpoint_summary_markdown,
    render_entropy_trace_markdown,
    render_frontier_markdown,
    render_human_questions_markdown,
    render_main_agent_status,
    render_relation_candidates_markdown,
    render_role_audit_markdown,
)
from .cache import CacheStats
from .control import load_synthesis_control
from .file_cache import FileDTECache
from .guards import enforce_run_spec_guard
from .models import DTERunSpec, SearchNode
from .runner import RunResult, run_frontier_search
from .subprocess_oracles import JudgeAdapter
from .synthesis import synthesize_report


StrictMode = Literal["smoke", "dry-run", "real"]


class StrictRunError(RuntimeError):
    """Raised when a strict-run invariant is violated."""


@dataclass(frozen=True)
class StrictRunPolicy:
    """Resolved policy for one strict run."""

    mode: StrictMode
    allow_mock: bool
    allow_hash_geometry: bool
    allow_heuristic_judge: bool
    require_cache_path: bool
    require_gemini_key: bool


def policy_for_mode(mode: StrictMode) -> StrictRunPolicy:
    if mode == "smoke":
        return StrictRunPolicy(
            mode=mode,
            allow_mock=True,
            allow_hash_geometry=True,
            allow_heuristic_judge=True,
            require_cache_path=False,
            require_gemini_key=False,
        )
    if mode == "dry-run":
        return StrictRunPolicy(
            mode=mode,
            allow_mock=False,
            allow_hash_geometry=True,
            allow_heuristic_judge=True,
            require_cache_path=True,
            require_gemini_key=False,
        )
    if mode == "real":
        return StrictRunPolicy(
            mode=mode,
            allow_mock=False,
            allow_hash_geometry=False,
            allow_heuristic_judge=False,
            require_cache_path=True,
            require_gemini_key=True,
        )
    raise StrictRunError(f"unsupported strict-run mode: {mode}")


def _is_mock_command(command: str | None) -> bool:
    if not command:
        return False
    lowered = command.replace("\\", "/").casefold()
    mock_adapters = [
        "examples/mock_executor_adapter.py",
        "examples/mock_judge_adapter.py",
        "examples/mock_relation_adapter.py",
    ]
    return any(adapter in lowered for adapter in mock_adapters)


def _has_gemini_key() -> bool:
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def enforce_strict_policy(
    spec: DTERunSpec,
    policy: StrictRunPolicy,
    cache_path: str | None,
    judge_command: str | None,
    executor_command: str | None = None,
) -> None:
    """Fail before a non-compliant slash-command run can start."""

    enforce_run_spec_guard(spec)

    if policy.require_cache_path and not cache_path:
        raise StrictRunError("strict-run requires --cache-path outside smoke mode")

    if not policy.allow_hash_geometry and spec.embedding_provider == "hash":
        raise StrictRunError("real strict-run forbids hash embedding geometry")

    if spec.embedding_provider == "gemini-embedding-2" and spec.embedding_dimension != 3072:
        raise StrictRunError("Gemini strict-run requires embedding_dimension=3072")

    if policy.require_gemini_key and spec.embedding_provider == "gemini-embedding-2" and not _has_gemini_key():
        raise StrictRunError("real strict-run with Gemini geometry requires GEMINI_API_KEY or GOOGLE_API_KEY")

    if not policy.allow_heuristic_judge and not judge_command:
        raise StrictRunError("real strict-run requires --judge-command for a real Judge oracle")

    if _is_mock_command(judge_command) and not policy.allow_mock:
        raise StrictRunError("mock Judge adapter is smoke-only and forbidden in this mode")

    if _is_mock_command(executor_command) and not policy.allow_mock:
        raise StrictRunError("mock Executor adapter is smoke-only and forbidden in this mode")


def write_run_artifacts(
    result: RunResult,
    out_dir: str | Path,
    strict_mode: StrictMode,
    control_path: str | Path | None = None,
    final: bool = True,
) -> None:
    """Write observable state, and write the synthesis report only at finalization."""

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    if final:
        report = result.report or synthesize_report(
            result.spec,
            result.nodes,
            forced_synthesis=result.forced_synthesis,
        )
        (out_path / "report.md").write_text(report, encoding="utf-8")
    (out_path / "nodes.json").write_text(
        json.dumps([n.model_dump() for n in result.nodes], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_path / "traces.json").write_text(
        json.dumps(
            [
                {
                    "iteration": t.iteration,
                    "notes": t.notes,
                    "allocations": [a.model_dump() for a in t.allocations],
                    "merges": [m.model_dump() for m in t.merges],
                    "entropy_state": None if t.entropy_state is None else t.entropy_state.__dict__,
                    "human_question": None if t.human_question is None else t.human_question.__dict__,
                }
                for t in result.traces
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_path / "cache_stats.json").write_text(
        json.dumps(result.cache.stats.__dict__ if hasattr(result.cache, "stats") else CacheStats().__dict__, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_path / "frontier.md").write_text(render_frontier_markdown(result), encoding="utf-8")
    (out_path / "entropy_trace.md").write_text(render_entropy_trace_markdown(result), encoding="utf-8")
    (out_path / "main_agent_status.md").write_text(render_main_agent_status(result), encoding="utf-8")
    (out_path / "checkpoint_summary.md").write_text(render_checkpoint_summary_markdown(result), encoding="utf-8")
    (out_path / "human_questions.md").write_text(render_human_questions_markdown(result), encoding="utf-8")
    (out_path / "role_audit.md").write_text(render_role_audit_markdown(result), encoding="utf-8")
    (out_path / "relation_candidates.md").write_text(render_relation_candidates_markdown(result), encoding="utf-8")
    (out_path / "strict_run_status.json").write_text(
        json.dumps(
            {
                "mode": strict_mode,
                "embedding_provider": result.spec.embedding_provider,
                "embedding_dimension": result.spec.embedding_dimension,
                "nodes": len(result.nodes),
                "traces": len(result.traces),
                "finalized": final,
                "stop_reason": result.stop_reason,
                "forced_synthesis": None
                if result.forced_synthesis is None
                else result.forced_synthesis.model_dump(),
                "control_path": None if control_path is None else str(control_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def strict_run(
    spec: DTERunSpec,
    mode: StrictMode,
    out_dir: str | Path,
    cache_path: str | None,
    initial_nodes: list[SearchNode] | None = None,
    judge_adapter: JudgeAdapter | None = None,
    judge_command: str | None = None,
    executor_adapter: ExecutorAdapter | None = None,
    executor_command: str | None = None,
    control_path: str | Path | None = None,
) -> RunResult:
    """Execute the locked DTE state machine used by slash-command skills."""

    policy = policy_for_mode(mode)
    enforce_strict_policy(
        spec,
        policy=policy,
        cache_path=cache_path,
        judge_command=judge_command,
        executor_command=executor_command,
    )

    if policy.allow_mock:
        os.environ["DTE_ALLOW_MOCK_ADAPTER"] = "1"

    cache = FileDTECache(cache_path) if cache_path else None
    def control_callback(spec: DTERunSpec, nodes: list[SearchNode], traces):
        return load_synthesis_control(control_path, nodes)

    def checkpoint_callback(result: RunResult) -> None:
        write_run_artifacts(
            result,
            out_dir=out_dir,
            strict_mode=mode,
            control_path=control_path,
            final=False,
        )

    result = run_frontier_search(
        spec,
        initial_nodes,
        executor_adapter=executor_adapter,
        judge_adapter=judge_adapter,
        cache=cache,
        control_callback=control_callback if control_path is not None else None,
        checkpoint_callback=checkpoint_callback,
        control_path=control_path,
    )
    write_run_artifacts(result, out_dir=out_dir, strict_mode=mode, control_path=control_path, final=True)
    return result
