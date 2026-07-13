"""Tiny CLI intended for agents/hooks, not for human-heavy operation."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

from .adapter import build_subprocess_adapter, run_subprocess_executor
from .app_driver import (
    app_run_status,
    cancel_app_episode,
    create_app_run,
    fail_app_episode,
    next_app_episode,
    request_app_synthesis,
    retry_app_episode,
    submit_app_episode_result,
)
from .artifacts import (
    render_entropy_trace_markdown,
    render_frontier_markdown,
    render_human_questions_markdown,
    render_main_agent_status,
    render_relation_candidates_markdown,
    render_role_audit_markdown,
)
from .control import OperatorAuthorizationError
from .file_cache import FileDTECache
from .guards import enforce_run_spec_guard
from .math_engine import allocate_frontier
from .episode_models import RuntimeLimits
from .models import DTERunSpec, ExpansionRequest, SearchNode, SynthesisControlRequest
from .oracle_validation import validate_relation_output
from .relation_workflow import relation_result_to_outputs
from .runner import run_frontier_search
from .strict_runner import StrictRunError, strict_run
from .subprocess_oracles import build_subprocess_judge_adapter, run_subprocess_judge, run_subprocess_relation
from .validators import load_json_list, load_json_model


def configure_stdio() -> None:
    """Use UTF-8 for machine-facing CLI output on Windows."""

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def split_command(command: str) -> list[str]:
    """Split command text without corrupting Windows absolute paths."""

    return shlex.split(command, posix=(os.name != "nt"))


def cmd_validate(args: argparse.Namespace) -> None:
    spec = load_json_model(args.path, DTERunSpec)
    enforce_run_spec_guard(spec)
    print(spec.model_dump_json(indent=2))


def cmd_allocate(args: argparse.Namespace) -> None:
    nodes = load_json_list(args.path, SearchNode)
    allocations = allocate_frontier(
        nodes,
        allocation_mass_per_iteration=args.allocation_mass_per_iteration,
        max_children_per_iteration=args.max_children_per_iteration,
        tau=args.tau,
        c_explore=args.c_explore,
        temperature=args.temperature,
        allocation_metric=args.allocation_metric,
    )
    print(json.dumps([a.model_dump() for a in allocations], ensure_ascii=False, indent=2))


def cmd_judge_oracle(args: argparse.Namespace) -> None:
    nodes = load_json_list(args.nodes, SearchNode)
    results = run_subprocess_judge(split_command(args.judge_command), nodes, timeout=args.timeout)
    print(json.dumps({"results": [r.__dict__ for r in results]}, ensure_ascii=False, indent=2))


def cmd_relation_oracle(args: argparse.Namespace) -> None:
    nodes = load_json_list(args.nodes, SearchNode)
    result = run_subprocess_relation(split_command(args.relation_command), nodes, timeout=args.timeout)
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))


def cmd_relation_artifacts(args: argparse.Namespace) -> None:
    nodes = load_json_list(args.nodes, SearchNode)
    raw = json.loads(Path(args.relation_output).read_text(encoding="utf-8"))
    relation = validate_relation_output(nodes, raw)
    proposal, discriminator = relation_result_to_outputs(relation, nodes)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    proposals = [] if proposal is None else [proposal.model_dump()]
    tasks = [] if discriminator is None else [discriminator.model_dump()]
    (out_dir / "relation_proposals.json").write_text(
        json.dumps(proposals, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "discriminator_tasks.json").write_text(
        json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"relation_proposals": len(proposals), "discriminator_tasks": len(tasks)}, ensure_ascii=False))


def cmd_validate_executor(args: argparse.Namespace) -> None:
    request = load_json_model(args.request, ExpansionRequest)
    children = run_subprocess_executor(
        split_command(args.executor_command),
        request,
        timeout=args.timeout,
    )
    print(json.dumps({"nodes": [node.model_dump() for node in children]}, ensure_ascii=False, indent=2))


def write_standard_artifacts(result, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(result.report, encoding="utf-8")
    (out_dir / "nodes.json").write_text(
        json.dumps([n.model_dump() for n in result.nodes], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "traces.json").write_text(
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
    (out_dir / "cache_stats.json").write_text(
        json.dumps(result.cache.stats.__dict__, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "frontier.md").write_text(render_frontier_markdown(result), encoding="utf-8")
    (out_dir / "entropy_trace.md").write_text(render_entropy_trace_markdown(result), encoding="utf-8")
    (out_dir / "main_agent_status.md").write_text(render_main_agent_status(result), encoding="utf-8")
    (out_dir / "human_questions.md").write_text(render_human_questions_markdown(result), encoding="utf-8")
    (out_dir / "role_audit.md").write_text(render_role_audit_markdown(result), encoding="utf-8")
    (out_dir / "relation_candidates.md").write_text(render_relation_candidates_markdown(result), encoding="utf-8")


def cmd_run(args: argparse.Namespace) -> None:
    spec = load_json_model(args.spec, DTERunSpec)
    enforce_run_spec_guard(spec)
    nodes = load_json_list(args.nodes, SearchNode) if args.nodes else None
    executor_adapter = None
    if args.executor_command:
        executor_adapter = build_subprocess_adapter(split_command(args.executor_command), timeout=args.executor_timeout)
    judge_adapter = None
    if args.judge_command:
        judge_adapter = build_subprocess_judge_adapter(split_command(args.judge_command), timeout=args.judge_timeout)
    cache = FileDTECache(args.cache_path) if args.cache_path else None
    result = run_frontier_search(
        spec,
        nodes,
        executor_adapter=executor_adapter,
        judge_adapter=judge_adapter,
        cache=cache,
    )

    out_dir = Path(args.out_dir)
    write_standard_artifacts(result, out_dir)
    print(json.dumps({"out_dir": str(out_dir), "nodes": len(result.nodes), "traces": len(result.traces)}, ensure_ascii=False))


def cmd_strict_run(args: argparse.Namespace) -> None:
    spec = load_json_model(args.spec, DTERunSpec)
    nodes = load_json_list(args.nodes, SearchNode) if args.nodes else None
    control_path = args.control_path or str(Path(args.out_dir) / "strict_run_control.json")
    executor_adapter = None
    if args.executor_command:
        executor_adapter = build_subprocess_adapter(split_command(args.executor_command), timeout=args.executor_timeout)
    judge_adapter = None
    if args.judge_command:
        judge_adapter = build_subprocess_judge_adapter(split_command(args.judge_command), timeout=args.judge_timeout)
    try:
        result = strict_run(
            spec=spec,
            mode=args.mode,
            out_dir=args.out_dir,
            cache_path=args.cache_path,
            initial_nodes=nodes,
            judge_adapter=judge_adapter,
            judge_command=args.judge_command,
            executor_adapter=executor_adapter,
            executor_command=args.executor_command,
            control_path=control_path,
        )
    except (StrictRunError, OperatorAuthorizationError) as exc:
        raise SystemExit(f"strict-run failed: {exc}") from exc
    print(json.dumps({"out_dir": str(args.out_dir), "mode": args.mode, "nodes": len(result.nodes), "traces": len(result.traces)}, ensure_ascii=False))


def _print_model(value) -> None:
    print(value.model_dump_json(indent=2))


def cmd_create_run(args: argparse.Namespace) -> None:
    spec = load_json_model(args.spec, DTERunSpec)
    enforce_run_spec_guard(spec)
    nodes = load_json_list(args.nodes, SearchNode)
    _print_model(create_app_run(args.run_dir, spec, nodes, run_id=args.run_id))


def cmd_next_episode(args: argparse.Namespace) -> None:
    limits = RuntimeLimits(
        wall_clock_seconds=args.wall_clock_seconds,
        max_retries=args.max_retries,
        selected_by=args.selected_by,
    )
    _print_model(next_app_episode(args.run_dir, runtime_limits=limits, profile=args.profile))


def cmd_submit_episode_result(args: argparse.Namespace) -> None:
    raw = json.loads(Path(args.result).read_text(encoding="utf-8"))
    _print_model(submit_app_episode_result(args.run_dir, raw))


def cmd_fail_episode(args: argparse.Namespace) -> None:
    _print_model(fail_app_episode(args.run_dir, args.episode_id, args.attempt_id, args.reason))


def cmd_cancel_episode(args: argparse.Namespace) -> None:
    _print_model(cancel_app_episode(args.run_dir, args.episode_id, args.attempt_id, args.reason))


def cmd_retry_episode(args: argparse.Namespace) -> None:
    _print_model(
        retry_app_episode(
            args.run_dir,
            args.episode_id,
            selected_by=args.selected_by,
            wall_clock_seconds=args.wall_clock_seconds,
        )
    )


def cmd_request_synthesis(args: argparse.Namespace) -> None:
    request = load_json_model(args.request, SynthesisControlRequest)
    _print_model(request_app_synthesis(args.run_dir, request))


def cmd_run_status(args: argparse.Namespace) -> None:
    _print_model(app_run_status(args.run_dir))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DTE backend helper")
    sub = parser.add_subparsers(required=True)

    validate = sub.add_parser("validate", help="validate a DTE run spec JSON file")
    validate.add_argument("path")
    validate.set_defaults(func=cmd_validate)

    allocate = sub.add_parser("allocate", help="allocate expansion budget for frontier nodes")
    allocate.add_argument("path")
    allocate.add_argument(
        "--allocation-mass",
        "--budget",
        dest="allocation_mass_per_iteration",
        type=int,
        default=3,
        help="soft allocation mass per iteration (--budget is deprecated)",
    )
    allocate.add_argument("--max-children", dest="max_children_per_iteration", type=int, default=5)
    allocate.add_argument("--tau", type=float, default=1.0)
    allocate.add_argument("--c-explore", type=float, default=1.0)
    allocate.add_argument("--temperature", type=float, default=1.0)
    allocate.add_argument("--allocation-metric", choices=["ucb", "score"], default="ucb")
    allocate.set_defaults(func=cmd_allocate)

    judge = sub.add_parser("judge-oracle", help="run and validate a Judge oracle command")
    judge.add_argument("--nodes", required=True)
    judge.add_argument("--judge-command", required=True)
    judge.add_argument("--timeout", type=float, default=360.0)
    judge.set_defaults(func=cmd_judge_oracle)

    relation = sub.add_parser("relation-oracle", help="run and validate a relation oracle command")
    relation.add_argument("--nodes", required=True)
    relation.add_argument("--relation-command", required=True)
    relation.add_argument("--timeout", type=float, default=360.0)
    relation.set_defaults(func=cmd_relation_oracle)

    relation_artifacts = sub.add_parser("relation-artifacts", help="convert a relation oracle result into machine artifacts")
    relation_artifacts.add_argument("--nodes", required=True)
    relation_artifacts.add_argument("--relation-output", required=True)
    relation_artifacts.add_argument("--out-dir", required=True)
    relation_artifacts.set_defaults(func=cmd_relation_artifacts)

    validate_executor = sub.add_parser("validate-executor", help="run and validate an executor adapter command")
    validate_executor.add_argument("--request", required=True)
    validate_executor.add_argument("--executor-command", required=True)
    validate_executor.add_argument("--timeout", type=float, default=120.0)
    validate_executor.set_defaults(func=cmd_validate_executor)

    run = sub.add_parser("run", help="run the flexible DTE backend helper")
    run.add_argument("--spec", required=True, help="DTE run spec JSON")
    run.add_argument("--nodes", help="optional initial SearchNode JSON list")
    run.add_argument("--out-dir", default="artifacts/run", help="directory for report/nodes/traces")
    run.add_argument("--executor-command", help="optional subprocess executor adapter command")
    run.add_argument("--executor-timeout", type=float, default=120.0)
    run.add_argument("--judge-command", help="optional subprocess Judge oracle command")
    run.add_argument("--judge-timeout", type=float, default=360.0)
    run.add_argument("--cache-path", help="optional JSON cache path for embeddings and scores")
    run.set_defaults(func=cmd_run)

    strict = sub.add_parser("strict-run", help="run the locked slash-command DTE workflow")
    strict.add_argument("--spec", required=True, help="DTE run spec JSON")
    strict.add_argument("--mode", choices=["smoke", "dry-run", "real"], default="real")
    strict.add_argument("--nodes", help="optional initial SearchNode JSON list")
    strict.add_argument("--out-dir", default="artifacts/strict", help="directory for artifacts")
    strict.add_argument("--cache-path", help="required outside smoke mode")
    strict.add_argument("--executor-command", help="optional subprocess executor adapter command")
    strict.add_argument("--executor-timeout", type=float, default=120.0)
    strict.add_argument("--judge-command", help="required in real mode")
    strict.add_argument("--judge-timeout", type=float, default=360.0)
    strict.add_argument(
        "--control-path",
        help="optional operator control JSON path; defaults to <out-dir>/strict_run_control.json",
    )
    strict.set_defaults(func=cmd_strict_run)

    create = sub.add_parser("create-run", help="create persistent state for a Codex App-driven DTE run")
    create.add_argument("--run-dir", required=True)
    create.add_argument("--spec", required=True)
    create.add_argument("--nodes", required=True, help="committed initial SearchNode JSON list")
    create.add_argument("--run-id")
    create.set_defaults(func=cmd_create_run)

    next_episode = sub.add_parser("next-episode", help="grant or resume one bounded App-native episode")
    next_episode.add_argument("--run-dir", required=True)
    next_episode.add_argument("--wall-clock-seconds", type=int)
    next_episode.add_argument("--max-retries", type=int, default=1)
    next_episode.add_argument(
        "--selected-by",
        choices=["user", "main_agent", "run_default"],
        default="run_default",
    )
    next_episode.add_argument(
        "--profile",
        choices=["legacy-explicit", "native-guided", "native-autonomous"],
        default="native-autonomous",
    )
    next_episode.set_defaults(func=cmd_next_episode)

    submit = sub.add_parser("submit-episode-result", help="validate and commit one complete EpisodeResult")
    submit.add_argument("--run-dir", required=True)
    submit.add_argument("--result", required=True)
    submit.set_defaults(func=cmd_submit_episode_result)

    fail = sub.add_parser("fail-episode", help="record failure of the active App-native attempt")
    fail.add_argument("--run-dir", required=True)
    fail.add_argument("--episode-id", required=True)
    fail.add_argument("--attempt-id", required=True)
    fail.add_argument("--reason", required=True)
    fail.set_defaults(func=cmd_fail_episode)

    cancel = sub.add_parser("cancel-episode", help="cancel the active App-native attempt")
    cancel.add_argument("--run-dir", required=True)
    cancel.add_argument("--episode-id", required=True)
    cancel.add_argument("--attempt-id", required=True)
    cancel.add_argument("--reason", required=True)
    cancel.set_defaults(func=cmd_cancel_episode)

    retry = sub.add_parser("retry-episode", help="supersede an attempt and grant a new attempt ID")
    retry.add_argument("--run-dir", required=True)
    retry.add_argument("--episode-id", required=True)
    retry.add_argument("--wall-clock-seconds", type=int)
    retry.add_argument(
        "--selected-by",
        choices=["user", "main_agent", "run_default"],
        default="main_agent",
    )
    retry.set_defaults(func=cmd_retry_episode)

    synthesis = sub.add_parser("request-synthesis", help="submit an OperatorPolicy-validated synthesis request")
    synthesis.add_argument("--run-dir", required=True)
    synthesis.add_argument("--request", required=True)
    synthesis.set_defaults(func=cmd_request_synthesis)

    status = sub.add_parser("run-status", help="read persistent App-native run and lifecycle state")
    status.add_argument("--run-dir", required=True)
    status.set_defaults(func=cmd_run_status)

    return parser


def main() -> None:
    configure_stdio()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
