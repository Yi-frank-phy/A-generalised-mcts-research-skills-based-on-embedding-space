"""DTE guard hook.

This script is intended for Codex hooks or CI-style checks. It is deliberately
small: it validates machine-facing artifacts and fails fast if an agent tries to
bypass DTE boundaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dte_backend.adapter import validate_adapter_output
from dte_backend.guards import enforce_run_spec_guard
from dte_backend.models import DTERunSpec, SearchNode
from dte_backend.oracle_validation import validate_judge_output, validate_relation_output


def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cmd_spec(args: argparse.Namespace) -> None:
    spec = DTERunSpec.model_validate(load_json(args.path))
    enforce_run_spec_guard(spec)
    print("DTE guard ok: run spec")


def cmd_executor(args: argparse.Namespace) -> None:
    parent = SearchNode.model_validate(load_json(args.parent))
    raw_output = load_json(args.output)
    validate_adapter_output(parent, args.child_count, raw_output)
    print("DTE guard ok: executor output")


def cmd_judge(args: argparse.Namespace) -> None:
    nodes = [SearchNode.model_validate(item) for item in load_json(args.nodes)]
    raw_output = load_json(args.output)
    validate_judge_output(nodes, raw_output)
    print("DTE guard ok: judge output")


def cmd_relation(args: argparse.Namespace) -> None:
    nodes = [SearchNode.model_validate(item) for item in load_json(args.nodes)]
    raw_output = load_json(args.output)
    validate_relation_output(nodes, raw_output)
    print("DTE guard ok: relation output")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DTE artifact guard")
    sub = parser.add_subparsers(required=True)

    spec = sub.add_parser("spec")
    spec.add_argument("path")
    spec.set_defaults(func=cmd_spec)

    executor = sub.add_parser("executor")
    executor.add_argument("--parent", required=True)
    executor.add_argument("--output", required=True)
    executor.add_argument("--child-count", type=int, required=True)
    executor.set_defaults(func=cmd_executor)

    judge = sub.add_parser("judge")
    judge.add_argument("--nodes", required=True)
    judge.add_argument("--output", required=True)
    judge.set_defaults(func=cmd_judge)

    relation = sub.add_parser("relation")
    relation.add_argument("--nodes", required=True)
    relation.add_argument("--output", required=True)
    relation.set_defaults(func=cmd_relation)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
