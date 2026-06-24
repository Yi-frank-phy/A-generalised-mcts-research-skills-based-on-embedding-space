"""Run the current DTE workflow smoke checks.

This is intentionally simple and cross-platform. It exercises the pieces Codex
should care about before doing broader work.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("$", " ".join(command))
    completed = subprocess.run(command, cwd=ROOT, text=True)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    out_dir = ROOT / "artifacts" / "smoke-workflow"
    cache_path = ROOT / ".dte_cache" / "smoke_cache.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    run([sys.executable, "hooks/dte_guard.py", "spec", "examples/run_spec.json"])
    run([
        sys.executable,
        "-m",
        "dte_backend",
        "judge-oracle",
        "--nodes",
        "examples/frontier_nodes.json",
        "--judge-command",
        f"{sys.executable} examples/mock_judge_adapter.py",
    ])
    run([
        sys.executable,
        "-m",
        "dte_backend",
        "relation-oracle",
        "--nodes",
        "examples/frontier_nodes.json",
        "--relation-command",
        f"{sys.executable} examples/mock_relation_adapter.py",
    ])
    run([
        sys.executable,
        "-m",
        "dte_backend",
        "run",
        "--spec",
        "examples/run_spec.json",
        "--out-dir",
        str(out_dir),
        "--cache-path",
        str(cache_path),
        "--judge-command",
        f"{sys.executable} examples/mock_judge_adapter.py",
    ])

    required = [
        "report.md",
        "nodes.json",
        "traces.json",
        "frontier.md",
        "entropy_trace.md",
        "main_agent_status.md",
        "human_questions.md",
        "role_audit.md",
    ]
    missing = [name for name in required if not (out_dir / name).exists()]
    if missing:
        raise SystemExit(f"missing smoke artifacts: {missing}")
    print("DTE smoke workflow ok")


if __name__ == "__main__":
    main()
