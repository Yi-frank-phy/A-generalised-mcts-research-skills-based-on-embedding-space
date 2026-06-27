"""Run the current DTE workflow smoke checks.

Smoke mode is the only workflow that enables mock adapters. It uses the same
`strict-run` entrypoint as slash-command usage, but with `--mode smoke`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SMOKE_ENV = {**os.environ, "DTE_ALLOW_MOCK_ADAPTER": "1"}


def run(command: list[str]) -> None:
    print("$", " ".join(command))
    completed = subprocess.run(command, cwd=ROOT, text=True, env=SMOKE_ENV)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    out_dir = ROOT / "artifacts" / "smoke-workflow"
    relation_out_dir = out_dir / "relation"
    cache_path = ROOT / ".dte_cache" / "smoke_cache.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    relation_out_dir.mkdir(parents=True, exist_ok=True)
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
        "relation-artifacts",
        "--nodes",
        "examples/frontier_nodes.json",
        "--relation-output",
        "examples/relation_output.json",
        "--out-dir",
        str(relation_out_dir),
    ])
    run([
        sys.executable,
        "-m",
        "dte_backend",
        "strict-run",
        "--mode",
        "smoke",
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
        "relation_candidates.md",
        "strict_run_status.json",
    ]
    missing = [name for name in required if not (out_dir / name).exists()]
    relation_required = ["relation_proposals.json", "discriminator_tasks.json"]
    missing.extend([f"relation/{name}" for name in relation_required if not (relation_out_dir / name).exists()])
    if missing:
        raise SystemExit(f"missing smoke artifacts: {missing}")

    for name in relation_required:
        value = json.loads((relation_out_dir / name).read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise SystemExit(f"relation/{name} must be a JSON array")
    print("DTE smoke workflow ok")


if __name__ == "__main__":
    main()
