import subprocess
import sys


def test_dte_guard_accepts_sample_artifacts():
    commands = [
        [sys.executable, "hooks/dte_guard.py", "spec", "examples/run_spec.json"],
        [
            sys.executable,
            "hooks/dte_guard.py",
            "executor",
            "--parent",
            "examples/executor_parent.json",
            "--output",
            "examples/executor_output.json",
            "--child-count",
            "1",
        ],
        [
            sys.executable,
            "hooks/dte_guard.py",
            "judge",
            "--nodes",
            "examples/frontier_nodes.json",
            "--output",
            "examples/judge_output.json",
        ],
        [
            sys.executable,
            "hooks/dte_guard.py",
            "relation",
            "--nodes",
            "examples/frontier_nodes.json",
            "--output",
            "examples/relation_output.json",
        ],
    ]

    for command in commands:
        completed = subprocess.run(command, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
