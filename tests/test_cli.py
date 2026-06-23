import json
import subprocess
import sys


def test_run_command_accepts_judge_adapter(tmp_path):
    out_dir = tmp_path / "judge-run"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "dte_backend",
            "run",
            "--spec",
            "examples/run_spec.json",
            "--nodes",
            "examples/frontier_nodes.json",
            "--out-dir",
            str(out_dir),
            "--judge-command",
            f"{sys.executable} examples/mock_judge_adapter.py",
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    nodes = json.loads((out_dir / "nodes.json").read_text(encoding="utf-8"))
    judged = [node for node in nodes if node["node_id"] in {"n1", "n2"}]
    assert judged
    assert all(node["judge_reasoning"].startswith("mock judge") for node in judged)
