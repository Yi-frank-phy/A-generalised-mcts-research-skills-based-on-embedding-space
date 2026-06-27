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


def test_run_command_rejects_spec_that_fails_dte_guard(tmp_path):
    bad_spec = json.loads(open("examples/run_spec.json", encoding="utf-8").read())
    bad_spec["embedding_provider"] = "gemini-embedding-2"
    bad_spec["embedding_dimension"] = 128
    spec_path = tmp_path / "bad_spec.json"
    spec_path.write_text(json.dumps(bad_spec), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "dte_backend",
            "run",
            "--spec",
            str(spec_path),
            "--out-dir",
            str(tmp_path / "bad-run"),
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "Gemini geometry must use embedding_dimension=3072" in completed.stderr
