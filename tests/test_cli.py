import json
import os
import subprocess
import sys


def test_run_command_accepts_judge_adapter(tmp_path):
    out_dir = tmp_path / "judge-run"
    env = {**os.environ, "DTE_ALLOW_MOCK_ADAPTER": "1"}
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
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    nodes = json.loads((out_dir / "nodes.json").read_text(encoding="utf-8"))
    judged = [node for node in nodes if node["node_id"] in {"n1", "n2"}]
    assert judged
    assert all(node["judge_reasoning"].startswith("SMOKE-ONLY mock judge") for node in judged)


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


def test_strict_run_rejects_legacy_main_agent_control_path(tmp_path):
    out_dir = tmp_path / "strict-control"
    control_path = tmp_path / "strict_run_control.json"
    control_path.write_text(
        json.dumps(
            {
                "action": "force_synthesis_after_current_task",
                "requested_by": "main_agent",
                "reason": "checkpoint has enough coverage",
                "scope": "all",
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "dte_backend",
            "strict-run",
            "--mode",
            "smoke",
            "--spec",
            "examples/run_spec.json",
            "--nodes",
            "examples/frontier_nodes.json",
            "--out-dir",
            str(out_dir),
            "--control-path",
            str(control_path),
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "requested_by" in completed.stderr
    status = json.loads((out_dir / "strict_run_status.json").read_text(encoding="utf-8"))
    assert status["stop_reason"] is None
    assert status["forced_synthesis"] is None
    assert status["finalized"] is False
    assert status["control_path"] == str(control_path)
    assert (out_dir / "checkpoint_summary.md").exists()


def test_strict_run_uses_default_control_path_in_out_dir(tmp_path):
    out_dir = tmp_path / "strict-default-control"
    out_dir.mkdir()
    control_path = out_dir / "strict_run_control.json"
    control_path.write_text(
        json.dumps(
            {
                "action": "force_synthesis_after_current_task",
                "requested_by": "user",
                "reason": "user interrupted the main agent",
                "scope": "all",
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "dte_backend",
            "strict-run",
            "--mode",
            "smoke",
            "--spec",
            "examples/run_spec.json",
            "--nodes",
            "examples/frontier_nodes.json",
            "--out-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    status = json.loads((out_dir / "strict_run_status.json").read_text(encoding="utf-8"))
    assert status["stop_reason"] == "user_interrupted_for_synthesis"
    assert status["finalized"] is True
    assert status["control_path"] == str(control_path)


def test_strict_run_help_labels_control_file_as_user_authored():
    completed = subprocess.run(
        [sys.executable, "-m", "dte_backend", "strict-run", "--help"],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "user-authored" in completed.stdout
    assert "main agent" not in completed.stdout.casefold()
