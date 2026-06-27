import subprocess
import sys
import json


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


def test_prompt_guard_injects_readable_chinese_dte_reminder():
    payload = {
        "cwd": r"C:\Users\zhaoy\Downloads\dte-codex-skill-backend",
        "prompt": "请运行 dte-extreme-research",
    }

    completed = subprocess.run(
        [sys.executable, "hooks/dte_prompt_guard.py"],
        input=json.dumps(payload, ensure_ascii=False),
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    output = json.loads(completed.stdout)
    reminder = output["hookSpecificOutput"]["additionalContext"]
    assert "必须先使用已安装的 `dte-extreme-research` skill" in reminder
    assert "任何 guard 失败都必须停止消费该产物" in reminder
    assert chr(0xFFFD) not in reminder


def test_dte_guard_rejects_bad_gemini_dimension(tmp_path):
    bad_spec = json.loads(open("examples/run_spec.json", encoding="utf-8").read())
    bad_spec["embedding_provider"] = "gemini-embedding-2"
    bad_spec["embedding_dimension"] = 128
    spec_path = tmp_path / "bad_spec.json"
    spec_path.write_text(json.dumps(bad_spec), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, "hooks/dte_guard.py", "spec", str(spec_path)],
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "Gemini geometry must use embedding_dimension=3072" in completed.stderr
