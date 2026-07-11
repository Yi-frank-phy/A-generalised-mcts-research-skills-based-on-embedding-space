import json
import subprocess
import sys
from pathlib import Path

from dte_backend.models import SearchNode
from dte_backend.prompt_builder import load_static_prefix
from scripts import codex_judge_adapter


def test_resolve_codex_executable_prefers_official_install(monkeypatch, tmp_path):
    home = tmp_path / "home"
    official = home / "AppData" / "Local" / "Programs" / "OpenAI" / "Codex" / "bin" / "codex.exe"
    official.parent.mkdir(parents=True)
    official.write_text("", encoding="utf-8")
    windowsapps = tmp_path / "WindowsApps"
    windowsapps.mkdir()
    (windowsapps / "codex.exe").write_text("", encoding="utf-8")

    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("PATH", str(windowsapps))

    assert codex_judge_adapter.resolve_codex_executable() == str(official)


def test_codex_judge_adapter_invokes_configured_command_and_validates_json(tmp_path, monkeypatch):
    fake_codex = tmp_path / "fake_codex.py"
    prompt_capture = tmp_path / "prompt.txt"
    fake_codex.write_text(
        "\n".join(
            [
                "import json, pathlib, sys",
                f"pathlib.Path({str(prompt_capture)!r}).write_text(sys.stdin.read(), encoding='utf-8')",
                "print(json.dumps({'results': [",
                "  {'node_id': 'n1', 'score': 0.8, 'reasoning': 'coherent route', 'risks': []},",
                "  {'node_id': 'n2', 'score': 0.4, 'reasoning': 'weaker evidence', 'risks': ['thin evidence']}",
                "]}))",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DTE_CODEX_JUDGE_COMMAND", f"{sys.executable} {fake_codex}")
    payload = {
        "task": {"kind": "judge"},
        "nodes": [
            SearchNode(node_id="n1", claim="First route").model_dump(),
            SearchNode(node_id="n2", claim="Second route").model_dump(),
        ],
    }

    completed = subprocess.run(
        [sys.executable, "scripts/codex_judge_adapter.py"],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    output = json.loads(completed.stdout)
    assert [item["node_id"] for item in output["results"]] == ["n1", "n2"]
    prompt = prompt_capture.read_text(encoding="utf-8")
    assert prompt.startswith(load_static_prefix())
    assert "# Judge Oracle Subagent Prompt" in prompt
    assert prompt.rfind('"nodes"') > prompt.find("# Judge Oracle Subagent Prompt")


def test_codex_judge_adapter_rejects_forbidden_judge_fields(tmp_path, monkeypatch):
    fake_codex = tmp_path / "fake_codex.py"
    fake_codex.write_text(
        "import json\n"
        "print(json.dumps({'results': [{'node_id': 'n1', 'score': 0.8, 'reasoning': 'bad', 'ucb_score': 9}]}))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DTE_CODEX_JUDGE_COMMAND", f"{sys.executable} {fake_codex}")
    payload = {
        "task": {"kind": "judge"},
        "nodes": [SearchNode(node_id="n1", claim="First route").model_dump()],
    }

    completed = subprocess.run(
        [sys.executable, "scripts/codex_judge_adapter.py"],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "forbidden fields" in completed.stderr


def test_codex_judge_adapter_can_be_used_as_judge_oracle_command(tmp_path, monkeypatch):
    fake_codex = tmp_path / "fake_codex.py"
    fake_codex.write_text(
        "\n".join(
            [
                "import json, sys",
                "prompt = sys.stdin.read()",
                "payload = json.loads(prompt.split('```json')[-1].split('```')[0])",
                "print(json.dumps({'results': [",
                "  {'node_id': node['node_id'], 'score': 0.5, 'reasoning': 'real command bridge exercised', 'risks': []}",
                "  for node in payload['nodes']",
                "]}))",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DTE_CODEX_JUDGE_COMMAND", f"{sys.executable} {fake_codex}")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "dte_backend",
            "judge-oracle",
            "--nodes",
            "examples/frontier_nodes.json",
            "--judge-command",
            f"{sys.executable} scripts/codex_judge_adapter.py",
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    output = json.loads(completed.stdout)
    assert [item["node_id"] for item in output["results"]] == ["n1", "n2"]
