"""Real Codex Judge adapter for DTE strict-run.

The DTE backend calls this script through `--judge-command`. The script is a
thin bridge: it builds the Judge prompt, calls a real Codex command, validates
the returned JSON, and prints normalized Judge results. It does not score
locally and does not compute controller fields.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from dte_backend.models import SearchNode
from dte_backend.oracle_validation import validate_judge_output
from dte_backend.prompt_builder import build_cached_subagent_prompt


ROOT = Path(__file__).resolve().parents[1]


def configure_stdio() -> None:
    """Use UTF-8 for machine-facing JSON output on Windows."""

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def split_command(command: str) -> list[str]:
    """Split a configured command without corrupting Windows paths."""

    return shlex.split(command, posix=(os.name != "nt"))


def subprocess_env() -> dict[str, str]:
    """Force UTF-8 boundaries for Python-based oracle commands on Windows."""

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    official_codex_bin = Path.home() / "AppData" / "Local" / "Programs" / "OpenAI" / "Codex" / "bin"
    if official_codex_bin.is_dir():
        current_path = env.get("PATH", "")
        env["PATH"] = f"{official_codex_bin}{os.pathsep}{current_path}" if current_path else str(official_codex_bin)
    return env


def resolve_codex_executable() -> str:
    """Resolve Codex without hitting the WindowsApps execution alias."""

    official_codex = Path.home() / "AppData" / "Local" / "Programs" / "OpenAI" / "Codex" / "bin" / "codex.exe"
    if official_codex.is_file():
        return str(official_codex)

    resolved = shutil.which("codex", path=subprocess_env().get("PATH"))
    if resolved:
        return resolved
    return "codex"


def extract_json_text(text: str) -> str:
    """Extract a JSON object/list from a model response that may use fences."""

    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    try:
        json.loads(stripped)
        return stripped
    except json.JSONDecodeError:
        pass

    starts = [index for index in [stripped.find("{"), stripped.find("[")] if index != -1]
    if not starts:
        raise ValueError("Codex Judge response did not contain JSON")
    start = min(starts)
    for end in range(len(stripped), start, -1):
        candidate = stripped[start:end].strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue
    raise ValueError("Codex Judge response did not contain valid JSON")


def judge_output_schema(nodes: list[SearchNode]) -> dict[str, Any]:
    """Build a per-call schema that restricts Judge results to known nodes."""

    node_ids = [node.node_id for node in nodes]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["results"],
        "properties": {
            "results": {
                "type": "array",
                "minItems": len(node_ids),
                "maxItems": len(node_ids),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["node_id", "score", "reasoning", "risks"],
                    "properties": {
                        "node_id": {"type": "string", "enum": node_ids},
                        "score": {"type": "number", "minimum": 0, "maximum": 1},
                        "reasoning": {"type": "string"},
                        "risks": {"type": "array", "items": {"type": "string"}},
                    },
                },
            }
        },
    }


def run_configured_command(prompt: str, nodes: list[SearchNode]) -> str:
    configured = os.getenv("DTE_CODEX_JUDGE_COMMAND")
    if configured:
        command = split_command(configured)
        completed = subprocess.run(
            command,
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=ROOT,
            check=False,
            env=subprocess_env(),
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
        return completed.stdout

    with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False, suffix=".json") as handle:
        output_path = Path(handle.name)
    with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False, suffix=".schema.json") as handle:
        schema_path = Path(handle.name)
        json.dump(judge_output_schema(nodes), handle, ensure_ascii=False, indent=2)

    command = [
        resolve_codex_executable(),
        "exec",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(output_path),
        "-",
    ]
    completed = subprocess.run(
        command,
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=ROOT,
        check=False,
        env=subprocess_env(),
    )
    try:
        final_message = output_path.read_text(encoding="utf-8").strip()
    finally:
        output_path.unlink(missing_ok=True)
        schema_path.unlink(missing_ok=True)

    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    return final_message or completed.stdout


def normalized_results(raw_output: str, nodes: list[SearchNode]) -> dict[str, Any]:
    json_text = extract_json_text(raw_output)
    results = validate_judge_output(nodes, json_text)
    return {"results": [result.__dict__ for result in results]}


def main() -> int:
    configure_stdio()
    try:
        payload = json.loads(sys.stdin.read())
        nodes = [SearchNode.model_validate(item) for item in payload.get("nodes", [])]
        if not nodes:
            raise ValueError("Codex Judge adapter requires at least one SearchNode")

        prompt = build_cached_subagent_prompt("judge", payload, repo_root=ROOT)
        raw_output = run_configured_command(prompt, nodes)
        print(json.dumps(normalized_results(raw_output, nodes), ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"Codex Judge adapter failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
