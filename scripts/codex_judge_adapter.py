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
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from dte_backend.models import SearchNode
from dte_backend.oracle_validation import validate_judge_output
from dte_backend.prompt_builder import build_cached_subagent_prompt


ROOT = Path(__file__).resolve().parents[1]


def split_command(command: str) -> list[str]:
    """Split a configured command without corrupting Windows paths."""

    return shlex.split(command, posix=(os.name != "nt"))


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


def run_configured_command(prompt: str) -> str:
    configured = os.getenv("DTE_CODEX_JUDGE_COMMAND")
    if configured:
        command = split_command(configured)
        completed = subprocess.run(
            command,
            input=prompt,
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
        return completed.stdout

    with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False, suffix=".json") as handle:
        output_path = Path(handle.name)

    command = [
        "codex",
        "exec",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--output-last-message",
        str(output_path),
        "-",
    ]
    completed = subprocess.run(
        command,
        input=prompt,
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    try:
        final_message = output_path.read_text(encoding="utf-8").strip()
    finally:
        output_path.unlink(missing_ok=True)

    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    return final_message or completed.stdout


def normalized_results(raw_output: str, nodes: list[SearchNode]) -> dict[str, Any]:
    json_text = extract_json_text(raw_output)
    results = validate_judge_output(nodes, json_text)
    return {"results": [result.__dict__ for result in results]}


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
        nodes = [SearchNode.model_validate(item) for item in payload.get("nodes", [])]
        if not nodes:
            raise ValueError("Codex Judge adapter requires at least one SearchNode")

        prompt = build_cached_subagent_prompt("judge", payload, repo_root=ROOT)
        raw_output = run_configured_command(prompt)
        print(json.dumps(normalized_results(raw_output, nodes), ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"Codex Judge adapter failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
