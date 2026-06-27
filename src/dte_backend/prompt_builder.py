"""Prefix-cache friendly prompt builder.

LLM prompt/prefix caches work best when every request shares the exact same
initial prefix. This helper builds prompts in the required order:

1. shared static prefix;
2. role-specific static contract;
3. dynamic JSON payload at the end.

The prompts live at the repository/skill root, not inside the Python package.
For public use, callers may set `DTE_REPO_ROOT` or pass `repo_root` explicitly.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal


RoleName = Literal["judge", "relation", "executor"]


PROMPT_FILES: dict[RoleName, str] = {
    "judge": "judge_oracle.md",
    "relation": "relation_oracle.md",
    "executor": "executor_subagent.md",
}


def _candidate_roots(repo_root: str | Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    if repo_root is not None:
        candidates.append(Path(repo_root))
    env_root = os.getenv("DTE_REPO_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(Path.cwd())
    # Editable installs from this repository resolve to <repo>/src/dte_backend.
    candidates.append(Path(__file__).resolve().parents[2])
    return candidates


def prompts_dir(repo_root: str | Path | None = None) -> Path:
    """Return the prompts directory.

    Resolution order:
    1. explicit `repo_root`;
    2. `DTE_REPO_ROOT`;
    3. current working directory;
    4. editable-source repository root.
    """

    for root in _candidate_roots(repo_root):
        prompt_path = root / "prompts"
        if (prompt_path / "DTE_STATIC_PREFIX.md").exists():
            return prompt_path
    roots = ", ".join(str(path) for path in _candidate_roots(repo_root))
    raise FileNotFoundError(
        "Could not locate prompts/DTE_STATIC_PREFIX.md. Run from the skill repository root, "
        "set DTE_REPO_ROOT, or pass repo_root explicitly. Searched: " + roots
    )


def load_static_prefix(repo_root: str | Path | None = None) -> str:
    return (prompts_dir(repo_root) / "DTE_STATIC_PREFIX.md").read_text(encoding="utf-8").strip()


def load_role_contract(role: RoleName, repo_root: str | Path | None = None) -> str:
    return (prompts_dir(repo_root) / PROMPT_FILES[role]).read_text(encoding="utf-8").strip()


def build_cached_subagent_prompt(
    role: RoleName,
    dynamic_payload: dict[str, Any],
    repo_root: str | Path | None = None,
) -> str:
    """Build a prompt with a stable prefix and dynamic JSON at the end."""

    static_prefix = load_static_prefix(repo_root)
    role_contract = load_role_contract(role, repo_root)
    dynamic_json = json.dumps(dynamic_payload, ensure_ascii=False, indent=2, sort_keys=True)
    return (
        f"{static_prefix}\n\n"
        f"{role_contract}\n\n"
        "# Dynamic task input — variable, intentionally last\n\n"
        "```json\n"
        f"{dynamic_json}\n"
        "```\n"
    )
