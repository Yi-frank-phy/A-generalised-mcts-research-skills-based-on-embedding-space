"""Prefix-cache friendly prompt builder.

LLM prompt/prefix caches work best when every request shares the exact same
initial prefix. This helper builds prompts in the required order:

1. shared static prefix;
2. role-specific static contract;
3. dynamic JSON payload at the end.

It is intentionally small and file-based so Codex can inspect and reuse it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal


RoleName = Literal["judge", "relation", "executor"]


PROMPT_FILES: dict[RoleName, str] = {
    "judge": "judge_oracle.md",
    "relation": "relation_oracle.md",
    "executor": "executor_subagent.md",
}


def prompts_dir(repo_root: str | Path | None = None) -> Path:
    if repo_root is not None:
        return Path(repo_root) / "prompts"
    return Path(__file__).resolve().parents[2] / "prompts"


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
