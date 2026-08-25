"""One-shot, fail-closed migration for phase 1 of generic Judge removal.

This script is intentionally temporary and public. It edits only the App-run
production path needed to make new runs bypass generic Judge episodes while
retaining legacy Judge schemas/read/retry validation for old persisted artifacts.
Every edit is guarded by exact structural assertions; a partial or unexpected
source tree aborts without writing the target file.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_DRIVER = ROOT / "src" / "dte_backend" / "app_driver.py"


def require_once(text: str, needle: str, label: str) -> None:
    count = text.count(needle)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    require_once(text, old, label)
    return text.replace(old, new, 1)


def delete_between(text: str, start: str, end: str, label: str) -> str:
    require_once(text, start, f"{label} start")
    require_once(text, end, f"{label} end")
    i = text.index(start)
    j = text.index(end, i)
    if j <= i:
        raise RuntimeError(f"{label}: invalid source ordering")
    return text[:i] + text[j:]


def migrate(text: str) -> str:
    # Idempotent success path after this migration has already been applied.
    if (
        "def _select_unjudged_frontier(" not in text
        and "controller progression requires every frontier node to be judged" not in text
        and "unjudged = _select_unjudged_frontier(state)" not in text
        and "purpose=\"provenance_repair\"" not in text
        and "judge_or_confidence=" not in text
    ):
        return text

    text = delete_between(
        text,
        "def _select_unjudged_frontier(state: AppRunState) -> list[SearchNode]:\n",
        "def _ensure_nonterminal(state: AppRunState, operation: str) -> None:\n",
        "remove unjudged-frontier scheduler helper",
    )

    text = replace_once(
        text,
        "    if any(node.score is None for node in frontier):\n"
        "        raise RuntimeError(\"controller progression requires every frontier node to be judged\")\n",
        "",
        "remove Judge prerequisite from controller progression",
    )

    text = replace_once(
        text,
        "        score = node.score if node.score is not None else node.confidence\n",
        "        score = node.confidence\n",
        "make synthesis disposition independent of Judge score",
    )
    text = replace_once(
        text,
        "                    f\"judge_or_confidence={score:.6f}\",\n",
        "                    f\"confidence={score:.6f}\",\n",
        "rename synthesis disposition provenance fact",
    )

    strict_start = (
        "        if (\n"
        "            relation_clear\n"
        "            and missing_provenance\n"
        "            and state.spec.material_provenance_policy == \"strict_repair\"\n"
        "        ):\n"
    )
    disclosure_start = (
        "        elif (\n"
        "            relation_clear\n"
        "            and missing_provenance\n"
        "            and state.spec.material_provenance_policy == \"terminal_disclosure\"\n"
        "        ):\n"
    )
    require_once(text, strict_start, "strict provenance-repair branch")
    require_once(text, disclosure_start, "terminal provenance-disclosure branch")
    i = text.index(strict_start)
    j = text.index(disclosure_start, i)
    replacement = (
        strict_start
        + "            # Generic Judge repair was removed from the production workflow.\n"
        + "            # Until a dedicated evidence/falsification mechanism exists, the\n"
        + "            # legacy strict-repair policy degrades explicitly rather than\n"
        + "            # spawning an LLM-as-a-Judge episode.\n"
        + "            state.provenance_repair_attempted_node_ids = sorted(missing_provenance)\n"
        + "            state.provenance_repair_exhausted_node_ids = sorted(missing_provenance)\n"
        + "            degradation_codes.append(\"material_provenance_repair_exhausted\")\n"
    )
    text = text[:i] + replacement + text[j:]

    judge_schedule_start = "        unjudged = _select_unjudged_frontier(state)\n"
    judge_schedule_end = "        if state.pending_terminal_action is not None:\n"
    text = delete_between(
        text,
        judge_schedule_start,
        judge_schedule_end,
        "remove initial/child generic Judge scheduling",
    )

    text = replace_once(
        text,
        "        # The iteration cap prevents another controller allocation. It does\n"
        "        # not revoke already-authorized Executor output: every committed child\n"
        "        # must still pass a bounded Judge episode before Relation/readiness.\n",
        "        # The iteration cap prevents another controller allocation. It does\n"
        "        # not revoke already-authorized Executor output.\n",
        "remove stale Judge-before-Relation comment",
    )

    text = replace_once(
        text,
        "            if _select_executor_parent(state) is not None or _select_unjudged_frontier(state):\n",
        "            if _select_executor_parent(state) is not None:\n",
        "remove Judge queue from pending terminal drain",
    )

    # New runs may persist proper-volume geometry/UCB with no Judge score. Keep
    # the old Judge observation validation only when legacy Judge-owned fields
    # are actually present.
    score_start = "        if node.score is None:\n"
    geometry_start = "        if any(value is not None for value in geometry) and not all(\n"
    require_once(text, score_start, "Judge-owned node-state validation start")
    require_once(text, geometry_start, "geometry validation continuation")
    i = text.index(score_start)
    j = text.index(geometry_start, i)
    old_block = text[i:j]
    owner_marker = "        owner = judge_observations.get(node.node_id)\n"
    require_once(old_block, owner_marker, "legacy Judge observation validation")
    owner_i = old_block.index(owner_marker)
    owner_block = old_block[owner_i:]
    indented_owner = "".join(
        ("    " + line if line.strip() else line)
        for line in owner_block.splitlines(keepends=True)
    )
    new_block = (
        "        if node.score is None:\n"
        "            if judge_fields_present:\n"
        "                raise ValueError(\"unscored node contains persisted Judge-owned state\")\n"
        "        else:\n"
        + indented_owner
    )
    text = text[:i] + new_block + text[j:]

    forbidden = {
        "generic Judge scheduler helper": "def _select_unjudged_frontier(",
        "Judge progression prerequisite": "controller progression requires every frontier node to be judged",
        "Judge scheduling statement": "unjudged = _select_unjudged_frontier(state)",
        "Judge provenance repair request": "purpose=\"provenance_repair\"",
        "Judge/confidence synthesis fallback": "judge_or_confidence=",
    }
    for label, needle in forbidden.items():
        if needle in text:
            raise RuntimeError(f"phase-1 invariant failed: {label} remains")

    # `build_judge_episode_request` is intentionally allowed only for retrying a
    # legacy already-persisted Judge attempt. Phase 2 removes that compatibility
    # path together with the Judge protocol itself.
    if text.count("build_judge_episode_request(") != 1:
        raise RuntimeError(
            "phase-1 invariant failed: expected exactly one legacy Judge retry builder"
        )
    if "Judge retry targets are no longer committed frontier nodes" not in text:
        raise RuntimeError("phase-1 invariant failed: legacy Judge builder is not retry-only")
    return text


def main() -> None:
    original = APP_DRIVER.read_text(encoding="utf-8")
    migrated = migrate(original)
    if migrated == original:
        print("generic Judge phase-1 migration already applied")
        return
    APP_DRIVER.write_text(migrated, encoding="utf-8")
    print("updated", APP_DRIVER.relative_to(ROOT))


if __name__ == "__main__":
    main()
