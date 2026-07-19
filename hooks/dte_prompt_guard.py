"""Codex UserPromptSubmit guard for DTE workflow requests.

This hook only injects a short checklist. Hard artifact validation still lives
in hooks/dte_guard.py and must run before machine-facing outputs are consumed.
"""

from __future__ import annotations

import json
import sys


def _contains_any(text: str, tokens: list[str]) -> bool:
    lowered = text.lower()
    return any(token.lower() in lowered for token in tokens)


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0

    prompt = str(payload.get("prompt") or "")
    cwd = str(payload.get("cwd") or "")
    text = f"{cwd}\n{prompt}"

    triggers = [
        "dte",
        "deep think evolving",
        "evolving-frontier-research",
        "searchnode",
        "judge oracle",
        "relation oracle",
        "evolutioncontroller",
        "dte-codex-skill-backend",
        "极深研究",
        "科研推演",
        "研究协议",
    ]

    if not _contains_any(text, triggers):
        return 0

    reminder = (
        "DTE hook reminder:\n"
        "1. 涉及 DTE 研究或 dte-codex-skill-backend 时，必须先使用已安装的 "
        "`evolving-frontier-research` skill，并保持 SearchNode -> Judge -> "
        "EvolutionController -> Executor -> Relation -> terminal handoff -> "
        "main-agent report 的角色边界；不要制造 final Synthesis episode。\n"
        "2. 后端 run/spec 输入在消费前运行 "
        "`python hooks/dte_guard.py spec <run_spec.json>`。\n"
        "3. Judge 输出在消费前运行 "
        "`python hooks/dte_guard.py judge --nodes <frontier_nodes.json> "
        "--output <judge_output.json>`。\n"
        "4. Relation 输出在消费前运行 "
        "`python hooks/dte_guard.py relation --nodes <frontier_nodes.json> "
        "--output <relation_output.json>`。\n"
        "5. Executor 输出在加入图状态前运行 "
        "`python hooks/dte_guard.py executor --parent <parent.json> "
        "--output <executor_output.json> --child-count <n>`。\n"
        "6. terminal 后同时读取 `observability-summary --format json` 与 "
        "`epistemic-summary --format json`，区分未搜索/未入选与受到挑战/反驳。\n"
        "7. 只有用户明确确认判断变化时才使用 `record-learning --source user`；"
        "不得从沉默或继续对话推断学习。\n"
        "8. 任何 guard 失败都必须停止消费该产物；不要让 Executor、Judge 或 "
        "Relation oracle 直接生成最终用户报告。"
    )

    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": reminder,
                }
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
