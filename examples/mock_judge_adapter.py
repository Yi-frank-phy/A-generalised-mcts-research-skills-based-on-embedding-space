"""Mock Judge oracle adapter for smoke tests only.

This adapter is deliberately blocked by default so it cannot be mistaken for a
real research Judge. Set `DTE_ALLOW_MOCK_ADAPTER=1` only in smoke tests or local
protocol checks.
"""

from __future__ import annotations

import json
import os
import sys


def require_smoke_mode() -> None:
    if os.getenv("DTE_ALLOW_MOCK_ADAPTER") != "1":
        raise SystemExit(
            "mock_judge_adapter.py is smoke-only. Set DTE_ALLOW_MOCK_ADAPTER=1 "
            "only for tests, or replace it with a real Codex Judge subagent."
        )


def main() -> None:
    require_smoke_mode()
    payload = json.loads(sys.stdin.read())
    results = []
    for node in payload.get("nodes", []):
        confidence = float(node.get("confidence", 0.5))
        evidence_bonus = min(0.2, 0.05 * len(node.get("evidence", [])))
        risk_penalty = min(0.2, 0.05 * len(node.get("risks", [])))
        score = max(0.0, min(1.0, confidence + evidence_bonus - risk_penalty))
        results.append(
            {
                "node_id": node["node_id"],
                "score": score,
                "reasoning": "SMOKE-ONLY mock judge score; not a research judgment",
                "risks": ["mock adapter used; replace with real Judge subagent for research"],
            }
        )
    print(json.dumps({"results": results}, ensure_ascii=False))


if __name__ == "__main__":
    main()
