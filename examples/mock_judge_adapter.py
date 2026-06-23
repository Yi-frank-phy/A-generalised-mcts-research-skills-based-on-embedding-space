"""Mock Judge oracle adapter.

Reads a Judge task JSON from stdin and writes observable scores to stdout.
A real Codex subagent can replace this command as long as it returns the same
shape.
"""

from __future__ import annotations

import json
import sys


def main() -> None:
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
                "reasoning": "mock judge oracle score; replace with a strong subagent",
                "risks": [],
            }
        )
    print(json.dumps({"results": results}, ensure_ascii=False))


if __name__ == "__main__":
    main()
