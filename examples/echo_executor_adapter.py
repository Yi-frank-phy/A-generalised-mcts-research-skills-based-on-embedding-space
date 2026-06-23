"""Example subprocess executor adapter for the DTE backend.

It reads one ExpansionRequest JSON object from stdin and writes structured
SearchNode children to stdout. This is only a local smoke-test adapter; real
Codex/Kimi/OpenClaw wrappers should keep the same JSON boundary.
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    request = json.loads(sys.stdin.read())
    parent = request["parent"]
    count = int(request.get("child_count", request.get("count", 0)))

    nodes = []
    for index in range(count):
        nodes.append(
            {
                "node_id": f"{parent['node_id']}-executor-{index + 1}",
                "node_type": "candidate",
                "claim": f"{parent['claim']} - executor expansion {index + 1}",
                "rationale": "Structured child returned by the example subprocess executor.",
                "assumptions": list(parent.get("assumptions", [])),
                "evidence": list(parent.get("evidence", [])),
                "risks": ["example adapter output"],
                "parent_ids": [parent["node_id"]],
                "confidence": 0.45,
                "status": "frontier",
            }
        )

    print(json.dumps({"nodes": nodes}, ensure_ascii=False))


if __name__ == "__main__":
    main()
