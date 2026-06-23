"""Mock external executor adapter for DTE.

Reads ExpansionRequest JSON from stdin and writes {"nodes": [...]} to stdout.
This simulates where a Codex/Kimi/OpenClaw episode would return structured
SearchNode children without judging or synthesizing.
"""

from __future__ import annotations

import json
import sys
import uuid


def main() -> None:
    request = json.loads(sys.stdin.read())
    parent = request["parent"]
    child_count = int(request["child_count"])
    iteration = int(request["iteration"])

    nodes = []
    for i in range(child_count):
        nodes.append(
            {
                "node_id": f"adapter-{iteration}-{i}-{uuid.uuid4()}",
                "node_type": "candidate",
                "claim": f"Adapter expansion {i + 1} for: {parent['claim']}",
                "rationale": "Mock adapter child. A real executor would add evidence, counterexamples, or derivation steps here.",
                "assumptions": parent.get("assumptions", []),
                "evidence": parent.get("evidence", []),
                "risks": ["mock adapter output; replace with real executor evidence"],
                "parent_ids": [parent["node_id"]],
                "confidence": max(0.05, min(0.95, float(parent.get("confidence", 0.5)) + 0.01 * i)),
                "status": "frontier",
            }
        )
    print(json.dumps({"nodes": nodes}, ensure_ascii=False))


if __name__ == "__main__":
    main()
