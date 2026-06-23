"""Mock relation oracle adapter."""

from __future__ import annotations

import json
import sys


def main() -> None:
    payload = json.loads(sys.stdin.read())
    nodes = payload.get("nodes", [])
    source_ids = [node["node_id"] for node in nodes[:2]]
    relation = "independent"
    rationale = "mock relation oracle; replace with a strong subagent"
    if len(nodes) >= 2 and nodes[0].get("claim", "").strip().casefold() == nodes[1].get("claim", "").strip().casefold():
        relation = "equivalent"
        rationale = "normalized claims match"
    print(
        json.dumps(
            {
                "relation": relation,
                "source_node_ids": source_ids,
                "rationale": rationale,
                "discriminator_question": None,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
