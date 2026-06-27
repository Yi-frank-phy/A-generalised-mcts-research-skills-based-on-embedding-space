"""Mock relation oracle adapter for smoke tests only."""

from __future__ import annotations

import json
import os
import sys


def require_smoke_mode() -> None:
    if os.getenv("DTE_ALLOW_MOCK_ADAPTER") != "1":
        raise SystemExit(
            "mock_relation_adapter.py is smoke-only. Set DTE_ALLOW_MOCK_ADAPTER=1 "
            "only for tests, or replace it with a real Codex Relation subagent."
        )


def main() -> None:
    require_smoke_mode()
    payload = json.loads(sys.stdin.read())
    nodes = payload.get("nodes", [])
    source_ids = [node["node_id"] for node in nodes[:2]]
    relation = "independent"
    rationale = "SMOKE-ONLY mock relation oracle; not a research judgment"
    if len(nodes) >= 2 and nodes[0].get("claim", "").strip().casefold() == nodes[1].get("claim", "").strip().casefold():
        relation = "equivalent"
        rationale = "SMOKE-ONLY normalized-claim match"
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
