"""Example hook: validate a SearchNode JSON artifact.

This can be called by Codex/Kimi/OpenClaw after an executor episode returns a
candidate node. The hook should fail closed if the artifact is invalid.
"""

from __future__ import annotations

import sys
from dte_backend.models import SearchNode
from dte_backend.validators import load_json_model


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python hooks/validate_search_node.py path/to/node.json", file=sys.stderr)
        return 2
    try:
        node = load_json_model(sys.argv[1], SearchNode)
        print(node.model_dump_json(indent=2))
        return 0
    except Exception as exc:
        print(f"Invalid SearchNode: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
