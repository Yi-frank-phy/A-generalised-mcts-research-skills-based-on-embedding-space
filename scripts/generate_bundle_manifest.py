"""Generate or verify the deterministic DTE Skill bundle manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dte_backend.bundle_manifest import verify_bundle_manifest, write_bundle_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    manifest = (
        verify_bundle_manifest(ROOT)
        if args.verify
        else json.loads(write_bundle_manifest(ROOT).read_text(encoding="utf-8"))
    )
    print(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
