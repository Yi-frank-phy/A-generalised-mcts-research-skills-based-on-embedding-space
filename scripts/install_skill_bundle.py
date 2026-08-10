"""Install an exact verified production DTE Skill bundle with rollback."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dte_backend.bundle_manifest import MANIFEST_NAME, verify_bundle_manifest


def install_bundle(source: Path, target: Path) -> dict[str, str | None]:
    source = source.resolve()
    target = target.resolve()
    if target == target.parent or target == Path.home().resolve():
        raise ValueError("refusing to install a Skill bundle at a broad filesystem root")
    if target == source or target in source.parents or source in target.parents:
        raise ValueError("source and target Skill trees must not overlap")
    manifest = verify_bundle_manifest(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.stage-", dir=target.parent))
    backup = target.with_name(f".{target.name}.backup-{uuid.uuid4().hex}")
    try:
        for record in manifest["files"]:
            relative = Path(record["path"])
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, destination)
        shutil.copy2(source / MANIFEST_NAME, stage / MANIFEST_NAME)
        verify_bundle_manifest(stage)
        if target.exists():
            target.replace(backup)
        stage.replace(target)
        verify_bundle_manifest(target)
    except Exception:
        if target.exists() and backup.exists():
            shutil.rmtree(target)
        if backup.exists() and not target.exists():
            backup.replace(target)
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return {
        "source": str(source),
        "target": str(target),
        "bundle_sha256": manifest["bundle_sha256"],
        "backup": str(backup) if backup.exists() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(install_bundle(args.source, args.target), ensure_ascii=False))


if __name__ == "__main__":
    main()
