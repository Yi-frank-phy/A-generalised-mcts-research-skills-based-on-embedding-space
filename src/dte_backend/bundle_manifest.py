"""Deterministic production Skill bundle inventory and verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


BUNDLE_SCHEMA = "dte-skill-bundle.v1"
MANIFEST_NAME = "bundle-manifest.json"
EXACT_FILES = (
    "SKILL.md",
    "hooks/dte_enforcement_hook.py",
    "scripts/dte_hook_driver_entry.py",
    "scripts/generate_bundle_manifest.py",
    "scripts/install_dte_hooks.py",
    "scripts/install_skill_bundle.py",
)
TREE_SUFFIXES = {
    "src/dte_backend": {".py"},
    "prompts": {".md"},
    "schemas": {".json"},
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _bundle_content_bytes(path: Path) -> bytes:
    """Return the Git-canonical text bytes used for cross-platform identity."""

    return path.read_bytes().replace(b"\r\n", b"\n")


def production_paths(root: str | Path) -> list[Path]:
    root_path = Path(root).resolve()
    paths = [root_path / relative for relative in EXACT_FILES]
    for relative, suffixes in TREE_SUFFIXES.items():
        tree = root_path / relative
        paths.extend(
            path
            for path in tree.rglob("*")
            if path.is_file()
            and path.suffix in suffixes
            and "__pycache__" not in path.parts
        )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "production Skill files are missing: "
            + ", ".join(str(path.relative_to(root_path)) for path in missing)
        )
    return sorted(set(paths), key=lambda path: path.relative_to(root_path).as_posix())


def build_bundle_manifest(root: str | Path) -> dict[str, Any]:
    root_path = Path(root).resolve()
    files = []
    for path in production_paths(root_path):
        content = _bundle_content_bytes(path)
        files.append(
            {
                "path": path.relative_to(root_path).as_posix(),
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    identity = {"schema_version": BUNDLE_SCHEMA, "files": files}
    return {
        **identity,
        "bundle_sha256": hashlib.sha256(_canonical_bytes(identity)).hexdigest(),
    }


def write_bundle_manifest(root: str | Path) -> Path:
    root_path = Path(root).resolve()
    target = root_path / MANIFEST_NAME
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(build_bundle_manifest(root_path), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def verify_bundle_manifest(root: str | Path) -> dict[str, Any]:
    root_path = Path(root).resolve()
    target = root_path / MANIFEST_NAME
    persisted = json.loads(target.read_text(encoding="utf-8"))
    expected = build_bundle_manifest(root_path)
    if persisted != expected:
        raise ValueError("Skill bundle manifest differs from the production files")
    return expected
