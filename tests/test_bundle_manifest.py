import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from dte_backend.bundle_manifest import (
    EXACT_FILES,
    build_bundle_manifest,
    verify_bundle_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def test_bundle_identity_is_stable_across_platform_line_endings(tmp_path):
    lf_root = tmp_path / "lf"
    crlf_root = tmp_path / "crlf"
    for root, newline in ((lf_root, "\n"), (crlf_root, "\r\n")):
        for relative in EXACT_FILES:
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(f"first{newline}second{newline}".encode("utf-8"))

    assert build_bundle_manifest(lf_root) == build_bundle_manifest(crlf_root)


def test_repository_bundle_manifest_is_current():
    manifest = verify_bundle_manifest(ROOT)
    assert manifest["schema_version"] == "dte-skill-bundle.v1"
    assert len(manifest["bundle_sha256"]) == 64
    paths = {record["path"] for record in manifest["files"]}
    assert "hooks/dte_enforcement_hook.py" in paths
    assert "scripts/dte_hook_driver_entry.py" in paths
    assert not any("tests/" in path or "__pycache__" in path for path in paths)


def test_installed_copy_runs_pinned_hook_driver_and_detects_tampering(tmp_path):
    target = tmp_path / "installed-skill"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "install_skill_bundle.py"),
            "--source",
            str(ROOT),
            "--target",
            str(target),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout)["target"] == str(target.resolve())
    assert verify_bundle_manifest(target)["bundle_sha256"]
    help_result = subprocess.run(
        [
            sys.executable,
            str(target / "scripts" / "dte_hook_driver_entry.py"),
            "hook-driver",
            "--help",
        ],
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0, help_result.stdout + help_result.stderr

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state_root = tmp_path / "hook-state"
    driver_env = {
        **os.environ,
        "DTE_HOOK_STATE_ROOT": str(state_root),
        "DTE_HOOK_SESSION_ID": "installed-session",
        "DTE_HOOK_TURN_ID": "installed-turn",
        "DTE_HOOK_CWD": str(workspace),
    }
    entry = target / "scripts" / "dte_hook_driver_entry.py"
    activated = subprocess.run(
        [sys.executable, str(entry), "hook-driver", "activate", "--source", "explicit"],
        capture_output=True,
        text=True,
        env=driver_env,
    )
    assert activated.returncode == 0, activated.stdout + activated.stderr
    capability_path = state_root / "capabilities" / "installed-session.json"
    driver_env["DTE_HOOK_CAPABILITY"] = json.loads(
        capability_path.read_text(encoding="utf-8")
    )["current"]
    spec = json.loads((ROOT / "examples" / "run_spec.json").read_text(encoding="utf-8"))
    spec["role_isolation_mode"] = "shared_context_single_agent"
    spec_path = tmp_path / "spec.json"
    nodes_path = tmp_path / "nodes.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    nodes_path.write_text(
        json.dumps([{"node_id": "seed", "claim": "installed copy probe"}]),
        encoding="utf-8",
    )
    initialized = subprocess.run(
        [
            sys.executable,
            str(entry),
            "hook-driver",
            "init",
            "--spec",
            str(spec_path),
            "--nodes",
            str(nodes_path),
        ],
        capture_output=True,
        text=True,
        env=driver_env,
    )
    assert initialized.returncode == 0, initialized.stdout + initialized.stderr
    driver_env["DTE_HOOK_CAPABILITY"] = json.loads(
        capability_path.read_text(encoding="utf-8")
    )["current"]
    stepped = subprocess.run(
        [sys.executable, str(entry), "hook-driver", "step"],
        capture_output=True,
        text=True,
        env=driver_env,
    )
    assert stepped.returncode == 0, stepped.stdout + stepped.stderr
    step_receipt = json.loads(stepped.stdout)
    assert step_receipt["payload"]["request_ref"]["schema_version"] == "dte-request-ref.v1"
    assert len(stepped.stdout.encode("utf-8")) < 4096

    hook = target / "hooks" / "dte_enforcement_hook.py"
    hook.write_text(hook.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest differs"):
        verify_bundle_manifest(target)
