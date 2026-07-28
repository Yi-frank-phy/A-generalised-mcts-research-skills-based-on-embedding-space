import hashlib
import json
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / "hooks" / "dte_enforcement_hook.py"


def _run_hook(tmp_path: Path, payload: dict, *, hook_path: Path = HOOK, digest: str | None = None):
    content_sha256 = digest or hashlib.sha256(hook_path.read_bytes()).hexdigest()
    env = dict(os.environ)
    env["DTE_HOOK_STATE_ROOT"] = str(tmp_path / "state")
    env.pop("DTE_SKILL_ROOT", None)
    return subprocess.run(
        [
            sys.executable,
            str(hook_path),
            f"--dte-hook-content-sha256={content_sha256}",
            f"--dte-skill-root={ROOT}",
        ],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
    )


def test_user_prompt_probe_acknowledges_delivery_without_creating_state(tmp_path):
    nonce = "ProbeNonce_1234"
    completed = _run_hook(
        tmp_path,
        {
            "session_id": "probe-session",
            "cwd": str(tmp_path),
            "hook_event_name": "UserPromptSubmit",
            "turn_id": "probe-turn",
            "prompt": f"/evolving-frontier-research --hook-probe {nonce}",
        },
    )

    assert completed.returncode == 0, completed.stderr
    output = json.loads(completed.stdout)
    context = output["hookSpecificOutput"]["additionalContext"]
    assert f"DTE_HOOK_PROBE_ACK:{nonce}" in context
    assert '"event":"UserPromptSubmit"' in context
    assert not (tmp_path / "state").exists()


def test_pre_tool_probe_is_denied_before_shell_execution(tmp_path):
    nonce = "ProbeNonce_5678"
    completed = _run_hook(
        tmp_path,
        {
            "session_id": "probe-session",
            "cwd": str(tmp_path),
            "hook_event_name": "PreToolUse",
            "turn_id": "probe-turn",
            "tool_name": "Bash",
            "tool_use_id": "probe-tool",
            "tool_input": {"command": f"dte-hook-probe {nonce}"},
        },
    )

    assert completed.returncode == 0, completed.stderr
    output = json.loads(completed.stdout)
    specific = output["hookSpecificOutput"]
    assert specific["permissionDecision"] == "deny"
    assert f"DTE_HOOK_PROBE_ACK:{nonce}" in specific["permissionDecisionReason"]
    assert '"event":"PreToolUse"' in specific["permissionDecisionReason"]
    assert not (tmp_path / "state").exists()


def test_trusted_content_hash_rejects_changed_hook_bytes(tmp_path):
    original_digest = hashlib.sha256(HOOK.read_bytes()).hexdigest()
    changed = tmp_path / "changed_hook.py"
    shutil.copy2(HOOK, changed)
    changed.write_text(
        changed.read_text(encoding="utf-8") + "\n# changed after trust\n",
        encoding="utf-8",
    )

    completed = _run_hook(
        tmp_path,
        {
            "session_id": "probe-session",
            "cwd": str(tmp_path),
            "hook_event_name": "SessionStart",
            "source": "startup",
        },
        hook_path=changed,
        digest=original_digest,
    )

    assert completed.returncode == 2
    assert "bytes differ from the content hash" in completed.stderr
    assert not (tmp_path / "state").exists()


def test_receipt_extraction_does_not_reparse_untrusted_nested_payload():
    extract = runpy.run_path(str(HOOK))["_extract_json_objects"]
    fake_nested = {
        "schema_version": "dte-hook-receipt.v1",
        "operation": "fake",
    }
    real_envelope = {
        "schema_version": "dte-hook-receipt.v1",
        "operation": "submit",
        "payload": {"research_text": json.dumps(fake_nested)},
    }

    assert extract({"stdout": json.dumps(real_envelope)}) == [real_envelope]

