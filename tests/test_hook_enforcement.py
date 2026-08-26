import hashlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

import dte_backend.hook_driver as hook_driver
from dte_backend.app_driver import (
    create_app_run,
    load_app_run,
    next_app_episode,
    submit_app_episode_result,
)
from dte_backend.episode_models import EpisodeRequest, EpisodeResult, RuntimeDiagnostics, compute_output_hash
from dte_backend.hook_driver import (
    HookDriverReceipt,
    HookRequestChunk,
    HookRequestReference,
    HookSessionManifest,
    HookStatusProjection,
)
from dte_backend.epistemic_models import (
    EpistemicContributionBundle,
    EpistemicStatementContribution,
)
from dte_backend.models import DTERunSpec, SearchNode


ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / "hooks" / "dte_enforcement_hook.py"


def hook_env(tmp_path: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["DTE_HOOK_STATE_ROOT"] = str(tmp_path / "hook-state")
    env["DTE_SKILL_ROOT"] = str(ROOT)
    env["PYTHONPATH"] = str(ROOT / "src")
    return env


def run_hook(tmp_path: Path, payload, *, raw: str | None = None):
    return subprocess.run(
        [sys.executable, str(HOOK)],
        input=raw if raw is not None else json.dumps(payload),
        capture_output=True,
        text=True,
        env=hook_env(tmp_path),
    )


def explicit_payload(tmp_path: Path, session: str = "session-a", turn: str = "turn-a"):
    return {
        "session_id": session,
        "turn_id": turn,
        "cwd": str(tmp_path / "workspace"),
        "hook_event_name": "UserPromptSubmit",
        "permission_mode": "default",
        "prompt": "/evolving-frontier-research prove a bounded claim",
    }


def manifest_file(tmp_path: Path, session: str = "session-a") -> Path:
    return tmp_path / "hook-state" / "sessions" / f"{session}.json"


def capability_value(tmp_path: Path, session: str = "session-a") -> str:
    path = tmp_path / "hook-state" / "capabilities" / f"{session}.json"
    return json.loads(path.read_text(encoding="utf-8"))["current"]


def driver_env(tmp_path: Path, *, capability: str, session="session-a", turn="turn-a"):
    env = hook_env(tmp_path)
    env.update(
        {
            "DTE_HOOK_SESSION_ID": session,
            "DTE_HOOK_TURN_ID": turn,
            "DTE_HOOK_CWD": str(tmp_path / "workspace"),
            "DTE_HOOK_CAPABILITY": capability,
        }
    )
    return env


def run_driver(
    tmp_path: Path,
    *arguments: str,
    capability: str | None = None,
    session: str = "session-a",
    turn: str = "turn-a",
):
    return subprocess.run(
        [sys.executable, "-m", "dte_backend", "hook-driver", *arguments],
        capture_output=True,
        encoding="utf-8",
        env=driver_env(
            tmp_path,
            capability=(
                capability
                if capability is not None
                else capability_value(tmp_path, session)
            ),
            session=session,
            turn=turn,
        ),
        cwd=ROOT,
    )


def read_request(
    tmp_path: Path,
    reference: HookRequestReference,
) -> EpisodeRequest:
    content: list[str] = []
    for chunk_index in range(reference.chunk_count):
        projected = run_driver(
            tmp_path,
            "request",
            "--chunk-index",
            str(chunk_index),
        )
        assert projected.returncode == 0, projected.stdout + projected.stderr
        content.append(HookRequestChunk.model_validate_json(projected.stdout).content)
    return EpisodeRequest.model_validate_json("".join(content))


def write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    spec = json.loads((ROOT / "examples" / "run_spec.json").read_text(encoding="utf-8"))
    spec["role_isolation_mode"] = "shared_context_single_agent"
    spec["budget"]["max_committed_search_nodes"] = 2
    spec["budget"]["max_iterations"] = 1
    spec_path = tmp_path / "spec.json"
    nodes_path = tmp_path / "nodes.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    nodes_path.write_text(
        json.dumps(
            [
                {
                    "node_id": "seed-a",
                    "claim": "Candidate A",
                    "rationale": "seed",
                    "status": "frontier",
                }
            ]
        ),
        encoding="utf-8",
    )
    return spec_path, nodes_path




def test_explicit_invocation_activates_but_meta_discussion_does_not(tmp_path):
    meta = explicit_payload(tmp_path, session="meta")
    meta["prompt"] = "Please review the DTE hook architecture; do not invoke the skill."
    completed = run_hook(tmp_path, meta)
    assert completed.returncode == 0
    assert completed.stdout == ""
    assert not manifest_file(tmp_path, "meta").exists()

    completed = run_hook(tmp_path, explicit_payload(tmp_path))
    assert completed.returncode == 0, completed.stderr
    output = json.loads(completed.stdout)
    assert "DTE enforcement activated" in output["hookSpecificOutput"]["additionalContext"]
    manifest = HookSessionManifest.model_validate_json(
        manifest_file(tmp_path).read_text(encoding="utf-8")
    )
    assert manifest.phase == "awaiting_init"
    assert manifest.activation_source == "explicit"
    assert str((ROOT / "src" / "dte_backend").resolve()) in manifest.protected_paths


def test_skill_link_activates_and_malformed_json_fails_closed(tmp_path):
    payload = explicit_payload(tmp_path, session="link")
    payload["prompt"] = (
        "[$evolving-frontier-research](C:\\Users\\zhaoy\\.codex\\skills\\"
        "evolving-frontier-research\\SKILL.md) research"
    )
    assert run_hook(tmp_path, payload).returncode == 0
    assert manifest_file(tmp_path, "link").exists()

    malformed = run_hook(tmp_path, {}, raw="{not-json")
    assert malformed.returncode == 2
    assert "failed closed" in malformed.stderr


def test_pretool_allows_research_denies_direct_control_and_rewrites_driver(tmp_path):
    assert run_hook(tmp_path, explicit_payload(tmp_path)).returncode == 0
    base = {
        "session_id": "session-a",
        "turn_id": "turn-a",
        "cwd": str(tmp_path / "workspace"),
        "hook_event_name": "PreToolUse",
        "permission_mode": "default",
        "tool_name": "Bash",
        "tool_use_id": "tool-1",
    }
    research = {**base, "tool_input": {"command": "python -c \"print(2 + 2)\""}}
    completed = run_hook(tmp_path, research)
    assert completed.returncode == 0
    assert completed.stdout == ""

    direct = {
        **base,
        "tool_input": {"command": "python -m dte_backend create-run --run-dir x --spec s --nodes n"},
    }
    denied = json.loads(run_hook(tmp_path, direct).stdout)
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"

    console_direct = {
        **base,
        "tool_input": {
            "command": "dte-backend create-run --run-dir x --spec s --nodes n"
        },
    }
    console_denied = json.loads(run_hook(tmp_path, console_direct).stdout)
    assert console_denied["hookSpecificOutput"]["permissionDecision"] == "deny"

    driver = {
        **base,
        "tool_input": {"command": "python -m dte_backend hook-driver status"},
    }
    rewritten = json.loads(run_hook(tmp_path, driver).stdout)
    updated = rewritten["hookSpecificOutput"]["updatedInput"]["command"]
    assert "DTE_HOOK_CAPABILITY" in updated
    assert "dte_hook_driver_entry.py" in updated
    assert updated.endswith("hook-driver status")

    console_driver = {
        **base,
        "tool_input": {"command": "dte-backend hook-driver status"},
    }
    console_rewritten = json.loads(run_hook(tmp_path, console_driver).stdout)
    console_updated = console_rewritten["hookSpecificOutput"]["updatedInput"]["command"]
    assert "DTE_HOOK_CAPABILITY" in console_updated
    assert "dte_hook_driver_entry.py" in console_updated
    assert console_updated.endswith("hook-driver status")

    quoted_driver = {
        **base,
        "tool_input": {
            "command": f"& '{sys.executable}' -m dte_backend hook-driver status"
        },
    }
    quoted_rewritten = json.loads(run_hook(tmp_path, quoted_driver).stdout)
    quoted_updated = quoted_rewritten["hookSpecificOutput"]["updatedInput"]["command"]
    assert "DTE_HOOK_CAPABILITY" in quoted_updated
    assert "dte_hook_driver_entry.py" in quoted_updated
    assert quoted_updated.endswith("hook-driver status")

    if os.name == "nt":
        cmd_driver = {
            **driver,
            "tool_input": {
                "command": "python -m dte_backend hook-driver status",
                "shell": "cmd.exe",
            },
        }
        cmd_denied = json.loads(run_hook(tmp_path, cmd_driver).stdout)
        assert cmd_denied["hookSpecificOutput"]["permissionDecision"] == "deny"
        assert "require PowerShell" in cmd_denied["hookSpecificOutput"][
            "permissionDecisionReason"
        ]

    child = {**driver, "turn_id": "subagent-turn"}
    child_denied = json.loads(run_hook(tmp_path, child).stdout)
    assert child_denied["hookSpecificOutput"]["permissionDecision"] == "deny"

    protected = {
        **base,
        "tool_input": {
            "command": f"Get-Content '{tmp_path / 'hook-state' / 'sessions' / 'session-a.json'}'"
        },
    }
    protected_denied = json.loads(run_hook(tmp_path, protected).stdout)
    assert protected_denied["hookSpecificOutput"]["permissionDecision"] == "deny"

    backend_source = {
        **base,
        "tool_input": {
            "command": (
                f"Set-Content -LiteralPath '{ROOT / 'src' / 'dte_backend' / 'validators.py'}' "
                "-Value '# bypass'"
            )
        },
    }
    backend_denied = json.loads(run_hook(tmp_path, backend_source).stdout)
    assert backend_denied["hookSpecificOutput"]["permissionDecision"] == "deny"

    relative_backend_source = {
        **base,
        "tool_input": {
            "workdir": str(ROOT),
            "command": (
                "Set-Content -LiteralPath "
                "'src\\dte_backend\\validators.py' -Value '# bypass'"
            ),
        },
    }
    relative_backend_denied = json.loads(
        run_hook(tmp_path, relative_backend_source).stdout
    )
    assert (
        relative_backend_denied["hookSpecificOutput"]["permissionDecision"]
        == "deny"
    )

    relative_posix_source = {
        **base,
        "tool_input": {
            "workdir": str(ROOT),
            "command": "printf bypass > src/dte_backend/validators.py",
        },
    }
    relative_posix_denied = json.loads(
        run_hook(tmp_path, relative_posix_source).stdout
    )
    assert relative_posix_denied["hookSpecificOutput"]["permissionDecision"] == "deny"

    relative_patch = {
        **base,
        "tool_name": "apply_patch",
        "tool_input": {
            "workdir": str(ROOT),
            "patch": (
                "*** Begin Patch\n"
                "*** Update File: src/dte_backend/validators.py\n"
                "@@\n"
                "-old\n"
                "+bypass\n"
                "*** End Patch\n"
            ),
        },
    }
    relative_patch_denied = json.loads(run_hook(tmp_path, relative_patch).stdout)
    assert relative_patch_denied["hookSpecificOutput"]["permissionDecision"] == "deny"

    strict_real = {
        **base,
        "tool_input": {
            "command": "python -m dte_backend strict-run --mode real --spec x --out-dir y"
        },
    }
    strict_denied = json.loads(run_hook(tmp_path, strict_real).stdout)
    assert strict_denied["hookSpecificOutput"]["permissionDecision"] == "deny"

    console_strict_real = {
        **base,
        "tool_input": {
            "command": "dte-backend strict-run --mode real --spec x --out-dir y"
        },
    }
    console_strict_denied = json.loads(run_hook(tmp_path, console_strict_real).stdout)
    assert console_strict_denied["hookSpecificOutput"]["permissionDecision"] == "deny"

    quoted_console_strict = {
        **base,
        "tool_input": {
            "command": (
                "& 'C:\\Program Files\\DTE\\dte-backend.exe' "
                "strict-run --mode real --spec x --out-dir y"
            )
        },
    }
    quoted_console_denied = json.loads(
        run_hook(tmp_path, quoted_console_strict).stdout
    )
    assert quoted_console_denied["hookSpecificOutput"]["permissionDecision"] == "deny"




def test_stop_blocks_early_and_session_start_restores_after_compact(tmp_path):
    assert run_hook(tmp_path, explicit_payload(tmp_path)).returncode == 0
    stop = {
        "session_id": "session-a",
        "turn_id": "turn-a",
        "cwd": str(tmp_path / "workspace"),
        "hook_event_name": "Stop",
        "permission_mode": "default",
        "stop_hook_active": False,
        "last_assistant_message": "done",
    }
    output = json.loads(run_hook(tmp_path, stop).stdout)
    assert output["decision"] == "block"
    assert "hook-driver init" in output["reason"]

    start = {
        "session_id": "session-a",
        "cwd": str(tmp_path / "workspace"),
        "hook_event_name": "SessionStart",
        "permission_mode": "default",
        "source": "compact",
    }
    restored = json.loads(run_hook(tmp_path, start).stdout)
    context = restored["hookSpecificOutput"]["additionalContext"]
    assert "phase=awaiting_init" in context
    assert "unique_next_action=hook-driver init" in context

    repeated = {**stop, "stop_hook_active": True}
    paused = run_hook(tmp_path, repeated)
    assert paused.returncode == 0
    assert paused.stdout == ""
    manifest = HookSessionManifest.model_validate_json(
        manifest_file(tmp_path).read_text(encoding="utf-8")
    )
    assert manifest.phase == "awaiting_init"
    assert manifest.failure_reason is None
    assert manifest.paused is True

    resumed_prompt = explicit_payload(tmp_path, turn="turn-b")
    resumed = json.loads(run_hook(tmp_path, resumed_prompt).stdout)
    assert "DTE session resumed" in resumed["hookSpecificOutput"]["additionalContext"]
    manifest = HookSessionManifest.model_validate_json(
        manifest_file(tmp_path).read_text(encoding="utf-8")
    )
    assert manifest.paused is False
    assert manifest.active_root_turn_id == "turn-b"


def test_large_unicode_request_is_chunked_without_rotating_state(tmp_path):
    assert run_hook(tmp_path, explicit_payload(tmp_path)).returncode == 0
    spec_path, nodes_path = write_inputs(tmp_path)
    nodes = json.loads(nodes_path.read_text(encoding="utf-8"))
    nodes[0]["claim"] = ("边界证据🙂" * 18000) + " end"
    nodes_path.write_text(json.dumps(nodes, ensure_ascii=False), encoding="utf-8")
    initialized = run_driver(
        tmp_path,
        "init",
        "--spec",
        str(spec_path),
        "--nodes",
        str(nodes_path),
    )
    assert initialized.returncode == 0, initialized.stdout + initialized.stderr

    step = run_driver(tmp_path, "step")
    receipt = HookDriverReceipt.model_validate_json(step.stdout)
    reference = HookRequestReference.model_validate(receipt.payload["request_ref"])
    assert reference.canonical_size_bytes >= 128 * 1024
    assert len(step.stdout.encode("utf-8")) < 4096
    before = HookSessionManifest.model_validate_json(
        manifest_file(tmp_path).read_text(encoding="utf-8")
    )
    capability = capability_value(tmp_path)

    content = []
    for index in range(reference.chunk_count):
        projected = run_driver(tmp_path, "request", "--chunk-index", str(index))
        chunk = HookRequestChunk.model_validate_json(projected.stdout)
        content.append(chunk.content)
    canonical = "".join(content).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == reference.request_hash
    assert len(canonical) == reference.canonical_size_bytes

    after = HookSessionManifest.model_validate_json(
        manifest_file(tmp_path).read_text(encoding="utf-8")
    )
    assert after.receipt_sequence == before.receipt_sequence
    assert capability_value(tmp_path) == capability

    first_status = HookStatusProjection.model_validate_json(
        run_driver(tmp_path, "status").stdout
    )
    second_status = HookStatusProjection.model_validate_json(
        run_driver(tmp_path, "status").stdout
    )
    assert first_status == second_status
    status_after = HookSessionManifest.model_validate_json(
        manifest_file(tmp_path).read_text(encoding="utf-8")
    )
    assert status_after.receipt_sequence == before.receipt_sequence
    assert capability_value(tmp_path) == capability

    first_chunk = HookRequestChunk.model_validate_json(
        run_driver(tmp_path, "request", "--chunk-index", "0").stdout
    )
    post = {
        "session_id": "session-a",
        "turn_id": "turn-a",
        "cwd": str(tmp_path / "workspace"),
        "hook_event_name": "PostToolUse",
        "permission_mode": "default",
        "tool_name": "Bash",
        "tool_use_id": "request-chunk",
        "tool_input": {
            "command": (
                f"python '{ROOT / 'scripts' / 'dte_hook_driver_entry.py'}' "
                "hook-driver request --chunk-index 0"
            )
        },
        "tool_response": (
            "Chunk ID: probe\nWall time: 0.1 seconds\nProcess exited with code 0\n"
            "Original token count: 20\nOutput:\n" + first_chunk.model_dump_json()
        ),
    }
    verified = json.loads(run_hook(tmp_path, post).stdout)
    assert "Verified DTE request chunk" in verified["hookSpecificOutput"][
        "additionalContext"
    ]
    tampered = first_chunk.model_copy(update={"content": first_chunk.content + "x"})
    post["tool_response"] = tampered.model_dump_json()
    blocked = json.loads(run_hook(tmp_path, post).stdout)
    assert blocked["decision"] == "block"
    assert "verification failed" in blocked["reason"]


def test_identical_invocation_can_resume_in_a_new_session(tmp_path):
    assert run_hook(tmp_path, explicit_payload(tmp_path)).returncode == 0
    spec_path, nodes_path = write_inputs(tmp_path)
    initialized = run_driver(
        tmp_path,
        "init",
        "--spec",
        str(spec_path),
        "--nodes",
        str(nodes_path),
    )
    assert initialized.returncode == 0, initialized.stdout + initialized.stderr
    stepped = run_driver(tmp_path, "step")
    assert stepped.returncode == 0, stepped.stdout + stepped.stderr
    original = HookSessionManifest.model_validate_json(
        manifest_file(tmp_path).read_text(encoding="utf-8")
    )
    original_capability = capability_value(tmp_path)
    assert original.run_id and original.run_dir
    assert original.current_episode_id and original.current_attempt_id

    assert run_hook(
        tmp_path,
        explicit_payload(tmp_path, session="session-b", turn="turn-b"),
    ).returncode == 0
    available = run_driver(
        tmp_path,
        "init",
        "--spec",
        str(spec_path),
        "--nodes",
        str(nodes_path),
        session="session-b",
        turn="turn-b",
    )
    assert available.returncode == 0, available.stdout + available.stderr
    available_receipt = HookDriverReceipt.model_validate_json(available.stdout)
    assert available_receipt.controller_action == "resume_available"
    assert available_receipt.payload["run_id"] == original.run_id

    resumed = run_driver(
        tmp_path,
        "resume",
        "--run-id",
        original.run_id,
        session="session-b",
        turn="turn-b",
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    resumed_receipt = HookDriverReceipt.model_validate_json(resumed.stdout)
    assert resumed_receipt.operation == "resume"
    replacement = HookSessionManifest.model_validate_json(
        manifest_file(tmp_path, "session-b").read_text(encoding="utf-8")
    )
    assert replacement.run_id == original.run_id
    assert replacement.current_episode_id == original.current_episode_id
    assert replacement.current_attempt_id == original.current_attempt_id
    state = load_app_run(replacement.run_dir)
    assert state.execution_contract.enforcement_session_id == "session-b"

    transferred = HookSessionManifest.model_validate_json(
        manifest_file(tmp_path).read_text(encoding="utf-8")
    )
    assert transferred.phase == "transferred"
    stale = run_driver(
        tmp_path,
        "step",
        capability=original_capability,
        session="session-a",
        turn="turn-a",
    )
    assert stale.returncode == 1
    assert not HookDriverReceipt.model_validate_json(stale.stdout).success

    reference = HookRequestReference.model_validate_json(
        run_driver(
            tmp_path,
            "step",
            session="session-b",
            turn="turn-b",
        ).stdout
    )
    chunk = run_driver(
        tmp_path,
        "request",
        "--chunk-index",
        "0",
        session="session-b",
        turn="turn-b",
    )
    assert chunk.returncode == 0
    assert HookRequestChunk.model_validate_json(chunk.stdout).request_hash == reference.request_hash








def test_duplicate_hook_init_is_idempotent_and_reuses_one_run(tmp_path):
    assert run_hook(tmp_path, explicit_payload(tmp_path)).returncode == 0
    spec_path, nodes_path = write_inputs(tmp_path)
    first = run_driver(
        tmp_path,
        "init",
        "--spec",
        str(spec_path),
        "--nodes",
        str(nodes_path),
    )
    assert first.returncode == 0, first.stdout + first.stderr
    first_receipt = HookDriverReceipt.model_validate_json(first.stdout)
    first_run_dir = first_receipt.payload["run_dir"]

    duplicate = run_driver(
        tmp_path,
        "init",
        "--spec",
        str(spec_path),
        "--nodes",
        str(nodes_path),
    )
    assert duplicate.returncode == 0, duplicate.stdout + duplicate.stderr
    duplicate_receipt = HookDriverReceipt.model_validate_json(duplicate.stdout)
    assert duplicate_receipt.payload["duplicate_invocation"] is True
    assert duplicate_receipt.payload["run_dir"] == first_run_dir
    runs = [
        path
        for path in (tmp_path / "workspace" / ".dte" / "runs").iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ]
    assert runs == [Path(first_run_dir)]


def test_hook_init_requires_explicit_shared_or_strict_isolation_mode(tmp_path):
    assert run_hook(tmp_path, explicit_payload(tmp_path)).returncode == 0
    spec_path, nodes_path = write_inputs(tmp_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["role_isolation_mode"] = "legacy_unverified"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    rejected = run_driver(
        tmp_path,
        "init",
        "--spec",
        str(spec_path),
        "--nodes",
        str(nodes_path),
    )
    receipt = HookDriverReceipt.model_validate_json(rejected.stdout)
    assert receipt.success is False
    assert "explicit role_isolation_mode" in (receipt.error or "")




def test_receipt_manifest_transaction_recovers_after_manifest_write_failure(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("DTE_HOOK_STATE_ROOT", str(tmp_path / "hook-state"))
    hook_driver.activate_session(
        "session-a",
        "turn-a",
        str(tmp_path / "workspace"),
        source="explicit",
    )
    original_save_manifest = hook_driver.save_manifest

    def fail_second_manifest_write(manifest):
        if manifest.receipt_sequence == 2:
            raise OSError("injected manifest write failure")
        return original_save_manifest(manifest)

    monkeypatch.setattr(hook_driver, "save_manifest", fail_second_manifest_write)
    with pytest.raises(OSError, match="injected manifest"):
        hook_driver.resume_session_turn("session-a", "turn-b")
    monkeypatch.setattr(hook_driver, "save_manifest", original_save_manifest)

    recovered = hook_driver.load_manifest("session-a")
    assert recovered is not None
    assert recovered.active_root_turn_id == "turn-b"
    assert recovered.receipt_sequence == 2
    assert not hook_driver.transaction_path("session-a").exists()
    hook_driver.audit_manifest(recovered)


def test_pending_capability_rotation_recovers_manifest_and_receipt(
    tmp_path,
    monkeypatch,
):
    assert run_hook(tmp_path, explicit_payload(tmp_path)).returncode == 0
    spec_path, nodes_path = write_inputs(tmp_path)
    initialized = run_driver(
        tmp_path,
        "init",
        "--spec",
        str(spec_path),
        "--nodes",
        str(nodes_path),
    )
    assert initialized.returncode == 0, initialized.stdout + initialized.stderr
    monkeypatch.setenv("DTE_HOOK_STATE_ROOT", str(tmp_path / "hook-state"))
    before = hook_driver.load_manifest("session-a")
    assert before is not None and before.run_dir
    old_capability = capability_value(tmp_path)
    original_atomic_json = hook_driver._atomic_json

    def fail_receipt_write(path, payload):
        if path.parent == hook_driver.receipts_dir("session-a"):
            raise OSError("injected receipt write failure")
        return original_atomic_json(path, payload)

    monkeypatch.setattr(hook_driver, "_atomic_json", fail_receipt_write)
    with pytest.raises(OSError, match="injected receipt"):
        hook_driver.step_session("session-a", "turn-a", old_capability)

    raw_manifest = HookSessionManifest.model_validate_json(
        manifest_file(tmp_path).read_text(encoding="utf-8")
    )
    rotated_state = load_app_run(before.run_dir)
    capability_record = json.loads(
        hook_driver.capability_path("session-a").read_text(encoding="utf-8")
    )
    assert raw_manifest.capability_hash == before.capability_hash
    assert rotated_state.execution_contract.capability_hash != before.capability_hash
    assert capability_record["current"] == old_capability
    assert capability_record["pending"] is not None

    monkeypatch.setattr(hook_driver, "_atomic_json", original_atomic_json)
    recovered = hook_driver.load_manifest("session-a")
    assert recovered is not None
    assert recovered.receipt_sequence == before.receipt_sequence + 1
    assert recovered.phase == "episode_required"
    assert recovered.capability_hash == load_app_run(before.run_dir).execution_contract.capability_hash
    assert capability_value(tmp_path) != old_capability
    assert not hook_driver.transaction_path("session-a").exists()
    hook_driver.audit_manifest(recovered)


def test_initialized_run_recovers_pre_receipt_intent(tmp_path, monkeypatch):
    assert run_hook(tmp_path, explicit_payload(tmp_path)).returncode == 0
    spec_path, nodes_path = write_inputs(tmp_path)
    monkeypatch.setenv("DTE_HOOK_STATE_ROOT", str(tmp_path / "hook-state"))
    original_record_receipt = hook_driver._record_receipt

    def crash_before_receipt(*args, **kwargs):
        raise RuntimeError("injected init crash before normal receipt")

    monkeypatch.setattr(hook_driver, "_record_receipt", crash_before_receipt)
    with pytest.raises(RuntimeError, match="init crash"):
        hook_driver.init_session(
            "session-a",
            "turn-a",
            capability_value(tmp_path),
            str(spec_path),
            str(nodes_path),
        )
    raw_manifest = HookSessionManifest.model_validate_json(
        manifest_file(tmp_path).read_text(encoding="utf-8")
    )
    assert raw_manifest.phase == "awaiting_init"
    assert raw_manifest.run_dir is None
    intent = json.loads(
        hook_driver.transaction_path("session-a").read_text(encoding="utf-8")
    )
    assert intent["stage"] == "intent"
    assert Path(intent["operation_metadata"]["final_dir"]).is_dir()

    monkeypatch.setattr(hook_driver, "_record_receipt", original_record_receipt)
    recovered = hook_driver.load_manifest("session-a")
    assert recovered is not None and recovered.run_dir
    assert recovered.phase == "awaiting_controller"
    run_dir = Path(recovered.run_dir)
    assert (run_dir / "run_spec.json").is_file()
    assert (run_dir / "initial_nodes.json").is_file()
    recovery_receipt = HookDriverReceipt.model_validate_json(
        (
            hook_driver.receipts_dir("session-a")
            / f"{recovered.receipt_sequence:08d}-{recovered.last_receipt_hash}.json"
        ).read_text(encoding="utf-8")
    )
    assert recovery_receipt.operation == "recovery:init"
    assert recovery_receipt.payload["recovery"]["original_outcome_available"] is False
    hook_driver.audit_manifest(recovered)




def test_session_lock_reclaims_dead_owner(tmp_path, monkeypatch):
    monkeypatch.setenv("DTE_HOOK_STATE_ROOT", str(tmp_path / "hook-state"))
    lock = tmp_path / "hook-state" / "locks" / "session-a.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text(
        json.dumps(
            {
                "schema_version": hook_driver.LOCK_SCHEMA,
                "pid": 424242,
                "owner_token": "abandoned",
                "acquired_at": 0,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(hook_driver, "_pid_is_alive", lambda pid: False)

    with hook_driver.session_lock("session-a", timeout=0.1):
        owner = json.loads(lock.read_text(encoding="utf-8"))
        assert owner["pid"] == os.getpid()
        assert owner["owner_token"] != "abandoned"
    assert not lock.exists()


def test_activation_recovers_when_manifest_precedes_first_receipt(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("DTE_HOOK_STATE_ROOT", str(tmp_path / "hook-state"))
    original_record = hook_driver._record_receipt

    def crash_before_receipt(*args, **kwargs):
        raise OSError("injected crash before activation receipt")

    monkeypatch.setattr(hook_driver, "_record_receipt", crash_before_receipt)
    with pytest.raises(OSError, match="injected crash"):
        hook_driver.activate_session(
            "activation-recovery",
            "turn-1",
            str(tmp_path),
            source="explicit",
        )

    monkeypatch.setattr(hook_driver, "_record_receipt", original_record)
    recovered = hook_driver.load_manifest("activation-recovery")
    assert recovered is not None
    hook_driver.audit_manifest(recovered)
    assert recovered.receipt_sequence == 1
    receipt_path = next(
        hook_driver.receipts_dir("activation-recovery").glob("*.json")
    )
    receipt = HookDriverReceipt.model_validate_json(
        receipt_path.read_text(encoding="utf-8")
    )
    assert receipt.operation == "recovery:activate"
    assert receipt.payload["recovery"]["original_outcome_available"] is False


def test_precontract_persisted_run_loads_as_direct_legacy_without_rewrite(tmp_path):
    run_dir = tmp_path / "legacy-run"
    create_app_run(
        run_dir,
        DTERunSpec(problem="p", goal="g"),
        [SearchNode(node_id="seed", claim="candidate")],
        run_id="legacy",
    )
    state_path = run_dir / "app_run_state.json"
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    raw.pop("execution_contract")
    state_path.write_text(json.dumps(raw), encoding="utf-8")
    loaded = load_app_run(run_dir)
    assert loaded.execution_contract.mode == "direct_legacy"
    assert "execution_contract" not in json.loads(state_path.read_text(encoding="utf-8"))


def test_git_worktree_identity_tracks_all_material_local_states(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "DTE Test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "dte@example.invalid"],
        check=True,
    )
    tracked = repository / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "tracked.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-q", "-m", "baseline"],
        check=True,
    )

    clean = hook_driver._commit_worktree_identity(str(repository))
    assert hook_driver._commit_worktree_identity(str(repository)) == clean

    tracked.write_text("unstaged\n", encoding="utf-8")
    unstaged = hook_driver._commit_worktree_identity(str(repository))
    assert unstaged != clean
    subprocess.run(["git", "-C", str(repository), "add", "tracked.txt"], check=True)
    staged = hook_driver._commit_worktree_identity(str(repository))
    assert staged not in {clean, unstaged}

    untracked = repository / "untracked.bin"
    untracked.write_bytes(b"\x00material-untracked\xff")
    with_untracked = hook_driver._commit_worktree_identity(str(repository))
    assert with_untracked != staged

    info_exclude = repository / ".git" / "info" / "exclude"
    info_exclude.write_text(".dte/\n", encoding="utf-8")
    ignored = repository / ".dte" / "volatile.json"
    ignored.parent.mkdir()
    ignored.write_text("one", encoding="utf-8")
    ignored_identity = hook_driver._commit_worktree_identity(str(repository))
    ignored.write_text("two", encoding="utf-8")
    assert hook_driver._commit_worktree_identity(str(repository)) == ignored_identity

    linked = tmp_path / "linked"
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "worktree",
            "add",
            "-q",
            "-b",
            "linked-test",
            str(linked),
            "HEAD",
        ],
        check=True,
    )
    linked_clean = hook_driver._commit_worktree_identity(str(linked))
    assert linked_clean != clean
    (linked / "tracked.txt").write_text("linked dirty\n", encoding="utf-8")
    assert hook_driver._commit_worktree_identity(str(linked)) != linked_clean


def test_invocation_registry_recovers_stale_and_failed_generations(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("DTE_HOOK_STATE_ROOT", str(tmp_path / "hook-state"))
    invocation_key = "a" * 64
    registry = hook_driver.invocation_path(invocation_key)
    registry.parent.mkdir(parents=True)
    registry.write_text(
        json.dumps(
            {
                "schema_version": "dte-hook-invocation.v2",
                "status": "initializing",
                "invocation_key": invocation_key,
                "session_id": "dead-session",
                "generation": 3,
                "owner_pid": 424242,
                "owner_token": "dead-owner",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(hook_driver, "_pid_is_alive", lambda pid: False)

    stale_owner, existing = hook_driver._claim_invocation(
        invocation_key=invocation_key,
        session_id="retry-session",
        trigger_source="explicit",
        cwd=str(tmp_path),
    )
    assert existing is None
    assert stale_owner is not None
    assert stale_owner["generation"] == 4
    assert stale_owner["recovery_history"][-1]["reason"] == "stale_initializing_owner"

    hook_driver._atomic_json(
        registry,
        {
            **stale_owner,
            "status": "failed",
            "error_type": "InjectedFailure",
        },
    )
    failed_owner, existing = hook_driver._claim_invocation(
        invocation_key=invocation_key,
        session_id="retry-session",
        trigger_source="explicit",
        cwd=str(tmp_path),
    )
    assert existing is None
    assert failed_owner is not None
    assert failed_owner["generation"] == 5
    assert failed_owner["recovery_history"][-1]["reason"] == "retry_after_failure"


def test_invocation_registry_grants_single_concurrent_owner(tmp_path, monkeypatch):
    monkeypatch.setenv("DTE_HOOK_STATE_ROOT", str(tmp_path / "hook-state"))
    invocation_key = "b" * 64
    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def claim(session_id: str) -> None:
        barrier.wait()
        try:
            owner, _ = hook_driver._claim_invocation(
                invocation_key=invocation_key,
                session_id=session_id,
                trigger_source="explicit",
                cwd=str(tmp_path),
            )
        except RuntimeError:
            outcomes.append("blocked")
        else:
            assert owner is not None
            outcomes.append("owner")

    threads = [
        threading.Thread(target=claim, args=("session-one",)),
        threading.Thread(target=claim, args=("session-two",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(outcomes) == ["blocked", "owner"]
    registry = json.loads(
        hook_driver.invocation_path(invocation_key).read_text(encoding="utf-8")
    )
    assert registry["status"] == "initializing"
    assert registry["generation"] == 1
