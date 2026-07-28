import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import dte_backend.hook_driver as hook_driver
from dte_backend.app_driver import create_app_run, load_app_run, next_app_episode
from dte_backend.episode_models import (
    EpisodeRequest,
    EpisodeResult,
    JudgeEpisodeOutput,
    JudgeObservation,
    RuntimeDiagnostics,
    compute_output_hash,
)
from dte_backend.hook_driver import HookDriverReceipt, HookSessionManifest
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


def run_driver(tmp_path: Path, *arguments: str, capability: str | None = None):
    return subprocess.run(
        [sys.executable, "-m", "dte_backend", "hook-driver", *arguments],
        capture_output=True,
        text=True,
        env=driver_env(
            tmp_path,
            capability=capability if capability is not None else capability_value(tmp_path),
        ),
        cwd=ROOT,
    )


def write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    spec = json.loads((ROOT / "examples" / "run_spec.json").read_text(encoding="utf-8"))
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
    assert updated.endswith("python -m dte_backend hook-driver status")

    console_driver = {
        **base,
        "tool_input": {"command": "dte-backend hook-driver status"},
    }
    console_rewritten = json.loads(run_hook(tmp_path, console_driver).stdout)
    console_updated = console_rewritten["hookSpecificOutput"]["updatedInput"]["command"]
    assert "DTE_HOOK_CAPABILITY" in console_updated
    assert console_updated.endswith("dte-backend hook-driver status")

    quoted_driver = {
        **base,
        "tool_input": {
            "command": f"& '{sys.executable}' -m dte_backend hook-driver status"
        },
    }
    quoted_rewritten = json.loads(run_hook(tmp_path, quoted_driver).stdout)
    quoted_updated = quoted_rewritten["hookSpecificOutput"]["updatedInput"]["command"]
    assert "DTE_HOOK_CAPABILITY" in quoted_updated
    assert quoted_updated.endswith(
        f"& '{sys.executable}' -m dte_backend hook-driver status"
    )

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


def test_driver_init_step_receipts_and_backend_antibypass(tmp_path):
    assert run_hook(tmp_path, explicit_payload(tmp_path)).returncode == 0
    spec_path, nodes_path = write_inputs(tmp_path)
    initial_capability = capability_value(tmp_path)
    init = subprocess.run(
        [
            sys.executable,
            "-m",
            "dte_backend",
            "hook-driver",
            "init",
            "--spec",
            str(spec_path),
            "--nodes",
            str(nodes_path),
        ],
        capture_output=True,
        text=True,
        env=driver_env(tmp_path, capability=initial_capability),
        cwd=ROOT,
    )
    assert init.returncode == 0, init.stdout + init.stderr
    init_receipt = HookDriverReceipt.model_validate_json(init.stdout)
    assert init_receipt.success
    manifest = HookSessionManifest.model_validate_json(
        manifest_file(tmp_path).read_text(encoding="utf-8")
    )
    assert manifest.run_dir
    assert manifest.phase == "awaiting_controller"
    state = load_app_run(manifest.run_dir)
    assert state.execution_contract.mode == "hook_enforced_v1"
    with pytest.raises(PermissionError, match="hook-driver"):
        next_app_episode(manifest.run_dir)

    current_capability = capability_value(tmp_path)
    step = subprocess.run(
        [sys.executable, "-m", "dte_backend", "hook-driver", "step"],
        capture_output=True,
        text=True,
        env=driver_env(tmp_path, capability=current_capability),
        cwd=ROOT,
    )
    assert step.returncode == 0, step.stdout + step.stderr
    step_receipt = HookDriverReceipt.model_validate_json(step.stdout)
    assert step_receipt.success
    assert step_receipt.controller_action == "episode_required"
    assert step_receipt.payload["outcome"]["request"]["role"] == "judge"

    post = {
        "session_id": "session-a",
        "turn_id": "turn-a",
        "cwd": str(tmp_path / "workspace"),
        "hook_event_name": "PostToolUse",
        "permission_mode": "default",
        "tool_name": "Bash",
        "tool_use_id": "driver-step",
        "tool_input": {"command": "python -m dte_backend hook-driver step"},
        "tool_response": step_receipt.model_dump_json(),
    }
    verified = json.loads(run_hook(tmp_path, post).stdout)
    assert "hookSpecificOutput" in verified, verified
    assert "Verified DTE receipt" in verified["hookSpecificOutput"]["additionalContext"]
    tampered = step_receipt.model_dump(mode="json")
    tampered["after_state_hash"] = "f" * 64
    post["tool_response"] = json.dumps(tampered)
    rejected = json.loads(run_hook(tmp_path, post).stdout)
    assert rejected["decision"] == "block"
    assert "verification failed" in rejected["reason"]

    stale = subprocess.run(
        [sys.executable, "-m", "dte_backend", "hook-driver", "step"],
        capture_output=True,
        text=True,
        env=driver_env(tmp_path, capability=current_capability),
        cwd=ROOT,
    )
    assert stale.returncode == 1
    stale_receipt = HookDriverReceipt.model_validate_json(stale.stdout)
    assert not stale_receipt.success
    assert "stale or invalid" in stale_receipt.error


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


def test_terminal_handoff_is_generated_before_stop_allows_report(tmp_path):
    assert run_hook(tmp_path, explicit_payload(tmp_path)).returncode == 0
    spec_path, nodes_path = write_inputs(tmp_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["budget"]["max_committed_search_nodes"] = 1
    spec["budget"]["max_relation_enrichment_pairs"] = 0
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    initialized = run_driver(
        tmp_path,
        "init",
        "--spec",
        str(spec_path),
        "--nodes",
        str(nodes_path),
    )
    assert initialized.returncode == 0, initialized.stdout + initialized.stderr
    grant = run_driver(tmp_path, "step")
    grant_receipt = HookDriverReceipt.model_validate_json(grant.stdout)
    request = EpisodeRequest.model_validate(grant_receipt.payload["outcome"]["request"])
    output = JudgeEpisodeOutput(
        observations=[
            JudgeObservation(
                node_id=node_id,
                score=0.8,
                reasoning="bounded Judge observation",
                risks=[],
            )
            for node_id in request.selected_node_revisions
        ]
    )
    result = EpisodeResult(
        episode_id=request.episode_id,
        attempt_id=request.attempt_id,
        run_id=request.run_id,
        role="judge",
        input_graph_revision=request.input_graph_revision,
        selected_node_revisions=request.selected_node_revisions,
        status="completed",
        structured_output=output,
        runtime_diagnostics=RuntimeDiagnostics(
            adapter_name="codex-app-main-agent",
            transport_name="current-app-runtime",
            profile="native-autonomous",
            usage_source="unavailable",
        ),
        output_hash=compute_output_hash(output, request.output_schema_version),
        schema_version=request.output_schema_version,
    )
    result_path = tmp_path / "judge-result.json"
    result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    submitted = run_driver(tmp_path, "submit", "--result", str(result_path))
    assert HookDriverReceipt.model_validate_json(submitted.stdout).submission_accepted is True
    terminal = run_driver(tmp_path, "step")
    terminal_receipt = HookDriverReceipt.model_validate_json(terminal.stdout)
    assert terminal_receipt.controller_action == "ready_for_synthesis"

    manifest = HookSessionManifest.model_validate_json(
        manifest_file(tmp_path).read_text(encoding="utf-8")
    )
    assert manifest.phase == "terminal_pending_handoff"
    stop = {
        "session_id": "session-a",
        "turn_id": "turn-a",
        "cwd": str(tmp_path / "workspace"),
        "hook_event_name": "Stop",
        "permission_mode": "default",
        "stop_hook_active": False,
        "last_assistant_message": "done",
    }
    continued = json.loads(run_hook(tmp_path, stop).stdout)
    assert continued["decision"] == "block"
    assert "terminal handoff is now ready" in continued["reason"]
    manifest = HookSessionManifest.model_validate_json(
        manifest_file(tmp_path).read_text(encoding="utf-8")
    )
    assert manifest.phase == "handoff_ready"
    run_dir = Path(manifest.run_dir)
    assert (run_dir / "observability-summary.json").is_file()
    assert (run_dir / "epistemic-summary.json").is_file()
    assert (run_dir / "terminal-handoff.json").is_file()
    allowed = run_hook(tmp_path, stop)
    assert allowed.returncode == 0
    assert allowed.stdout == ""


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


def test_committed_submit_recovers_pre_receipt_intent_without_duplicate(
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
    granted = run_driver(tmp_path, "step")
    grant_receipt = HookDriverReceipt.model_validate_json(granted.stdout)
    request = EpisodeRequest.model_validate(grant_receipt.payload["outcome"]["request"])
    output = JudgeEpisodeOutput(
        observations=[
            JudgeObservation(
                node_id=node_id,
                score=0.8,
                reasoning="bounded Judge observation",
                risks=[],
            )
            for node_id in request.selected_node_revisions
        ]
    )
    result = EpisodeResult(
        episode_id=request.episode_id,
        attempt_id=request.attempt_id,
        run_id=request.run_id,
        role="judge",
        input_graph_revision=request.input_graph_revision,
        selected_node_revisions=request.selected_node_revisions,
        status="completed",
        structured_output=output,
        runtime_diagnostics=RuntimeDiagnostics(
            adapter_name="codex-app-main-agent",
            transport_name="current-app-runtime",
            profile="native-autonomous",
            usage_source="unavailable",
        ),
        output_hash=compute_output_hash(output, request.output_schema_version),
        schema_version=request.output_schema_version,
    )
    result_path = tmp_path / "crash-submit-result.json"
    result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    monkeypatch.setenv("DTE_HOOK_STATE_ROOT", str(tmp_path / "hook-state"))
    before = hook_driver.load_manifest("session-a")
    assert before is not None and before.run_dir
    capability = capability_value(tmp_path)
    original_record_receipt = hook_driver._record_receipt

    def crash_before_receipt(*args, **kwargs):
        raise RuntimeError("injected crash before normal receipt")

    monkeypatch.setattr(hook_driver, "_record_receipt", crash_before_receipt)
    with pytest.raises(RuntimeError, match="before normal receipt"):
        hook_driver.submit_session(
            "session-a",
            "turn-a",
            capability,
            str(result_path),
        )
    committed_state = load_app_run(before.run_dir)
    assert sum(
        attempt.status == "committed"
        for episode in committed_state.episodes
        for attempt in episode.attempts
    ) == 1
    intent = json.loads(
        hook_driver.transaction_path("session-a").read_text(encoding="utf-8")
    )
    assert intent["stage"] == "intent"
    assert intent["receipt_fields"]["operation"] == "submit"
    raw_manifest = HookSessionManifest.model_validate_json(
        manifest_file(tmp_path).read_text(encoding="utf-8")
    )
    assert raw_manifest.receipt_sequence == before.receipt_sequence

    monkeypatch.setattr(hook_driver, "_record_receipt", original_record_receipt)
    recovered = hook_driver.load_manifest("session-a")
    assert recovered is not None
    assert recovered.receipt_sequence == before.receipt_sequence + 1
    assert not hook_driver.transaction_path("session-a").exists()
    recovery_receipt = HookDriverReceipt.model_validate_json(
        (
            hook_driver.receipts_dir("session-a")
            / f"{recovered.receipt_sequence:08d}-{recovered.last_receipt_hash}.json"
        ).read_text(encoding="utf-8")
    )
    assert recovery_receipt.operation == "recovery:submit"
    assert recovery_receipt.submission_accepted is None
    assert recovery_receipt.payload["recovery"]["original_outcome_available"] is False
    hook_driver.audit_manifest(recovered)

    committed_before_retry = sum(
        attempt.status == "committed"
        for episode in load_app_run(before.run_dir).episodes
        for attempt in episode.attempts
    )
    with pytest.raises(ValueError, match="submit requires"):
        hook_driver.submit_session(
            "session-a",
            "turn-a",
            capability_value(tmp_path),
            str(result_path),
        )
    committed_after_retry = sum(
        attempt.status == "committed"
        for episode in load_app_run(before.run_dir).episodes
        for attempt in episode.attempts
    )
    assert committed_after_retry == committed_before_retry == 1


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

