"""Deterministic Codex hook driver for the App-native DTE protocol.

This module owns session/receipt mechanics only. Scientific role judgments and
controller transitions remain in :mod:`dte_backend.app_driver`.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Literal

from pydantic import Field

from .app_driver import (
    AppRunState,
    DriverExecutionContext,
    ExecutionContract,
    cancel_app_episode,
    create_app_run,
    load_app_run,
    next_app_episode,
    request_app_synthesis,
    retry_app_episode,
    rotate_app_execution_capability,
    submit_app_episode_result,
)
from .epistemic import build_terminal_epistemic_handoff
from .episode_models import canonical_json_bytes
from .guards import enforce_run_spec_guard
from .models import DTEBaseModel, DTERunSpec, SearchNode, SynthesisControlRequest
from .observability import build_run_observability_summary
from .validators import load_json_list, load_json_model


SESSION_SCHEMA = "dte-hook-session.v1"
RECEIPT_SCHEMA = "dte-hook-receipt.v1"
DRIVER_PROTOCOL = "hook-driver.v1"
ZERO_HASH = "0" * 64
TRANSACTION_SCHEMA = "dte-hook-transaction.internal.v1"
LOCK_SCHEMA = "dte-hook-lock.internal.v1"
MALFORMED_LOCK_STALE_SECONDS = 30.0

SessionPhase = Literal[
    "awaiting_init",
    "awaiting_controller",
    "episode_required",
    "awaiting_operator",
    "terminal_pending_handoff",
    "handoff_ready",
    "cancelled",
    "failed",
]


class HookSessionManifest(DTEBaseModel):
    schema_version: Literal["dte-hook-session.v1"] = SESSION_SCHEMA
    session_id: str
    active_root_turn_id: str
    activation_source: Literal["explicit", "main_agent"]
    cwd: str
    run_id: str | None = None
    run_dir: str | None = None
    phase: SessionPhase = "awaiting_init"
    current_episode_id: str | None = None
    current_attempt_id: str | None = None
    current_graph_revision: int | None = Field(default=None, ge=0)
    last_committed_graph_revision: int = Field(default=0, ge=0)
    last_receipt_hash: str = Field(default=ZERO_HASH, pattern=r"^[0-9a-f]{64}$")
    receipt_sequence: int = Field(default=0, ge=0)
    next_required_action: str
    protected_paths: list[str]
    capability_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_identity_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    repeated_stop_count: int = Field(default=0, ge=0)
    failure_reason: str | None = None
    trigger_source: str | None = None
    invocation_key: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    replay_of_run_id: str | None = None
    source_episode_result_hashes: list[str] = Field(default_factory=list)
    model_execution_disposition: Literal["reused", "rerun", "unknown"] | None = None


class HookDriverReceipt(DTEBaseModel):
    schema_version: Literal["dte-hook-receipt.v1"] = RECEIPT_SCHEMA
    operation: str
    success: bool
    before_state_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    after_state_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    session_id: str
    root_turn_id: str | None = None
    run_id: str | None = None
    episode_id: str | None = None
    attempt_id: str | None = None
    graph_revision: int | None = Field(default=None, ge=0)
    submission_accepted: bool | None = None
    controller_action: str | None = None
    next_required_action: str | None = None
    previous_receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    error: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


def state_root() -> Path:
    override = os.environ.get("DTE_HOOK_STATE_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / ".codex" / "dte-hook-state").resolve()


def _session_component(session_id: str) -> str:
    if session_id and all(c.isalnum() or c in "-_" for c in session_id):
        return session_id
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()


def manifest_path(session_id: str) -> Path:
    return state_root() / "sessions" / f"{_session_component(session_id)}.json"


def capability_path(session_id: str) -> Path:
    return state_root() / "capabilities" / f"{_session_component(session_id)}.json"


def receipts_dir(session_id: str) -> Path:
    return state_root() / "receipts" / _session_component(session_id)


def transaction_path(session_id: str) -> Path:
    return state_root() / "transactions" / f"{_session_component(session_id)}.json"


def invocation_path(invocation_key: str) -> Path:
    return state_root() / "invocations" / f"{invocation_key}.json"


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _try_reclaim_stale_lock(lock: Path) -> bool:
    """Remove a lock only when its recorded owner is no longer alive."""

    try:
        raw = lock.read_bytes()
        observed = lock.stat()
    except FileNotFoundError:
        return True
    owner_pid: int | None = None
    try:
        payload = json.loads(raw.decode("utf-8"))
        if isinstance(payload, dict):
            candidate = payload.get("pid")
            if isinstance(candidate, int):
                owner_pid = candidate
        elif isinstance(payload, int):
            owner_pid = payload
    except (UnicodeDecodeError, json.JSONDecodeError):
        try:
            owner_pid = int(raw.decode("ascii").strip())
        except (UnicodeDecodeError, ValueError):
            owner_pid = None
    if owner_pid is not None:
        if _pid_is_alive(owner_pid):
            return False
    elif time.time() - observed.st_mtime < MALFORMED_LOCK_STALE_SECONDS:
        # A process can die between O_EXCL creation and writing its owner
        # record. Give such an incomplete lock a short lease before recovery.
        return False
    try:
        current = lock.stat()
        if (
            current.st_mtime_ns != observed.st_mtime_ns
            or current.st_size != observed.st_size
            or lock.read_bytes() != raw
        ):
            return False
        lock.unlink()
        return True
    except FileNotFoundError:
        return True


def _release_owned_lock(lock: Path, owner_token: str) -> None:
    try:
        payload = json.loads(lock.read_text(encoding="utf-8"))
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError):
        return
    if isinstance(payload, dict) and payload.get("owner_token") == owner_token:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def session_lock(session_id: str, *, timeout: float = 10.0) -> Iterator[None]:
    lock = state_root() / "locks" / f"{_session_component(session_id)}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    descriptor: int | None = None
    owner_token = uuid.uuid4().hex
    owner_payload = json.dumps(
        {
            "schema_version": LOCK_SCHEMA,
            "pid": os.getpid(),
            "owner_token": owner_token,
            "acquired_at": time.time(),
        },
        sort_keys=True,
    ).encode("utf-8")
    while descriptor is None:
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            written = 0
            while written < len(owner_payload):
                written += os.write(descriptor, owner_payload[written:])
            os.fsync(descriptor)
        except FileExistsError:
            if _try_reclaim_stale_lock(lock):
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError("DTE hook session is locked by another transition")
            time.sleep(0.05)
    try:
        _recover_pending_transaction(session_id)
        _recover_interrupted_activation(session_id)
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
        _release_owned_lock(lock, owner_token)


def _load_manifest_unreconciled(session_id: str) -> HookSessionManifest | None:
    path = manifest_path(session_id)
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    manifest = HookSessionManifest.model_validate(raw)
    if manifest.session_id != session_id:
        raise ValueError("session manifest identity mismatch")
    return manifest


def load_manifest(session_id: str) -> HookSessionManifest | None:
    # A crashed writer leaves a durable transaction intent. Joining the session
    # lock completes it before any hook observes a split receipt/capability head.
    if transaction_path(session_id).is_file():
        with session_lock(session_id):
            pass
    manifest = _load_manifest_unreconciled(session_id)
    if manifest is not None and _needs_activation_receipt_recovery(manifest):
        with session_lock(session_id):
            pass
        manifest = _load_manifest_unreconciled(session_id)
    return manifest


def save_manifest(manifest: HookSessionManifest) -> None:
    _atomic_json(manifest_path(manifest.session_id), manifest.model_dump(mode="json"))


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _commit_worktree_identity(cwd: str) -> str:
    root = Path(cwd).resolve()
    git_entry = root / ".git"
    if git_entry.is_file():
        pointer = git_entry.read_text(encoding="utf-8", errors="replace").strip()
        if pointer.startswith("gitdir:"):
            git_entry = (root / pointer.split(":", 1)[1].strip()).resolve()
    head = ""
    if git_entry.is_dir() and (git_entry / "HEAD").is_file():
        head = (git_entry / "HEAD").read_text(
            encoding="utf-8", errors="replace"
        ).strip()
        if head.startswith("ref:"):
            ref_path = git_entry / head.split(":", 1)[1].strip()
            if ref_path.is_file():
                head = f"{head}:{ref_path.read_text(encoding='ascii').strip()}"
    return _canonical_hash({"cwd": str(root), "git_head": head or "unavailable"})


def hook_invocation_key(
    *,
    cwd: str,
    hook_type: str,
    spec: DTERunSpec,
    nodes: list[SearchNode],
    invocation_nonce: str | None = None,
    replay_of_run_id: str | None = None,
) -> str:
    return _canonical_hash(
        {
            "repository_identity": str(Path(cwd).resolve()),
            "commit_worktree_identity": _commit_worktree_identity(cwd),
            "hook_type": hook_type,
            "run_spec_hash": _canonical_hash(spec.model_dump(mode="json")),
            "initial_node_hash": _canonical_hash(
                [node.model_dump(mode="json") for node in nodes]
            ),
            "invocation_nonce": invocation_nonce,
            "replay_of_run_id": replay_of_run_id,
        }
    )


def _capability_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _manifest_identity(session_id: str, cwd: str, run_id: str | None, run_dir: str | None) -> str:
    return _canonical_hash(
        {
            "schema_version": SESSION_SCHEMA,
            "session_id": session_id,
            "cwd": str(Path(cwd).resolve()),
            "run_id": run_id,
            "run_dir": run_dir,
        }
    )


def _without_timestamps(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_timestamps(item)
            for key, item in value.items()
            if not key.endswith("_at") and key not in {"last_receipt_hash", "receipt_sequence"}
        }
    if isinstance(value, list):
        return [_without_timestamps(item) for item in value]
    return value


def state_identity_hash(manifest: HookSessionManifest) -> str:
    payload: dict[str, Any] = {
        "manifest": _without_timestamps(manifest.model_dump(mode="json")),
        "run_state": None,
    }
    if manifest.run_dir:
        state_path = Path(manifest.run_dir) / "app_run_state.json"
        if state_path.is_file():
            payload["run_state"] = _without_timestamps(
                json.loads(state_path.read_text(encoding="utf-8"))
            )
    return _canonical_hash(payload)


def _save_capability(session_id: str, current: str, pending: str | None = None) -> None:
    _atomic_json(
        capability_path(session_id),
        {"schema_version": "dte-hook-capability.v1", "current": current, "pending": pending},
    )


def load_capability(session_id: str, manifest: HookSessionManifest) -> str:
    raw = json.loads(capability_path(session_id).read_text(encoding="utf-8"))
    candidates = [raw.get("current"), raw.get("pending")]
    for candidate in candidates:
        if isinstance(candidate, str) and _capability_hash(candidate) == manifest.capability_hash:
            if raw.get("current") != candidate or raw.get("pending") is not None:
                _save_capability(session_id, candidate)
            return candidate
    raise PermissionError("persisted driver capability does not match the session manifest")


def _new_capability() -> str:
    return secrets.token_urlsafe(48)


def _receipt_hash(receipt_payload: dict[str, Any]) -> str:
    return _canonical_hash({key: value for key, value in receipt_payload.items() if key != "receipt_hash"})


def _needs_activation_receipt_recovery(manifest: HookSessionManifest) -> bool:
    """Recognize only the narrow pre-init state written by interrupted activation."""

    if (
        manifest.phase != "awaiting_init"
        or manifest.run_id is not None
        or manifest.run_dir is not None
        or manifest.current_episode_id is not None
        or manifest.current_attempt_id is not None
        or manifest.current_graph_revision is not None
        or manifest.last_committed_graph_revision != 0
        or manifest.manifest_identity_hash
        != _manifest_identity(manifest.session_id, manifest.cwd, None, None)
    ):
        return False
    load_capability(manifest.session_id, manifest)
    files = sorted(receipts_dir(manifest.session_id).glob("*.json"))
    if len(files) != manifest.receipt_sequence:
        return False
    if not files:
        return (
            manifest.receipt_sequence == 0
            and manifest.last_receipt_hash == ZERO_HASH
        )
    last = HookDriverReceipt.model_validate_json(files[-1].read_text(encoding="utf-8"))
    if (
        last.receipt_hash != manifest.last_receipt_hash
        or not files[-1].name.startswith(
            f"{manifest.receipt_sequence:08d}-{last.receipt_hash}"
        )
    ):
        return False
    return last.after_state_hash != state_identity_hash(manifest)


def _recover_interrupted_activation(session_id: str) -> None:
    manifest = _load_manifest_unreconciled(session_id)
    if manifest is None or not _needs_activation_receipt_recovery(manifest):
        return
    before_hash = ZERO_HASH
    if manifest.receipt_sequence:
        last_path = sorted(receipts_dir(session_id).glob("*.json"))[-1]
        last = HookDriverReceipt.model_validate_json(
            last_path.read_text(encoding="utf-8")
        )
        before_hash = last.after_state_hash
    _record_receipt(
        manifest,
        operation="recovery:activate",
        success=True,
        before_hash=before_hash,
        controller_action="awaiting_init",
        payload={
            "recovery": {
                "original_operation": "activate",
                "original_outcome_available": False,
                "reason": (
                    "activation manifest existed before its receipt was durably prepared"
                ),
            }
        },
    )


def _build_transaction_receipt(
    transaction: dict[str, Any],
    target: HookSessionManifest,
) -> HookDriverReceipt:
    fields = transaction["receipt_fields"]
    receipt_payload = {
        "schema_version": RECEIPT_SCHEMA,
        "operation": fields["operation"],
        "success": fields["success"],
        "before_state_hash": fields["before_hash"],
        "after_state_hash": state_identity_hash(target),
        "session_id": target.session_id,
        "root_turn_id": target.active_root_turn_id,
        "run_id": target.run_id,
        "episode_id": target.current_episode_id,
        "attempt_id": target.current_attempt_id,
        "graph_revision": target.last_committed_graph_revision,
        "submission_accepted": fields.get("accepted"),
        "controller_action": fields.get("controller_action"),
        "next_required_action": target.next_required_action,
        "previous_receipt_hash": transaction["previous_receipt_hash"],
        "receipt_hash": ZERO_HASH,
        "error": fields.get("error"),
        "payload": fields.get("payload") or {},
    }
    receipt_payload["receipt_hash"] = _receipt_hash(receipt_payload)
    return HookDriverReceipt.model_validate(receipt_payload)


def _discard_transaction(session_id: str) -> None:
    try:
        transaction_path(session_id).unlink()
    except FileNotFoundError:
        pass


def _begin_operation_intent(
    manifest: HookSessionManifest,
    *,
    operation: str,
    before_hash: str,
    current_capability: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Persist authority and replay facts before a backend call can mutate state."""

    path = transaction_path(manifest.session_id)
    if path.exists():
        raise RuntimeError("session already has a pending hook transaction")
    if (
        current_capability is not None
        and _capability_hash(current_capability) != manifest.capability_hash
    ):
        raise PermissionError("provided capability is stale or invalid")
    _atomic_json(
        path,
        {
            "schema_version": TRANSACTION_SCHEMA,
            "stage": "intent",
            "session_id": manifest.session_id,
            "previous_receipt_sequence": manifest.receipt_sequence,
            "previous_receipt_hash": manifest.last_receipt_hash,
            "baseline_manifest": manifest.model_dump(mode="json"),
            "target_manifest": None,
            "current_capability": current_capability,
            "next_capability": None,
            "receipt_fields": {
                "operation": operation,
                "before_hash": before_hash,
            },
            "operation_metadata": metadata or {},
        },
    )


def _prepare_recovery_transaction(
    transaction: dict[str, Any],
    target: HookSessionManifest,
    *,
    original_operation: str,
    controller_action: str | None,
    current_capability: str | None,
    payload: dict[str, Any],
) -> dict[str, Any]:
    next_capability: str | None = None
    if current_capability is not None:
        if not target.run_dir:
            raise ValueError("recovered capability rotation has no initialized run")
        next_capability = _new_capability()
        target.capability_hash = _capability_hash(next_capability)
    prepared = dict(transaction)
    prepared.update(
        {
            "stage": "prepared",
            "target_manifest": target.model_dump(mode="json"),
            "current_capability": current_capability,
            "next_capability": next_capability,
            "receipt_fields": {
                "operation": f"recovery:{original_operation}",
                "success": True,
                "before_hash": transaction["receipt_fields"]["before_hash"],
                "accepted": None,
                "controller_action": controller_action,
                "error": None,
                "payload": {
                    "recovery": {
                        "original_operation": original_operation,
                        "original_outcome_available": False,
                        **payload,
                    }
                },
            },
        }
    )
    _atomic_json(transaction_path(target.session_id), prepared)
    return prepared


def _recover_init_intent(
    transaction: dict[str, Any],
    baseline: HookSessionManifest,
) -> None:
    metadata = transaction.get("operation_metadata") or {}
    run_id = metadata.get("run_id")
    final_dir_raw = metadata.get("final_dir")
    temporary_dir_raw = metadata.get("temporary_dir")
    next_capability = metadata.get("next_capability")
    identity = metadata.get("manifest_identity_hash")
    if not all(
        isinstance(item, str) and item
        for item in (run_id, final_dir_raw, temporary_dir_raw, next_capability, identity)
    ):
        raise ValueError("init recovery intent is incomplete")
    final_dir = Path(final_dir_raw).resolve()
    temporary_dir = Path(temporary_dir_raw).resolve()
    runs_root = (Path(baseline.cwd) / ".dte" / "runs").resolve()
    if runs_root not in final_dir.parents or runs_root not in temporary_dir.parents:
        raise ValueError("init recovery path escaped the activation directory")
    final_state = final_dir / "app_run_state.json"
    temporary_state = temporary_dir / "app_run_state.json"
    if not final_state.is_file() and temporary_state.is_file():
        state = load_app_run(temporary_dir)
        _atomic_json(temporary_dir / "run_spec.json", state.spec.model_dump(mode="json"))
        _atomic_json(
            temporary_dir / "initial_nodes.json",
            [node.model_dump(mode="json") for node in state.initial_nodes],
        )
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary_dir.replace(final_dir)
    elif final_state.is_file():
        state = load_app_run(final_dir)
        _atomic_json(final_dir / "run_spec.json", state.spec.model_dump(mode="json"))
        _atomic_json(
            final_dir / "initial_nodes.json",
            [node.model_dump(mode="json") for node in state.initial_nodes],
        )
    else:
        if temporary_dir.is_dir():
            shutil.rmtree(temporary_dir)
        _discard_transaction(baseline.session_id)
        return
    if (
        state.run_id != run_id
        or state.execution_contract.mode != "hook_enforced_v1"
        or state.execution_contract.enforcement_session_id != baseline.session_id
        or state.execution_contract.manifest_identity_hash != identity
        or state.execution_contract.capability_hash != _capability_hash(next_capability)
    ):
        raise ValueError("persisted init state disagrees with its recovery intent")
    target = baseline.model_copy(deep=True)
    target.run_id = run_id
    target.run_dir = str(final_dir)
    target.manifest_identity_hash = identity
    target.capability_hash = _capability_hash(next_capability)
    target.protected_paths = _protected_paths(target.cwd, str(final_dir))
    _sync_from_state(target, state)
    _save_capability(target.session_id, next_capability)
    prepared = _prepare_recovery_transaction(
        transaction,
        target,
        original_operation="init",
        controller_action=state.controller_action,
        current_capability=None,
        payload={
            "reason": "initialized App state existed before the normal receipt was prepared",
            "run_dir": str(final_dir),
        },
    )
    _finish_receipt_transaction(target.session_id, prepared)


def _recover_handoff_intent(
    transaction: dict[str, Any],
    baseline: HookSessionManifest,
) -> None:
    if not baseline.run_dir:
        raise ValueError("handoff recovery has no initialized run")
    current_capability = transaction.get("current_capability")
    if (
        not isinstance(current_capability, str)
        or _capability_hash(current_capability) != baseline.capability_hash
    ):
        raise ValueError("handoff recovery capability is invalid")
    state = load_app_run(baseline.run_dir)
    if state.controller_action not in {"ready_for_synthesis", "run_complete"}:
        if state_identity_hash(baseline) == transaction["receipt_fields"]["before_hash"]:
            _discard_transaction(baseline.session_id)
            return
        raise ValueError("handoff recovery found a nonterminal backend state")
    target = baseline.model_copy(deep=True)
    _sync_from_state(target, state)
    handoff_paths = _materialize_handoff(target, state)
    prepared = _prepare_recovery_transaction(
        transaction,
        target,
        original_operation="handoff",
        controller_action=state.controller_action,
        current_capability=current_capability,
        payload={
            "reason": "terminal artifacts were rematerialized from authoritative state",
            **handoff_paths,
        },
    )
    _finish_receipt_transaction(target.session_id, prepared)


def _recover_operation_intent(transaction: dict[str, Any]) -> None:
    session_id = transaction.get("session_id")
    if not isinstance(session_id, str):
        raise ValueError("pending operation intent has no session identity")
    baseline = HookSessionManifest.model_validate(transaction.get("baseline_manifest"))
    if baseline.session_id != session_id:
        raise ValueError("pending operation intent manifest identity is invalid")
    operation = transaction.get("receipt_fields", {}).get("operation")
    if operation == "init":
        _recover_init_intent(transaction, baseline)
        return
    if operation == "handoff":
        _recover_handoff_intent(transaction, baseline)
        return
    if operation not in {
        "step",
        "submit",
        "status",
        "control:retry",
        "control:cancel",
        "control:request-synthesis",
    }:
        raise ValueError("pending operation intent names an unsupported transition")
    if not baseline.run_dir:
        raise ValueError("pending App operation has no initialized run")
    current_capability = transaction.get("current_capability")
    if (
        not isinstance(current_capability, str)
        or _capability_hash(current_capability) != baseline.capability_hash
    ):
        raise ValueError("pending App operation capability is invalid")
    before_hash = transaction["receipt_fields"]["before_hash"]
    if state_identity_hash(baseline) == before_hash:
        # The backend call did not durably change authoritative state. There is
        # no successful transition to reconstruct and retry remains safe.
        _discard_transaction(session_id)
        return
    state = load_app_run(baseline.run_dir)
    target = baseline.model_copy(deep=True)
    _sync_from_state(target, state)
    if operation == "control:cancel":
        target.phase = "cancelled"
        target.next_required_action = "report explicit cancellation; do not claim success"
    prepared = _prepare_recovery_transaction(
        transaction,
        target,
        original_operation=operation,
        controller_action=state.controller_action,
        current_capability=current_capability,
        payload={
            "reason": "backend state advanced before the normal receipt was prepared",
        },
    )
    _finish_receipt_transaction(session_id, prepared)


def _finish_receipt_transaction(
    session_id: str,
    transaction: dict[str, Any] | None = None,
) -> tuple[HookDriverReceipt, HookSessionManifest]:
    path = transaction_path(session_id)
    if transaction is None:
        transaction = json.loads(path.read_text(encoding="utf-8"))
    if (
        transaction.get("schema_version") != TRANSACTION_SCHEMA
        or transaction.get("session_id") != session_id
        or transaction.get("stage", "prepared") != "prepared"
    ):
        raise ValueError("pending hook transaction identity is invalid")
    persisted = _load_manifest_unreconciled(session_id)
    if persisted is None:
        raise ValueError("pending hook transaction has no session manifest")
    target = HookSessionManifest.model_validate(transaction["target_manifest"])
    previous_sequence = transaction["previous_receipt_sequence"]
    previous_hash = transaction["previous_receipt_hash"]
    if target.receipt_sequence != previous_sequence or target.last_receipt_hash != previous_hash:
        raise ValueError("pending hook transaction does not extend its recorded receipt head")

    current_capability = transaction.get("current_capability")
    next_capability = transaction.get("next_capability")
    if (current_capability is None) != (next_capability is None):
        raise ValueError("pending capability rotation is incomplete")
    if current_capability is not None:
        if _capability_hash(next_capability) != target.capability_hash:
            raise ValueError("pending capability rotation target is invalid")
        current_hash = _capability_hash(current_capability)
        next_hash = _capability_hash(next_capability)
        _save_capability(session_id, current_capability, next_capability)
        if target.run_dir:
            state = load_app_run(target.run_dir)
            durable_hash = state.execution_contract.capability_hash
            if durable_hash == current_hash:
                _rotate_capability(target, current_capability, next_capability)
            elif durable_hash != next_hash:
                raise ValueError(
                    "App capability disagrees with the pending hook transaction"
                )

    receipt = _build_transaction_receipt(transaction, target)
    committed = target.model_copy(deep=True)
    committed.receipt_sequence = previous_sequence + 1
    committed.last_receipt_hash = receipt.receipt_hash
    persisted_head = (persisted.receipt_sequence, persisted.last_receipt_hash)
    if persisted_head not in {
        (previous_sequence, previous_hash),
        (committed.receipt_sequence, committed.last_receipt_hash),
    }:
        raise ValueError("session manifest advanced past its pending hook transaction")
    receipt_path = receipts_dir(session_id) / (
        f"{committed.receipt_sequence:08d}-{receipt.receipt_hash}.json"
    )
    if receipt_path.is_file():
        existing = HookDriverReceipt.model_validate_json(
            receipt_path.read_text(encoding="utf-8")
        )
        if existing.model_dump(mode="json") != receipt.model_dump(mode="json"):
            raise ValueError("pending receipt conflicts with its persisted chain record")
    else:
        _atomic_json(receipt_path, receipt.model_dump(mode="json"))
    save_manifest(committed)
    if next_capability is not None:
        _save_capability(session_id, next_capability)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    return receipt, committed


def _recover_pending_transaction(session_id: str) -> None:
    path = transaction_path(session_id)
    if path.is_file():
        transaction = json.loads(path.read_text(encoding="utf-8"))
        if transaction.get("stage", "prepared") == "intent":
            _recover_operation_intent(transaction)
        else:
            _finish_receipt_transaction(session_id, transaction)


def _record_receipt(
    manifest: HookSessionManifest,
    *,
    operation: str,
    success: bool,
    before_hash: str,
    accepted: bool | None = None,
    controller_action: str | None = None,
    error: str | None = None,
    payload: dict[str, Any] | None = None,
    rotate_from: str | None = None,
) -> HookDriverReceipt:
    path = transaction_path(manifest.session_id)
    transaction: dict[str, Any] | None = None
    if path.exists():
        transaction = json.loads(path.read_text(encoding="utf-8"))
        if (
            transaction.get("stage") != "intent"
            or transaction.get("session_id") != manifest.session_id
            or transaction.get("receipt_fields", {}).get("operation") != operation
            or transaction.get("previous_receipt_sequence") != manifest.receipt_sequence
            or transaction.get("previous_receipt_hash") != manifest.last_receipt_hash
        ):
            raise RuntimeError("session already has a conflicting hook transaction")
    target = manifest.model_copy(deep=True)
    next_capability: str | None = None
    if rotate_from is not None:
        if _capability_hash(rotate_from) != target.capability_hash:
            raise PermissionError("provided capability is stale or invalid")
        next_capability = _new_capability()
        target.capability_hash = _capability_hash(next_capability)
    transaction = {
        **(transaction or {}),
        "schema_version": TRANSACTION_SCHEMA,
        "stage": "prepared",
        "session_id": manifest.session_id,
        "previous_receipt_sequence": manifest.receipt_sequence,
        "previous_receipt_hash": manifest.last_receipt_hash,
        "target_manifest": target.model_dump(mode="json"),
        "current_capability": rotate_from,
        "next_capability": next_capability,
        "receipt_fields": {
            "operation": operation,
            "success": success,
            "before_hash": before_hash,
            "accepted": accepted,
            "controller_action": controller_action,
            "error": error,
            "payload": payload or {},
        },
    }
    _atomic_json(path, transaction)
    receipt, committed = _finish_receipt_transaction(manifest.session_id, transaction)
    for field_name in HookSessionManifest.model_fields:
        setattr(manifest, field_name, getattr(committed, field_name))
    return receipt


def _protected_paths(cwd: str, run_dir: str | None = None) -> list[str]:
    skill_root = Path(__file__).resolve().parents[2]
    paths = [
        str(state_root()),
        str(skill_root / "hooks"),
        str(skill_root / "src" / "dte_backend"),
        str(skill_root / "scripts" / "install_dte_hooks.py"),
        str(Path.home() / ".codex" / "hooks" / "dte_enforcement_hook.py"),
    ]
    if run_dir:
        paths.append(str(Path(run_dir)))
    return [str(Path(item).resolve()) for item in paths]


def is_terminal_phase(phase: SessionPhase) -> bool:
    return phase in {"handoff_ready", "cancelled", "failed"}


def is_active_manifest(manifest: HookSessionManifest | None) -> bool:
    return manifest is not None and not is_terminal_phase(manifest.phase)


def _archive_terminal_manifest(manifest: HookSessionManifest) -> None:
    archive = state_root() / "archive" / _session_component(manifest.session_id)
    archive.mkdir(parents=True, exist_ok=True)
    target = archive / f"{manifest.receipt_sequence:08d}-{manifest.last_receipt_hash}.json"
    if not target.exists():
        _atomic_json(target, manifest.model_dump(mode="json"))


def activate_session(
    session_id: str,
    turn_id: str,
    cwd: str,
    *,
    source: Literal["explicit", "main_agent"],
    capability: str | None = None,
) -> HookDriverReceipt:
    if not session_id or not turn_id or not cwd:
        raise ValueError("activate requires session_id, turn_id, and cwd")
    cwd_path = Path(cwd).resolve()
    with session_lock(session_id):
        existing = load_manifest(session_id)
        if is_active_manifest(existing):
            assert existing is not None
            if capability is None or _capability_hash(capability) != existing.capability_hash:
                raise PermissionError("resuming activate requires the current driver capability")
            before = state_identity_hash(existing)
            existing.active_root_turn_id = turn_id
            existing.repeated_stop_count = 0
            return _record_receipt(
                existing,
                operation="activate",
                success=True,
                before_hash=before,
                controller_action=existing.phase,
                payload={"resumed": True},
                rotate_from=capability,
            )
        previous_hash = ZERO_HASH
        previous_sequence = 0
        before = ZERO_HASH
        if existing is not None:
            audit_manifest(existing)
            _archive_terminal_manifest(existing)
            previous_hash = existing.last_receipt_hash
            previous_sequence = existing.receipt_sequence
            before = state_identity_hash(existing)
        capability = _new_capability()
        identity = _manifest_identity(session_id, str(cwd_path), None, None)
        manifest = HookSessionManifest(
            session_id=session_id,
            active_root_turn_id=turn_id,
            activation_source=source,
            cwd=str(cwd_path),
            next_required_action="hook-driver init --spec <spec.json> --nodes <nodes.json>",
            protected_paths=_protected_paths(str(cwd_path)),
            capability_hash=_capability_hash(capability),
            manifest_identity_hash=identity,
            last_receipt_hash=previous_hash,
            receipt_sequence=previous_sequence,
        )
        _save_capability(session_id, capability)
        save_manifest(manifest)
        return _record_receipt(
            manifest,
            operation="activate",
            success=True,
            before_hash=before,
            controller_action="awaiting_init",
            payload={"resumed": False, "activation_source": source},
        )


def resume_session_turn(session_id: str, turn_id: str) -> HookDriverReceipt | None:
    with session_lock(session_id):
        manifest = load_manifest(session_id)
        if not is_active_manifest(manifest):
            return None
        assert manifest is not None
        if manifest.active_root_turn_id == turn_id:
            return None
        before = state_identity_hash(manifest)
        manifest.active_root_turn_id = turn_id
        manifest.repeated_stop_count = 0
        return _record_receipt(
            manifest,
            operation="resume-turn",
            success=True,
            before_hash=before,
            controller_action=manifest.phase,
        )


def _execution_context(manifest: HookSessionManifest, capability: str) -> DriverExecutionContext:
    if _capability_hash(capability) != manifest.capability_hash:
        raise PermissionError("provided capability is stale or invalid")
    return DriverExecutionContext(
        session_id=manifest.session_id,
        manifest_identity_hash=manifest.manifest_identity_hash,
        capability=capability,
    )


def _rotate_capability(
    manifest: HookSessionManifest,
    current: str,
    pending: str,
) -> None:
    """Apply the App half of a journaled capability rotation."""

    if not manifest.run_dir:
        raise ValueError("cannot rotate run capability before init")
    _save_capability(manifest.session_id, current, pending)
    rotate_app_execution_capability(
        manifest.run_dir,
        DriverExecutionContext(
            session_id=manifest.session_id,
            manifest_identity_hash=manifest.manifest_identity_hash,
            capability=current,
        ),
        pending,
    )


def _sync_from_state(manifest: HookSessionManifest, state: AppRunState) -> None:
    manifest.last_committed_graph_revision = state.graph_revision
    manifest.current_graph_revision = state.graph_revision
    manifest.current_episode_id = state.active_episode_id
    manifest.current_attempt_id = state.active_attempt_id
    action = state.controller_action
    if action == "episode_required":
        manifest.phase = "episode_required"
        manifest.next_required_action = (
            "execute the current EpisodeRequest and call hook-driver submit --result <result.json>"
        )
    elif action == "await_operator_decision":
        manifest.phase = "awaiting_operator"
        manifest.next_required_action = (
            "ask the user or call hook-driver control --action retry|cancel|request-synthesis"
        )
    elif action in {"ready_for_synthesis", "run_complete"}:
        manifest.phase = "terminal_pending_handoff"
        manifest.next_required_action = "hook-driver handoff"
        manifest.current_episode_id = None
        manifest.current_attempt_id = None
    else:
        manifest.phase = "awaiting_controller"
        manifest.next_required_action = "hook-driver step"
        manifest.current_episode_id = None
        manifest.current_attempt_id = None


def init_session(
    session_id: str,
    turn_id: str,
    capability: str,
    spec_path: str,
    nodes_path: str,
    *,
    invocation_nonce: str | None = None,
    replay_of_run_id: str | None = None,
) -> HookDriverReceipt:
    with session_lock(session_id):
        manifest = load_manifest(session_id)
        if manifest is None:
            raise ValueError("session is not awaiting initialization")
        if manifest.active_root_turn_id != turn_id:
            raise PermissionError("only the active root turn may initialize DTE")
        _execution_context(manifest, capability)
        spec = load_json_model(spec_path, DTERunSpec)
        if spec.role_isolation_mode == "legacy_unverified":
            raise ValueError(
                "hook-enforced App runs require an explicit role_isolation_mode: "
                "strict_fresh_context or shared_context_single_agent"
            )
        enforce_run_spec_guard(spec)
        nodes = load_json_list(nodes_path, SearchNode)
        invocation_key = hook_invocation_key(
            cwd=manifest.cwd,
            hook_type="init",
            spec=spec,
            nodes=nodes,
            invocation_nonce=invocation_nonce,
            replay_of_run_id=replay_of_run_id,
        )
        before = state_identity_hash(manifest)
        if manifest.phase != "awaiting_init":
            if manifest.invocation_key != invocation_key or manifest.run_dir is None:
                raise ValueError("session is not awaiting initialization")
            return _record_receipt(
                manifest,
                operation="init",
                success=True,
                before_hash=before,
                controller_action=manifest.phase,
                payload={
                    "run_dir": manifest.run_dir,
                    "duplicate_invocation": True,
                    "invocation_key": invocation_key,
                    "model_execution_disposition": "reused",
                },
            )
        registry = invocation_path(invocation_key)
        registry.parent.mkdir(parents=True, exist_ok=True)
        owner_payload = {
            "schema_version": "dte-hook-invocation.v1",
            "status": "initializing",
            "invocation_key": invocation_key,
            "session_id": session_id,
            "trigger_source": manifest.activation_source,
        }
        try:
            descriptor = os.open(
                registry, os.O_CREAT | os.O_EXCL | os.O_WRONLY
            )
        except FileExistsError:
            existing = json.loads(registry.read_text(encoding="utf-8"))
            if (
                existing.get("status") == "complete"
                and isinstance(existing.get("run_dir"), str)
            ):
                return _record_receipt(
                    manifest,
                    operation="init",
                    success=True,
                    before_hash=before,
                    controller_action="existing_run",
                    payload={
                        "run_dir": existing["run_dir"],
                        "run_id": existing.get("run_id"),
                        "duplicate_invocation": True,
                        "invocation_key": invocation_key,
                        "model_execution_disposition": "reused",
                    },
                )
            raise RuntimeError(
                "an identical hook invocation is already initializing"
            )
        else:
            encoded = json.dumps(owner_payload, sort_keys=True).encode("utf-8")
            try:
                os.write(descriptor, encoded)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        source_hashes: list[str] = []
        if replay_of_run_id is not None:
            source_dir = Path(manifest.cwd) / ".dte" / "runs" / replay_of_run_id
            source_state = load_app_run(source_dir)
            source_hashes = sorted(
                {
                    attempt.result_hash
                    for episode in source_state.episodes
                    for attempt in episode.attempts
                    if attempt.result_hash is not None
                }
            )
        run_id = str(uuid.uuid4())
        runs_root = Path(manifest.cwd) / ".dte" / "runs"
        final_dir = (runs_root / run_id).resolve()
        temporary_dir = (runs_root / f".{run_id}.tmp").resolve()
        if runs_root.resolve() not in temporary_dir.parents:
            raise ValueError("computed run path escaped the activation directory")
        next_capability = _new_capability()
        identity = _manifest_identity(session_id, manifest.cwd, run_id, str(final_dir))
        contract = ExecutionContract(
            mode="hook_enforced_v1",
            enforcement_session_id=session_id,
            activation_source=manifest.activation_source,
            manifest_identity_hash=identity,
            driver_protocol_version=DRIVER_PROTOCOL,
            capability_hash=_capability_hash(next_capability),
        )
        _begin_operation_intent(
            manifest,
            operation="init",
            before_hash=before,
            metadata={
                "run_id": run_id,
                "final_dir": str(final_dir),
                "temporary_dir": str(temporary_dir),
                "next_capability": next_capability,
                "manifest_identity_hash": identity,
            },
        )
        try:
            state = create_app_run(
                temporary_dir,
                spec,
                nodes,
                run_id=run_id,
                execution_contract=contract,
                creation_capability=next_capability,
                hook_trigger_source=manifest.activation_source,
                hook_invocation_key=invocation_key,
                replay_of_run_id=replay_of_run_id,
                source_episode_result_hashes=source_hashes,
                model_execution_disposition=(
                    "rerun" if replay_of_run_id is not None else "unknown"
                ),
            )
            _atomic_json(temporary_dir / "run_spec.json", spec.model_dump(mode="json"))
            _atomic_json(
                temporary_dir / "initial_nodes.json",
                [node.model_dump(mode="json") for node in nodes],
            )
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            temporary_dir.replace(final_dir)
        except Exception as exc:
            if temporary_dir.is_dir() and runs_root.resolve() in temporary_dir.parents:
                shutil.rmtree(temporary_dir)
            _atomic_json(
                registry,
                {
                    **owner_payload,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                },
            )
            raise
        _atomic_json(
            registry,
            {
                **owner_payload,
                "status": "complete",
                "run_id": run_id,
                "run_dir": str(final_dir),
                "replay_of_run_id": replay_of_run_id,
                "source_episode_result_hashes": source_hashes,
                "model_execution_disposition": (
                    "rerun" if replay_of_run_id is not None else "unknown"
                ),
            },
        )
        manifest.run_id = run_id
        manifest.run_dir = str(final_dir)
        manifest.trigger_source = manifest.activation_source
        manifest.invocation_key = invocation_key
        manifest.replay_of_run_id = replay_of_run_id
        manifest.source_episode_result_hashes = source_hashes
        manifest.model_execution_disposition = (
            "rerun" if replay_of_run_id is not None else "unknown"
        )
        manifest.manifest_identity_hash = identity
        manifest.capability_hash = _capability_hash(next_capability)
        manifest.protected_paths = _protected_paths(manifest.cwd, str(final_dir))
        _sync_from_state(manifest, state)
        _save_capability(session_id, next_capability)
        return _record_receipt(
            manifest,
            operation="init",
            success=True,
            before_hash=before,
            controller_action=state.controller_action,
            payload={
                "run_dir": str(final_dir),
                "invocation_key": invocation_key,
                "duplicate_invocation": False,
                "replay_of_run_id": replay_of_run_id,
                "source_episode_result_hashes": source_hashes,
                "model_execution_disposition": manifest.model_execution_disposition,
            },
        )


def step_session(session_id: str, turn_id: str, capability: str) -> HookDriverReceipt:
    with session_lock(session_id):
        manifest = _require_operable(session_id, turn_id, capability)
        if manifest.phase not in {"awaiting_controller", "episode_required"}:
            raise ValueError(f"step is invalid in phase={manifest.phase}")
        before = state_identity_hash(manifest)
        _begin_operation_intent(
            manifest,
            operation="step",
            before_hash=before,
            current_capability=capability,
        )
        outcome = next_app_episode(
            manifest.run_dir or "",
            execution_context=_execution_context(manifest, capability),
        )
        state = load_app_run(manifest.run_dir or "")
        _sync_from_state(manifest, state)
        return _record_receipt(
            manifest,
            operation="step",
            success=True,
            before_hash=before,
            controller_action=outcome.controller_action,
            payload={"outcome": outcome.model_dump(mode="json")},
            rotate_from=capability,
        )


def submit_session(
    session_id: str,
    turn_id: str,
    capability: str,
    result_path: str,
) -> HookDriverReceipt:
    with session_lock(session_id):
        manifest = _require_operable(session_id, turn_id, capability)
        if manifest.phase != "episode_required":
            raise ValueError("submit requires an active EpisodeRequest")
        raw = json.loads(Path(result_path).read_text(encoding="utf-8"))
        if raw.get("episode_id") != manifest.current_episode_id:
            raise ValueError("result episode_id does not match the manifest grant")
        if raw.get("attempt_id") != manifest.current_attempt_id:
            raise ValueError("result attempt_id does not match the manifest grant")
        if raw.get("input_graph_revision") != manifest.current_graph_revision:
            raise ValueError("result graph revision does not match the manifest grant")
        before = state_identity_hash(manifest)
        _begin_operation_intent(
            manifest,
            operation="submit",
            before_hash=before,
            current_capability=capability,
        )
        outcome = submit_app_episode_result(
            manifest.run_dir or "",
            raw,
            execution_context=_execution_context(manifest, capability),
        )
        state = load_app_run(manifest.run_dir or "")
        _sync_from_state(manifest, state)
        accepted = outcome.commit_outcome.accepted
        return _record_receipt(
            manifest,
            operation="submit",
            success=True,
            before_hash=before,
            accepted=accepted,
            controller_action=outcome.next_controller_action,
            payload={"outcome": outcome.model_dump(mode="json")},
            rotate_from=capability,
        )


def control_session(
    session_id: str,
    turn_id: str,
    capability: str,
    action: Literal["retry", "cancel", "request-synthesis"],
    *,
    reason: str | None = None,
    requested_by: Literal["user", "main_agent"] = "main_agent",
    scope: Literal["all", "node_ids"] = "all",
    node_ids: list[str] | None = None,
) -> HookDriverReceipt:
    with session_lock(session_id):
        manifest = _require_operable(session_id, turn_id, capability)
        before = state_identity_hash(manifest)
        context = _execution_context(manifest, capability)
        _begin_operation_intent(
            manifest,
            operation=f"control:{action}",
            before_hash=before,
            current_capability=capability,
            metadata={"action": action},
        )
        if action == "retry":
            if not manifest.current_episode_id:
                raise ValueError("retry requires a current logical episode")
            outcome = retry_app_episode(
                manifest.run_dir or "",
                manifest.current_episode_id,
                execution_context=context,
            )
            controller_action = outcome.controller_action
            payload = {"outcome": outcome.model_dump(mode="json")}
        elif action == "cancel":
            if not manifest.current_episode_id or not manifest.current_attempt_id:
                raise ValueError("cancel requires a current active attempt")
            outcome = cancel_app_episode(
                manifest.run_dir or "",
                manifest.current_episode_id,
                manifest.current_attempt_id,
                reason or "cancelled through the DTE hook driver",
                execution_context=context,
            )
            controller_action = outcome.controller_action
            payload = {"outcome": outcome.model_dump(mode="json")}
        else:
            request = SynthesisControlRequest(
                action="force_synthesis_after_current_task",
                requested_by=requested_by,
                reason=reason or "synthesis requested through the DTE hook driver",
                scope=scope,
                node_ids=node_ids or [],
            )
            state = request_app_synthesis(
                manifest.run_dir or "",
                request,
                execution_context=context,
            )
            controller_action = state.controller_action
            payload = {"control_request": request.model_dump(mode="json")}
        state = load_app_run(manifest.run_dir or "")
        _sync_from_state(manifest, state)
        if action == "cancel":
            manifest.phase = "cancelled"
            manifest.next_required_action = "report explicit cancellation; do not claim success"
        return _record_receipt(
            manifest,
            operation=f"control:{action}",
            success=True,
            before_hash=before,
            controller_action=controller_action,
            payload=payload,
            rotate_from=capability,
        )


def status_session(session_id: str, turn_id: str, capability: str) -> HookDriverReceipt:
    with session_lock(session_id):
        manifest = _require_operable(session_id, turn_id, capability, allow_terminal=True)
        before = state_identity_hash(manifest)
        _begin_operation_intent(
            manifest,
            operation="status",
            before_hash=before,
            current_capability=capability if manifest.run_dir else None,
        )
        state = None if not manifest.run_dir else load_app_run(manifest.run_dir)
        if state is not None and not is_terminal_phase(manifest.phase):
            _sync_from_state(manifest, state)
        return _record_receipt(
            manifest,
            operation="status",
            success=True,
            before_hash=before,
            controller_action=None if state is None else state.controller_action,
            payload={"manifest": manifest.model_dump(mode="json")},
            rotate_from=capability if state is not None else None,
        )


def _materialize_handoff(
    manifest: HookSessionManifest,
    state: AppRunState,
) -> dict[str, Any]:
    """Idempotently derive all terminal artifacts from authoritative App state."""

    if not manifest.run_dir:
        raise ValueError("handoff requires an initialized run")
    if state.controller_action not in {"ready_for_synthesis", "run_complete"}:
        raise ValueError("handoff requires a backend terminal action")
    observability = build_run_observability_summary(manifest.run_dir)
    epistemic = build_terminal_epistemic_handoff(manifest.run_dir)
    run_dir = Path(manifest.run_dir)
    observability_payload = observability.model_dump(mode="json")
    epistemic_payload = epistemic.model_dump(mode="json")
    _atomic_json(run_dir / "observability-summary.json", observability_payload)
    _atomic_json(run_dir / "epistemic-summary.json", epistemic_payload)
    handoff_payload = {
        "schema_version": "dte-terminal-handoff.v1",
        "run_id": manifest.run_id,
        "terminal_action": state.controller_action,
        "graph_revision": state.graph_revision,
        "observability_sha256": _canonical_hash(observability_payload),
        "epistemic_sha256": _canonical_hash(epistemic_payload),
        "observability_path": "observability-summary.json",
        "epistemic_path": "epistemic-summary.json",
    }
    _atomic_json(run_dir / "terminal-handoff.json", handoff_payload)
    manifest.phase = "handoff_ready"
    manifest.next_required_action = (
        "read terminal-handoff.json and its two summaries, then write the main-agent report"
    )
    manifest.current_episode_id = None
    manifest.current_attempt_id = None
    return {
        "handoff_path": str(run_dir / "terminal-handoff.json"),
        "observability_path": str(run_dir / "observability-summary.json"),
        "epistemic_path": str(run_dir / "epistemic-summary.json"),
    }


def handoff_session(
    session_id: str,
    turn_id: str,
    capability: str,
) -> HookDriverReceipt:
    with session_lock(session_id):
        manifest = _require_operable(session_id, turn_id, capability)
        if not manifest.run_dir:
            raise ValueError("handoff requires an initialized run")
        before = state_identity_hash(manifest)
        _begin_operation_intent(
            manifest,
            operation="handoff",
            before_hash=before,
            current_capability=capability,
        )
        state = load_app_run(manifest.run_dir)
        if state.controller_action not in {"ready_for_synthesis", "run_complete"}:
            raise ValueError("handoff requires a backend terminal action")
        handoff_paths = _materialize_handoff(manifest, state)
        return _record_receipt(
            manifest,
            operation="handoff",
            success=True,
            before_hash=before,
            controller_action=state.controller_action,
            payload=handoff_paths,
            rotate_from=capability,
        )


def _require_operable(
    session_id: str,
    turn_id: str,
    capability: str,
    *,
    allow_terminal: bool = False,
) -> HookSessionManifest:
    manifest = load_manifest(session_id)
    if manifest is None:
        raise ValueError("DTE hook session is not active")
    if not allow_terminal and is_terminal_phase(manifest.phase):
        raise ValueError(f"DTE hook session is terminal: {manifest.phase}")
    if manifest.active_root_turn_id != turn_id:
        raise PermissionError("only the active root turn may control DTE")
    if not manifest.run_dir and manifest.phase != "awaiting_init":
        raise ValueError("session manifest has no run directory")
    _execution_context(manifest, capability)
    return manifest


def validate_receipt(receipt: HookDriverReceipt, manifest: HookSessionManifest) -> None:
    payload = receipt.model_dump(mode="json")
    if _receipt_hash(payload) != receipt.receipt_hash:
        raise ValueError("receipt hash is invalid")
    if receipt.session_id != manifest.session_id:
        raise ValueError("receipt session does not match manifest")
    if receipt.root_turn_id != manifest.active_root_turn_id:
        raise ValueError("receipt turn does not match active root turn")
    if receipt.run_id != manifest.run_id:
        raise ValueError("receipt run does not match manifest")
    if receipt.receipt_hash != manifest.last_receipt_hash:
        raise ValueError("receipt is not the current chain head")
    if receipt.after_state_hash != state_identity_hash(manifest):
        raise ValueError("receipt state hash does not match persisted state")
    path = receipts_dir(manifest.session_id) / (
        f"{manifest.receipt_sequence:08d}-{receipt.receipt_hash}.json"
    )
    persisted = HookDriverReceipt.model_validate_json(path.read_text(encoding="utf-8"))
    if persisted.model_dump(mode="json") != receipt.model_dump(mode="json"):
        raise ValueError("receipt output differs from the persisted chain record")


def audit_manifest(manifest: HookSessionManifest) -> None:
    """Validate manifest/run identity, capability binding, and receipt chain."""

    expected_identity = _manifest_identity(
        manifest.session_id,
        manifest.cwd,
        manifest.run_id,
        manifest.run_dir,
    )
    if expected_identity != manifest.manifest_identity_hash:
        raise ValueError("manifest identity hash is invalid")
    load_capability(manifest.session_id, manifest)
    if manifest.run_id is not None or manifest.run_dir is not None:
        if not manifest.run_id or not manifest.run_dir:
            raise ValueError("manifest run identity is incomplete")
        expected_dir = (Path(manifest.cwd) / ".dte" / "runs" / manifest.run_id).resolve()
        if Path(manifest.run_dir).resolve() != expected_dir:
            raise ValueError("manifest run directory escaped its activation root")
        state = load_app_run(expected_dir)
        contract = state.execution_contract
        if state.run_id != manifest.run_id:
            raise ValueError("manifest and App state run IDs disagree")
        if contract.mode != "hook_enforced_v1":
            raise ValueError("active hook session points to a non-enforced run")
        if contract.enforcement_session_id != manifest.session_id:
            raise ValueError("App execution contract session mismatch")
        if contract.manifest_identity_hash != manifest.manifest_identity_hash:
            raise ValueError("App execution contract manifest mismatch")
        if contract.capability_hash != manifest.capability_hash:
            raise ValueError("App execution contract capability mismatch")
    files = sorted(receipts_dir(manifest.session_id).glob("*.json"))
    if len(files) != manifest.receipt_sequence:
        raise ValueError("receipt sequence count does not match persisted chain")
    previous = ZERO_HASH
    previous_after_hash: str | None = None
    last: HookDriverReceipt | None = None
    for index, path in enumerate(files, start=1):
        receipt = HookDriverReceipt.model_validate_json(path.read_text(encoding="utf-8"))
        if _receipt_hash(receipt.model_dump(mode="json")) != receipt.receipt_hash:
            raise ValueError(f"receipt {index} hash is invalid")
        if receipt.previous_receipt_hash != previous:
            raise ValueError(f"receipt {index} does not link to its predecessor")
        legacy_activation_reset = (
            index > 1
            and receipt.operation == "activate"
            and receipt.payload.get("resumed") is False
            and receipt.before_state_hash == ZERO_HASH
        )
        expected_before = ZERO_HASH if index == 1 else previous_after_hash
        if (
            receipt.before_state_hash != expected_before
            and not legacy_activation_reset
        ):
            raise ValueError(f"receipt {index} state transition is discontinuous")
        if not path.name.startswith(f"{index:08d}-{receipt.receipt_hash}"):
            raise ValueError(f"receipt {index} filename does not bind its identity")
        previous = receipt.receipt_hash
        previous_after_hash = receipt.after_state_hash
        last = receipt
    if last is None or last.receipt_hash != manifest.last_receipt_hash:
        raise ValueError("manifest receipt head is missing or inconsistent")
    if last.after_state_hash != state_identity_hash(manifest):
        raise ValueError("manifest state no longer matches the current receipt head")


def driver_environment() -> tuple[str, str, str, str]:
    session_id = os.environ.get("DTE_HOOK_SESSION_ID", "")
    turn_id = os.environ.get("DTE_HOOK_TURN_ID", "")
    cwd = os.environ.get("DTE_HOOK_CWD", os.getcwd())
    capability = os.environ.get("DTE_HOOK_CAPABILITY", "")
    if not session_id or not turn_id:
        raise PermissionError("hook-driver requires injected Codex session and root-turn identity")
    return session_id, turn_id, cwd, capability


def record_driver_failure(
    operation: str,
    session_id: str,
    turn_id: str | None,
    error: Exception | str,
) -> HookDriverReceipt:
    """Return a strict failure receipt without advancing protected run state."""

    message = str(error)
    manifest = load_manifest(session_id) if session_id else None
    if manifest is not None and turn_id == manifest.active_root_turn_id:
        with session_lock(session_id):
            manifest = load_manifest(session_id)
            assert manifest is not None
            before = state_identity_hash(manifest)
            return _record_receipt(
                manifest,
                operation=operation,
                success=False,
                before_hash=before,
                error=message,
                controller_action=manifest.phase,
            )
    payload = {
        "schema_version": RECEIPT_SCHEMA,
        "operation": operation,
        "success": False,
        "before_state_hash": ZERO_HASH,
        "after_state_hash": ZERO_HASH,
        "session_id": session_id or "unavailable",
        "root_turn_id": turn_id,
        "run_id": None,
        "episode_id": None,
        "attempt_id": None,
        "graph_revision": None,
        "submission_accepted": None,
        "controller_action": None,
        "next_required_action": None,
        "previous_receipt_hash": ZERO_HASH,
        "receipt_hash": ZERO_HASH,
        "error": message,
        "payload": {},
    }
    payload["receipt_hash"] = _receipt_hash(payload)
    return HookDriverReceipt.model_validate(payload)


def mark_stop_impasse(
    session_id: str,
    turn_id: str,
    reason: str,
    *,
    fatal: bool = False,
) -> HookDriverReceipt:
    """Persist a non-success stop recovery state instead of looping forever."""

    with session_lock(session_id):
        manifest = load_manifest(session_id)
        if manifest is None or manifest.active_root_turn_id != turn_id:
            raise PermissionError("stop recovery does not own the active root turn")
        before = state_identity_hash(manifest)
        manifest.repeated_stop_count += 1
        manifest.phase = "failed" if fatal else "awaiting_operator"
        manifest.failure_reason = reason
        manifest.next_required_action = (
            "report the explicit DTE enforcement failure"
            if fatal
            else "ask the user how to recover, retry, or cancel the current DTE run"
        )
        return _record_receipt(
            manifest,
            operation="stop-recovery",
            success=False,
            before_hash=before,
            controller_action=manifest.phase,
            error=reason,
        )
