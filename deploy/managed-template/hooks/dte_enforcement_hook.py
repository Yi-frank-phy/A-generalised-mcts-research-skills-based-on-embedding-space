"""Unified deterministic enforcement hook for App-native DTE sessions."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any


CONTENT_HASH_ARGUMENT = "--dte-hook-content-sha256="
SKILL_ROOT_ARGUMENT = "--dte-skill-root="


def _content_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _verify_content_binding(arguments: list[str]) -> str:
    """Fail before importing the backend when a trusted definition pins other bytes."""

    markers = [
        argument.removeprefix(CONTENT_HASH_ARGUMENT)
        for argument in arguments
        if argument.startswith(CONTENT_HASH_ARGUMENT)
    ]
    if not markers:
        # Direct development and unit-test execution remains supported. Installed
        # production definitions always include the marker and are verified by
        # the installer.
        return _content_sha256()
    if len(markers) != 1 or re.fullmatch(r"[0-9a-f]{64}", markers[0]) is None:
        raise RuntimeError("DTE hook definition contains an invalid content hash")
    actual = _content_sha256()
    if markers[0] != actual:
        raise RuntimeError(
            "DTE hook bytes differ from the content hash in the trusted definition"
        )
    return actual


# This check deliberately precedes all imports from the mutable Skill tree.
try:
    HOOK_CONTENT_SHA256 = _verify_content_binding(sys.argv[1:])
except Exception as exc:
    print(f"DTE enforcement hook failed closed: {exc}", file=sys.stderr)
    raise SystemExit(2) from exc


def _skill_root(arguments: list[str]) -> Path:
    pinned_roots = [
        argument.removeprefix(SKILL_ROOT_ARGUMENT)
        for argument in arguments
        if argument.startswith(SKILL_ROOT_ARGUMENT)
    ]
    if pinned_roots:
        if len(pinned_roots) != 1 or not pinned_roots[0].strip():
            raise RuntimeError("DTE hook definition contains an invalid Skill root")
        root = Path(pinned_roots[0]).expanduser().resolve()
        if not (root / "src" / "dte_backend" / "hook_driver.py").is_file():
            raise RuntimeError(
                "DTE hook definition points to a Skill root without hook_driver.py"
            )
        return root
    configured = os.environ.get("DTE_SKILL_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    colocated = Path(__file__).resolve().parents[1]
    if (colocated / "src" / "dte_backend" / "hook_driver.py").is_file():
        return colocated
    return (Path.home() / ".codex" / "skills" / "evolving-frontier-research").resolve()


SKILL_ROOT = _skill_root(sys.argv[1:])
SRC = SKILL_ROOT / "src"
# An older editable DTE checkout may also be installed. The enforcement hook
# must bind to the skill tree which supplied this dispatcher, not whichever
# `.pth` entry happens to sort first in site-packages.
filtered_path = []
for entry in sys.path:
    try:
        competing_package = (Path(entry) / "dte_backend").is_dir()
    except (OSError, TypeError):
        competing_package = False
    if not competing_package:
        filtered_path.append(entry)
sys.path[:] = [str(SRC), *filtered_path]

from dte_backend.hook_driver import (  # noqa: E402
    HookDriverReceipt,
    HookRequestChunk,
    HookRequestReference,
    HookStatusProjection,
    activate_session,
    audit_manifest,
    handoff_session,
    is_active_manifest,
    is_terminal_phase,
    load_capability,
    load_manifest,
    mark_stop_impasse,
    pause_session_turn,
    resume_session_turn,
    validate_request_projection,
    validate_receipt,
    validate_status_projection,
)


EXPLICIT_INVOCATION = re.compile(
    r"^\s*(?:"
    r"/(?:evolving-frontier-research)(?:\s|$)|"
    r"\$(?:evolving-frontier-research)(?:\s|$)|"
    r"\[\$?evolving-frontier-research\]\([^\r\n)]*[\\/]evolving-frontier-research[\\/]SKILL\.md\)(?:\s|$)"
    r")",
    re.IGNORECASE,
)
PROMPT_PROBE = re.compile(
    r"^\s*/evolving-frontier-research\s+--hook-probe\s+"
    r"([A-Za-z0-9][A-Za-z0-9_-]{7,63})\s*$",
    re.IGNORECASE,
)
COMMAND_PROBE = re.compile(
    r"^\s*dte-hook-probe\s+([A-Za-z0-9][A-Za-z0-9_-]{7,63})\s*$",
    re.IGNORECASE,
)

PYTHON_EXECUTABLE_PATTERN = (
    r"(?:"
    r'"[^"\r\n]*python(?:3(?:\.\d+)?)?(?:\.exe)?"|'
    r"'[^'\r\n]*python(?:3(?:\.\d+)?)?(?:\.exe)?'|"
    r"[^\s;&|\r\n]*python(?:3(?:\.\d+)?)?(?:\.exe)?|"
    r"py(?:\.exe)?"
    r")"
)
CONSOLE_EXECUTABLE_PATTERN = (
    r"(?:"
    r'"[^"\r\n]*dte-backend(?:\.exe)?"|'
    r"'[^'\r\n]*dte-backend(?:\.exe)?'|"
    r"[^\s;&|\r\n]*dte-backend(?:\.exe)?"
    r")"
)
DTE_ENTRYPOINT_PATTERN = (
    rf"(?:{PYTHON_EXECUTABLE_PATTERN}\s+-m\s+dte_backend|"
    rf"{CONSOLE_EXECUTABLE_PATTERN})"
)
WRAPPER_PATH_PATTERN = (
    r"(?:"
    r'"[^"\r\n]*dte_hook_driver_entry\.py"|'
    r"'[^'\r\n]*dte_hook_driver_entry\.py'|"
    r"[^\s;&|\r\n]*dte_hook_driver_entry\.py"
    r")"
)
PINNED_ENTRYPOINT_PATTERN = rf"{PYTHON_EXECUTABLE_PATTERN}\s+{WRAPPER_PATH_PATTERN}"
ANY_DTE_ENTRYPOINT_PATTERN = rf"(?:{DTE_ENTRYPOINT_PATTERN}|{PINNED_ENTRYPOINT_PATTERN})"
PYTHON_MODULE_ENTRYPOINT = re.compile(
    rf"(?P<python>{PYTHON_EXECUTABLE_PATTERN})\s+-m\s+dte_backend",
    re.IGNORECASE,
)
CONSOLE_ENTRYPOINT = re.compile(CONSOLE_EXECUTABLE_PATTERN, re.IGNORECASE)
DRIVER_ACTION_PATTERN = r"(activate|init|step|request|resume|submit|control|status|handoff)"
DRIVER_COMMAND = re.compile(
    rf"^\s*(?:&\s*)?{DTE_ENTRYPOINT_PATTERN}\s+hook-driver\s+"
    rf"{DRIVER_ACTION_PATTERN}\b[^\r\n;&|]*$",
    re.IGNORECASE,
)
DRIVER_ACTION_SEARCH = re.compile(
    rf"{ANY_DTE_ENTRYPOINT_PATTERN}\s+hook-driver\s+{DRIVER_ACTION_PATTERN}\b",
    re.IGNORECASE,
)
DIRECT_MUTATOR = re.compile(
    rf"{ANY_DTE_ENTRYPOINT_PATTERN}\s+"
    r"(?:create-run|next-episode|submit-episode-result|fail-episode|cancel-episode|"
    r"retry-episode|request-synthesis)\b",
    re.IGNORECASE,
)
STRICT_REAL = re.compile(
    rf"{ANY_DTE_ENTRYPOINT_PATTERN}\s+strict-run\b"
    r"(?=[^\r\n]*(?:--mode(?:=|\s+)real\b))",
    re.IGNORECASE,
)

SENSITIVE_BASENAMES = (
    "app_run_state.json",
    "episode_events.jsonl",
    "strict_run_control.json",
    "terminal-handoff.json",
    "dte-hook-state",
)


class HookInputError(ValueError):
    pass


def _read_input() -> dict[str, Any]:
    try:
        raw = json.load(sys.stdin)
    except Exception as exc:
        raise HookInputError(f"malformed DTE hook JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise HookInputError("DTE hook input must be a JSON object")
    for field in ("session_id", "cwd", "hook_event_name"):
        if not isinstance(raw.get(field), str) or not raw[field].strip():
            raise HookInputError(f"DTE hook input is missing {field}")
    return raw


def _emit(payload: dict[str, Any] | None) -> int:
    if payload is not None:
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    return 0


def _deny_pre(reason: str) -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def _context(event: str, text: str) -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": event,
            "additionalContext": text,
        }
    }


def _probe_ack(event: str, nonce: str) -> str:
    evidence = json.dumps(
        {
            "event": event,
            "content_sha256": HOOK_CONTENT_SHA256,
            "skill_root": str(SKILL_ROOT),
            "python": sys.executable,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"DTE_HOOK_PROBE_ACK:{nonce} {evidence}"


def _all_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [item for child in value.values() for item in _all_strings(child)]
    if isinstance(value, list):
        return [item for child in value for item in _all_strings(child)]
    return []


def _tool_command(payload: dict[str, Any]) -> str | None:
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return None
    for field in ("command", "cmd"):
        command = tool_input.get(field)
        if isinstance(command, str):
            return command
    return None


def _tool_command_field(payload: dict[str, Any]) -> str | None:
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return None
    for field in ("command", "cmd"):
        if isinstance(tool_input.get(field), str):
            return field
    return None


def _driver_action(command: str | None, *, strict: bool = True) -> str | None:
    if command is None:
        return None
    match = (
        DRIVER_COMMAND.fullmatch(command)
        if strict
        else DRIVER_ACTION_SEARCH.search(command)
    )
    return None if match is None else match.group(1).lower()


def _quote_env(value: str) -> str:
    return value.replace("'", "''")


def _pin_driver_entrypoint(command: str) -> str:
    """Replace ambiguous module/console resolution with the supplying wrapper."""

    wrapper = SKILL_ROOT / "scripts" / "dte_hook_driver_entry.py"
    if not wrapper.is_file():
        raise HookInputError("the supplying DTE Skill lacks its pinned driver wrapper")
    module_match = PYTHON_MODULE_ENTRYPOINT.search(command)
    if module_match is not None:
        wrapper_argument = (
            f"'{_quote_env(str(wrapper))}'"
            if os.name == "nt"
            else shlex.quote(str(wrapper))
        )
        replacement = f"{module_match.group('python')} {wrapper_argument}"
        return command[: module_match.start()] + replacement + command[module_match.end() :]

    console_match = CONSOLE_ENTRYPOINT.search(command)
    if console_match is None:
        raise HookInputError("DTE driver command has no recognized entrypoint")
    if os.name == "nt":
        python_argument = f"'{_quote_env(str(Path(sys.executable).resolve()))}'"
        wrapper_argument = f"'{_quote_env(str(wrapper))}'"
        prefix = command[: console_match.start()]
        call_operator = "" if prefix.rstrip().endswith("&") else "& "
        replacement = f"{call_operator}{python_argument} {wrapper_argument}"
    else:
        replacement = shlex.join([str(Path(sys.executable).resolve()), str(wrapper)])
    return command[: console_match.start()] + replacement + command[console_match.end() :]


def _rewrite_driver_command(command: str, payload: dict[str, Any], capability: str) -> str:
    values = {
        "DTE_HOOK_SESSION_ID": payload["session_id"],
        "DTE_HOOK_TURN_ID": payload["turn_id"],
        "DTE_HOOK_CWD": payload["cwd"],
        "DTE_HOOK_CAPABILITY": capability,
        "DTE_SKILL_ROOT": str(SKILL_ROOT),
        "PYTHONPATH": str(SRC),
    }
    pinned_command = _pin_driver_entrypoint(command)
    if os.name == "nt":
        prefix = "; ".join(
            f"$env:{key}='{_quote_env(value)}'" for key, value in values.items()
        )
        return f"{prefix}; {pinned_command}"
    prefix = " ".join(
        f"{key}='{value.replace(chr(39), chr(39) + chr(34) + chr(39) + chr(34) + chr(39))}'"
        for key, value in values.items()
    )
    return f"{prefix} {pinned_command}"


def _driver_shell_is_supported(payload: dict[str, Any]) -> bool:
    if os.name != "nt":
        return True
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return False
    shell = tool_input.get("shell")
    if shell is None:
        # Codex's Windows default shell is PowerShell for this workspace.
        return True
    if not isinstance(shell, str):
        return False
    normalized = shell.casefold()
    return "powershell" in normalized or "pwsh" in normalized


def _tool_workdir(payload: dict[str, Any]) -> Path:
    base = Path(payload["cwd"]).expanduser().resolve()
    tool_input = payload.get("tool_input")
    raw = tool_input.get("workdir") if isinstance(tool_input, dict) else None
    if not isinstance(raw, str) or not raw.strip():
        return base
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def _normalized_path(value: str | Path) -> str:
    return str(value).casefold().replace("/", "\\").rstrip("\\")


def _is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _mentions_path(payload: dict[str, Any], text: str, target: str | Path) -> bool:
    target_path = Path(target).expanduser().resolve()
    normalized_target = _normalized_path(target_path)
    if normalized_target and normalized_target in text:
        return True
    workdir = _tool_workdir(payload)
    if _is_within(workdir, target_path):
        # A tool already rooted inside protected state/source can address it
        # without spelling any path prefix.
        return True
    try:
        relative = _normalized_path(Path(os.path.relpath(target_path, workdir)))
    except ValueError:
        # Windows cannot form a relative path across drive letters. The
        # absolute-path check above remains authoritative in that case.
        return False
    return bool(relative and relative != "." and relative in text)


def _touches_protected_state(payload: dict[str, Any], protected_paths: list[str]) -> bool:
    text = "\n".join(_all_strings(payload.get("tool_input"))).casefold()
    normalized = text.replace("/", "\\")
    for path in protected_paths:
        if _mentions_path(payload, normalized, path):
            return True
    return any(name.casefold() in normalized for name in SENSITIVE_BASENAMES)


def _looks_like_protected_source_write(payload: dict[str, Any]) -> bool:
    tool_name = str(payload.get("tool_name", ""))
    text = "\n".join(_all_strings(payload.get("tool_input"))).casefold().replace("/", "\\")
    source_paths = (
        SKILL_ROOT / "hooks",
        SKILL_ROOT / "src" / "dte_backend",
        SKILL_ROOT / "scripts" / "install_dte_hooks.py",
    )
    if not any(_mentions_path(payload, text, path) for path in source_paths):
        return False
    if tool_name == "apply_patch":
        return True
    mutating_tokens = (
        "set-content",
        "add-content",
        "clear-content",
        "out-file",
        "new-item",
        "remove-item",
        "move-item",
        "copy-item",
        "rename-item",
        " >",
        ">>",
        "del ",
        "rm ",
        "mv ",
        "cp ",
        "tee ",
        "sed -i",
        "perl -pi",
        "git apply",
        "git checkout",
        "git restore",
        "git reset",
        "patch ",
        "touch ",
        "truncate ",
    )
    return any(token in text for token in mutating_tokens)


def _extract_json_objects(value: Any) -> list[dict[str, Any]]:
    """Extract wrapper-level JSON objects without descending into a receipt payload.

    Episode content is untrusted research data and may itself contain text which
    resembles a receipt. Once a strict receipt envelope is found, its nested
    payload must never be reparsed as another command result.
    """

    model_facing_schemas = {
        "dte-hook-receipt.v1",
        "dte-request-ref.v1",
        "dte-request-chunk.v1",
        "dte-hook-status.v1",
    }
    objects: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if value.get("schema_version") in model_facing_schemas:
            return [value]
        for child in value.values():
            objects.extend(_extract_json_objects(child))
    elif isinstance(value, list):
        for child in value:
            objects.extend(_extract_json_objects(child))
    elif isinstance(value, str):
        stripped = value.strip()
        try:
            parsed = json.loads(stripped)
        except Exception:
            rendered = re.fullmatch(
                r"(?:Chunk ID:[^\r\n]*\r?\n)?"
                r"Wall time:[^\r\n]*\r?\n"
                r"Process exited with code -?\d+\r?\n"
                r"(?:(?:Original token count:[^\r\n]*\r?\n)?Output:|"
                r"Final output:)\r?\n"
                r"(?P<stdout>[\s\S]*)",
                stripped,
            )
            if rendered is None:
                return objects
            try:
                parsed = json.loads(rendered.group("stdout").strip())
            except Exception:
                return objects
        if isinstance(parsed, dict):
            if parsed.get("schema_version") in model_facing_schemas:
                objects.append(parsed)
            else:
                objects.extend(_extract_json_objects(parsed))
        elif isinstance(parsed, list):
            objects.extend(_extract_json_objects(parsed))
    return objects


def handle_user_prompt(payload: dict[str, Any]) -> dict[str, Any] | None:
    prompt = payload.get("prompt")
    turn_id = payload.get("turn_id")
    if not isinstance(prompt, str) or not isinstance(turn_id, str) or not turn_id:
        raise HookInputError("UserPromptSubmit requires prompt and turn_id")
    probe = PROMPT_PROBE.fullmatch(prompt)
    if probe is not None:
        # A delivery probe must not create or resume a real DTE manifest.
        return _context(
            "UserPromptSubmit",
            _probe_ack("UserPromptSubmit", probe.group(1)),
        )
    session_id = payload["session_id"]
    manifest = load_manifest(session_id)
    explicit = EXPLICIT_INVOCATION.search(prompt) is not None
    if explicit and not is_active_manifest(manifest):
        receipt = activate_session(
            session_id,
            turn_id,
            payload["cwd"],
            source="explicit",
        )
        return _context(
            "UserPromptSubmit",
            "DTE enforcement activated. Before research, create validated spec and initial nodes, then call "
            "`python -m dte_backend hook-driver init --spec <spec.json> --nodes <nodes.json>`. "
            f"Activation receipt: {receipt.receipt_hash}.",
        )
    if is_active_manifest(manifest):
        assert manifest is not None
        audit_manifest(manifest)
        receipt = resume_session_turn(session_id, turn_id)
        manifest = load_manifest(session_id)
        assert manifest is not None
        return _context(
            "UserPromptSubmit",
            "DTE session resumed; do not create a parallel run. "
            f"run_id={manifest.run_id or 'not-initialized'} phase={manifest.phase} "
            f"episode_id={manifest.current_episode_id or 'none'} "
            f"attempt_id={manifest.current_attempt_id or 'none'} "
            f"next={manifest.next_required_action} "
            f"resume_receipt={None if receipt is None else receipt.receipt_hash}",
        )
    return None


def handle_pre_tool(payload: dict[str, Any]) -> dict[str, Any] | None:
    for field in ("turn_id", "tool_name", "tool_use_id", "tool_input"):
        if field not in payload:
            raise HookInputError(f"PreToolUse requires {field}")
    session_id = payload["session_id"]
    command = _tool_command(payload)
    probe = None if command is None else COMMAND_PROBE.fullmatch(command)
    if probe is not None:
        # Denial is the observable event-bus proof and guarantees the sentinel
        # command never reaches a shell.
        return _deny_pre(_probe_ack("PreToolUse", probe.group(1)))
    action = _driver_action(command)
    if action is not None and not _driver_shell_is_supported(payload):
        return _deny_pre(
            "DTE hook-driver commands on Windows require PowerShell so capability "
            "environment injection is unambiguous."
        )
    manifest = load_manifest(session_id)
    if not is_active_manifest(manifest):
        if action == "status" and manifest is not None:
            if payload["turn_id"] != manifest.active_root_turn_id:
                return _deny_pre("Only the owning root turn may inspect the terminal DTE session.")
            capability = load_capability(session_id, manifest)
        elif action == "activate":
            capability = ""
        else:
            return None
    else:
        assert manifest is not None
        if action is not None:
            if payload["turn_id"] != manifest.active_root_turn_id:
                return _deny_pre("Only the active root turn may call the DTE hook-driver.")
            capability = load_capability(session_id, manifest)
        else:
            text = command or "\n".join(_all_strings(payload.get("tool_input")))
            if DIRECT_MUTATOR.search(text):
                return _deny_pre("Direct DTE control CLI is disabled for an enforced App run; use hook-driver.")
            if STRICT_REAL.search(text):
                return _deny_pre("strict-run --mode real is headless legacy and cannot run inside an enforced App session.")
            if _touches_protected_state(payload, manifest.protected_paths):
                return _deny_pre("DTE manifest, run state, ledger, control, receipt, and handoff paths are driver-protected.")
            if _looks_like_protected_source_write(payload):
                return _deny_pre("DTE enforcement source files cannot be modified while an enforced run is active.")
            return None
    assert command is not None
    rewritten = _rewrite_driver_command(command, payload, capability)
    command_field = _tool_command_field(payload)
    assert command_field is not None
    updated_input = dict(payload["tool_input"])
    updated_input[command_field] = rewritten
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "updatedInput": updated_input,
        }
    }


def handle_post_tool(payload: dict[str, Any]) -> dict[str, Any] | None:
    for field in ("turn_id", "tool_name", "tool_use_id", "tool_input", "tool_response"):
        if field not in payload:
            raise HookInputError(f"PostToolUse requires {field}")
    action = _driver_action(_tool_command(payload), strict=False)
    if action is None:
        return None
    manifest = load_manifest(payload["session_id"])
    if manifest is None:
        return {"decision": "block", "reason": "DTE driver returned without a persisted session manifest."}
    receipts: list[HookDriverReceipt] = []
    references: list[HookRequestReference] = []
    chunks: list[HookRequestChunk] = []
    statuses: list[HookStatusProjection] = []
    for candidate in _extract_json_objects(payload["tool_response"]):
        try:
            schema = candidate.get("schema_version")
            if schema == "dte-hook-receipt.v1":
                receipts.append(HookDriverReceipt.model_validate(candidate))
            elif schema == "dte-request-ref.v1":
                references.append(HookRequestReference.model_validate(candidate))
            elif schema == "dte-request-chunk.v1":
                chunks.append(HookRequestChunk.model_validate(candidate))
            elif schema == "dte-hook-status.v1":
                statuses.append(HookStatusProjection.model_validate(candidate))
        except Exception:
            continue
    if action == "request":
        if len(chunks) != 1 or receipts or references or statuses:
            return {
                "decision": "block",
                "reason": "DTE request output did not contain exactly one strict request chunk; reread the chunk.",
            }
        try:
            validate_request_projection(chunks[0], manifest)
        except Exception as exc:
            return {
                "decision": "block",
                "reason": f"DTE request chunk verification failed: {exc}. Reread the chunk.",
            }
        return _context(
            "PostToolUse",
            "Verified DTE request chunk "
            f"{chunks[0].chunk_index + 1}/{chunks[0].chunk_count}; request_hash={chunks[0].request_hash}",
        )
    if (
        action == "step"
        and len(references) == 1
        and not receipts
        and not chunks
        and not statuses
    ):
        try:
            validate_request_projection(references[0], manifest)
        except Exception as exc:
            return {
                "decision": "block",
                "reason": f"DTE request reference verification failed: {exc}. Reread the request reference.",
            }
        return _context(
            "PostToolUse",
            "Verified existing DTE request reference; step was idempotent and did not consume a retry. "
            f"request_hash={references[0].request_hash}",
        )
    if action == "status":
        if len(statuses) != 1 or receipts or references or chunks:
            return {
                "decision": "block",
                "reason": "DTE status output did not contain exactly one strict status projection.",
            }
        try:
            validate_status_projection(statuses[0], manifest)
        except Exception as exc:
            return {
                "decision": "block",
                "reason": f"DTE status verification failed: {exc}.",
            }
        return _context(
            "PostToolUse",
            "Verified read-only DTE status; no capability or attempt was consumed. "
            f"phase={statuses[0].phase} next={statuses[0].next_required_action}",
        )
    if len(receipts) != 1:
        return {"decision": "block", "reason": "DTE driver output did not contain exactly one strict receipt; run hook-driver status for audit."}
    receipt = receipts[0]
    try:
        validate_receipt(receipt, manifest)
    except Exception as exc:
        return {"decision": "block", "reason": f"DTE receipt verification failed: {exc}. Run hook-driver status for audit."}
    expected = action if action != "control" else "control:"
    if not receipt.operation.startswith(expected):
        return {"decision": "block", "reason": "DTE receipt operation does not match the executed driver command."}
    if not receipt.success:
        return {"decision": "block", "reason": f"DTE driver failed without a successful state transition: {receipt.error}"}
    return _context(
        "PostToolUse",
        f"Verified DTE receipt {receipt.receipt_hash}; unique next action: {receipt.next_required_action}",
    )


def handle_stop(payload: dict[str, Any]) -> dict[str, Any] | None:
    turn_id = payload.get("turn_id")
    if not isinstance(turn_id, str) or not turn_id:
        raise HookInputError("Stop requires turn_id")
    manifest = load_manifest(payload["session_id"])
    if manifest is None or is_terminal_phase(manifest.phase):
        return None
    if manifest.phase == "awaiting_operator":
        return None
    if payload.get("stop_hook_active") is True:
        pause_session_turn(
            payload["session_id"],
            turn_id,
            "turn ended before the required DTE transition; preserve the active run for resume",
        )
        return None
    if manifest.phase == "terminal_pending_handoff":
        try:
            capability = load_capability(payload["session_id"], manifest)
            receipt = handoff_session(payload["session_id"], turn_id, capability)
        except Exception as exc:
            mark_stop_impasse(payload["session_id"], turn_id, f"terminal handoff failed: {exc}")
            return None
        return {
            "decision": "block",
            "reason": (
                "Authoritative DTE terminal handoff is now ready. Read terminal-handoff.json, "
                "observability-summary.json, and epistemic-summary.json, then write the final report. "
                f"Receipt: {receipt.receipt_hash}"
            ),
        }
    return {
        "decision": "block",
        "reason": (
            "DTE enforcement prevents an early final response. Perform the unique next action: "
            f"{manifest.next_required_action}"
            + (
                ". If the request display was truncated, wrapped, or lost chunks, "
                "reread the request chunks; do not fail or retry the scientific attempt"
                if manifest.phase == "episode_required"
                else ""
            )
        ),
    }


def handle_session_start(payload: dict[str, Any]) -> dict[str, Any] | None:
    source = payload.get("source")
    if source not in {"startup", "resume", "clear", "compact"}:
        raise HookInputError("SessionStart source is invalid")
    manifest = load_manifest(payload["session_id"])
    if not is_active_manifest(manifest):
        return None
    assert manifest is not None
    audit_manifest(manifest)
    return _context(
        "SessionStart",
        "Persistent DTE enforcement state restored from disk. "
        f"run_id={manifest.run_id or 'not-initialized'} phase={manifest.phase} "
        f"episode_id={manifest.current_episode_id or 'none'} "
        f"attempt_id={manifest.current_attempt_id or 'none'} "
        f"graph_revision={manifest.current_graph_revision} "
        f"unique_next_action={manifest.next_required_action}",
    )


def dispatch(payload: dict[str, Any]) -> dict[str, Any] | None:
    event = payload["hook_event_name"]
    if event == "UserPromptSubmit":
        return handle_user_prompt(payload)
    if event == "PreToolUse":
        return handle_pre_tool(payload)
    if event == "PostToolUse":
        return handle_post_tool(payload)
    if event == "Stop":
        return handle_stop(payload)
    if event == "SessionStart":
        return handle_session_start(payload)
    raise HookInputError(f"unsupported DTE hook event: {event}")


def main() -> int:
    try:
        payload = _read_input()
        return _emit(dispatch(payload))
    except Exception as exc:
        print(f"DTE enforcement hook failed closed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
