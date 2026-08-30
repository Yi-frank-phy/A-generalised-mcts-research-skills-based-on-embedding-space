"""Workspace Agent adapter for the DTE strict backend protocol.

Workspace Agents currently do not expose a host lifecycle hook bus that a Skill
can register. This adapter therefore keeps strict state transitions inside the
DTE backend and transports the required session/turn/capability identity into
one backend command at a time.

Security / assurance boundary:
- backend state transitions, capability rotation, receipt hashing and commit
  validation remain strict and are re-audited after every mutating command;
- the host does NOT force the agent to invoke this wrapper, so this is not a
  replacement for Codex PreToolUse/Stop enforcement against a non-cooperating
  agent;
- Workspace runs must report host_hook_enforcement=false and
  isolation_verified=false unless the host independently attests otherwise.

Secrets are loaded without ever being printed. Existing environment variables
win. Otherwise the adapter checks DTE_SECRETS_FILE, the user-level DTE secret
file, and finally the gitignored checkout .env file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

CHECKOUT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CHECKOUT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

STATE_SCHEMA = "workspace-dte-session.v1"
TURN_SCHEMA = "workspace-dte-turn.v1"
ATTESTATION_SCHEMA = "workspace-dte-attestation.v1"
ALLOWED_SECRET_KEYS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
DEFAULT_STATE_ROOT = Path.home() / ".dte-workspace" / "state"


def default_secret_path() -> Path:
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "DTE" / "secrets.env"
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return config_home / "dte" / "secrets.env"


def _parse_env_file(path: Path) -> dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in ALLOWED_SECRET_KEYS:
            continue
        value = value.strip().strip('"').strip("'")
        if value:
            values[key] = value
    return values


def load_secrets() -> str:
    """Load allowed provider secrets without overriding the process env.

    Returns only a non-sensitive source label. Secret values are never logged.
    """

    if any(os.environ.get(key) for key in ALLOWED_SECRET_KEYS):
        return "environment"

    candidates: list[tuple[str, Path]] = []
    explicit = os.environ.get("DTE_SECRETS_FILE", "").strip()
    if explicit:
        candidates.append(("DTE_SECRETS_FILE", Path(explicit).expanduser()))
    candidates.append(("user-secret-file", default_secret_path()))
    candidates.append(("gitignored-workspace-dotenv", CHECKOUT_ROOT / ".env"))

    for source, path in candidates:
        values = _parse_env_file(path)
        if not values:
            continue
        for key, value in values.items():
            os.environ.setdefault(key, value)
        if any(os.environ.get(key) for key in ALLOWED_SECRET_KEYS):
            return source
    return "missing"


def _state_root(explicit: str | None = None) -> Path:
    if explicit:
        root = Path(explicit).expanduser().resolve()
    else:
        env = os.environ.get("DTE_WORKSPACE_STATE_ROOT", "").strip()
        root = Path(env).expanduser().resolve() if env else DEFAULT_STATE_ROOT.resolve()
    os.environ["DTE_HOOK_STATE_ROOT"] = str(root)
    return root


def _safe_component(value: str) -> str:
    if value and all(c.isalnum() or c in "-_" for c in value):
        return value
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _session_path(state_root: Path) -> Path:
    return state_root / "workspace-session.json"


def resolve_session_id(state_root: Path) -> tuple[str, str]:
    explicit = os.environ.get("DTE_WORKSPACE_SESSION_ID", "").strip()
    if explicit:
        return explicit, "DTE_WORKSPACE_SESSION_ID"

    path = _session_path(state_root)
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            value = raw.get("session_id")
            if isinstance(value, str) and value:
                return value, "sticky-workspace-session"
        except (OSError, ValueError):
            pass

    session_id = f"workspace-{uuid.uuid4()}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"schema_version": STATE_SCHEMA, "session_id": session_id},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return session_id, "sticky-workspace-session"


def _turn_path(state_root: Path, session_id: str) -> Path:
    return state_root / "turns" / f"{_safe_component(session_id)}.json"


def read_turn_id(state_root: Path, session_id: str, *, create: bool) -> tuple[str, bool]:
    path = _turn_path(state_root, session_id)
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            value = raw.get("turn_id")
            if isinstance(value, str) and value:
                return value, False
        except (OSError, ValueError):
            pass
    if not create:
        raise FileNotFoundError("no Workspace DTE root turn; run `workspace_driver.py new-turn`")
    turn_id = f"turn-{uuid.uuid4()}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": TURN_SCHEMA,
                "turn_id": turn_id,
                "rotated_at": time.time(),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return turn_id, True


def read_current_capability(state_root: Path, session_id: str) -> str:
    path = state_root / "capabilities" / f"{_safe_component(session_id)}.json"
    if not path.is_file():
        return ""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""

    from dte_backend.hook_driver import load_manifest

    manifest = load_manifest(session_id)
    manifest_hash = None if manifest is None else manifest.capability_hash
    for candidate in (raw.get("current"), raw.get("pending")):
        if not isinstance(candidate, str) or not candidate:
            continue
        if manifest_hash is None:
            return candidate
        digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
        if digest == manifest_hash:
            return candidate
    return ""


def _backend_env(state_root: Path, session_id: str, turn_id: str, capability: str) -> dict[str, str]:
    load_secrets()
    env = dict(os.environ)
    env["DTE_HOOK_STATE_ROOT"] = str(state_root)
    env["DTE_HOOK_SESSION_ID"] = session_id
    env["DTE_HOOK_TURN_ID"] = turn_id
    env["DTE_HOOK_CWD"] = os.getcwd()
    env["DTE_HOOK_CAPABILITY"] = capability
    env["DTE_WORKSPACE_ADAPTER"] = "1"
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SRC_ROOT) + (os.pathsep + existing if existing else "")
    return env


def _post_audit(session_id: str) -> dict[str, Any]:
    from dte_backend.hook_driver import audit_manifest, load_manifest

    manifest = load_manifest(session_id)
    if manifest is None:
        return {"audited": False, "reason": "no session manifest"}
    audit_manifest(manifest)
    return {
        "audited": True,
        "phase": manifest.phase,
        "run_id": manifest.run_id,
        "receipt_sequence": manifest.receipt_sequence,
        "next_required_action": manifest.next_required_action,
    }


def _attestation(*, receipt_chain_verified: bool) -> dict[str, Any]:
    return {
        "schema_version": ATTESTATION_SCHEMA,
        "backend_receipt_chain_verified": receipt_chain_verified,
        "host_hook_enforcement": False,
        "wrapper_use_host_enforced": False,
        "context_isolation_verified": False,
        "reasoning_effort_attested": False,
        "assurance": "backend_strict_host_best_effort",
    }


def cmd_preflight(args: argparse.Namespace) -> int:
    state_root = _state_root(args.state_root)
    source = load_secrets()
    session_id, session_source = resolve_session_id(state_root)
    report: dict[str, Any] = {
        "ok": True,
        "checkout_root": str(CHECKOUT_ROOT),
        "state_root": str(state_root),
        "session_id": session_id,
        "session_id_source": session_source,
        "gemini_secret_available": any(os.environ.get(k) for k in ALLOWED_SECRET_KEYS),
        "gemini_secret_source": source,
        "attestation": _attestation(receipt_chain_verified=False),
    }
    try:
        from dte_backend.bundle_manifest import verify_bundle_manifest

        manifest = verify_bundle_manifest(CHECKOUT_ROOT)
        report["bundle_sha256"] = manifest["bundle_sha256"]
    except Exception as exc:
        report["ok"] = False
        report["error"] = f"bundle verification failed: {exc}"
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


def cmd_new_turn(args: argparse.Namespace) -> int:
    state_root = _state_root(args.state_root)
    session_id, _ = resolve_session_id(state_root)
    turn_id = f"turn-{uuid.uuid4()}"
    path = _turn_path(state_root, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"schema_version": TURN_SCHEMA, "turn_id": turn_id, "rotated_at": time.time()},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"session_id": session_id, "turn_id": turn_id}, ensure_ascii=False))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    state_root = _state_root(args.state_root)
    session_id, session_source = resolve_session_id(state_root)
    try:
        turn_id, _ = read_turn_id(state_root, session_id, create=False)
    except FileNotFoundError:
        turn_id = None
    audit: dict[str, Any]
    verified = False
    try:
        audit = _post_audit(session_id)
        verified = bool(audit.get("audited"))
    except Exception as exc:
        audit = {"audited": False, "error": str(exc)}
    print(
        json.dumps(
            {
                "session_id": session_id,
                "session_id_source": session_source,
                "turn_id": turn_id,
                "audit": audit,
                "attestation": _attestation(receipt_chain_verified=verified),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if "error" not in audit else 2


def cmd_driver(args: argparse.Namespace) -> int:
    if not args.passthrough:
        raise SystemExit("driver requires `-- <hook-driver operation> [args...]`")
    state_root = _state_root(args.state_root)
    session_id, _ = resolve_session_id(state_root)
    turn_id, _ = read_turn_id(state_root, session_id, create=True)
    capability = read_current_capability(state_root, session_id)
    command = [sys.executable, "-m", "dte_backend", "hook-driver", *args.passthrough]
    if args.source and args.passthrough[0] == "activate":
        command += ["--source", args.source]
    completed = subprocess.run(command, env=_backend_env(state_root, session_id, turn_id, capability))
    try:
        audit = _post_audit(session_id)
        verified = bool(audit.get("audited"))
    except Exception as exc:
        audit = {"audited": False, "error": str(exc)}
        verified = False
    print(
        json.dumps(
            {
                "schema_version": "workspace-dte-post-audit.v1",
                **audit,
                "attestation": _attestation(receipt_chain_verified=verified),
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
    )
    if not verified and completed.returncode == 0:
        return 2
    return completed.returncode


def cmd_backend(args: argparse.Namespace) -> int:
    """Run support/read-only backend commands pinned to this checkout."""
    load_secrets()
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SRC_ROOT) + (os.pathsep + existing if existing else "")
    command = [sys.executable, "-m", "dte_backend", *args.passthrough]
    return subprocess.run(command, env=env, cwd=str(CHECKOUT_ROOT)).returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Workspace Agent adapter for the DTE strict backend")
    parser.add_argument("--state-root", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    preflight = sub.add_parser("preflight")
    preflight.set_defaults(func=cmd_preflight)
    new_turn = sub.add_parser("new-turn")
    new_turn.set_defaults(func=cmd_new_turn)
    status = sub.add_parser("status")
    status.set_defaults(func=cmd_status)

    driver = sub.add_parser("driver")
    driver.add_argument("--source", default="main_agent", choices=["explicit", "main_agent"])
    driver.set_defaults(func=cmd_driver)

    backend = sub.add_parser("backend")
    backend.set_defaults(func=cmd_backend)
    return parser


def main() -> int:
    argv = sys.argv[1:]
    tail: list[str] = []
    if "--" in argv:
        split_at = argv.index("--")
        head, tail = argv[:split_at], argv[split_at + 1 :]
    else:
        head = argv
    args = build_parser().parse_args(head)
    args.passthrough = [item for item in tail if item != "--"]
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
