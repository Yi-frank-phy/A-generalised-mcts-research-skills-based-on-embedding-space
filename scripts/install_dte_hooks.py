"""Transactional installer for the unified DTE Codex lifecycle hook."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EVENTS = ("UserPromptSubmit", "PreToolUse", "PostToolUse", "Stop", "SessionStart")
TOOL_EVENTS = {"PreToolUse", "PostToolUse"}
HOOK_TIMEOUT = 30
HOOK_STATUS_MESSAGE = "Enforcing App-native DTE protocol"


def skill_root() -> Path:
    return Path(__file__).resolve().parents[1]


def codex_home() -> Path:
    return Path(os.environ.get("DTE_CODEX_HOME", Path.home() / ".codex")).resolve()


def source_hook() -> Path:
    return skill_root() / "hooks" / "dte_enforcement_hook.py"


def installed_hook() -> Path:
    return codex_home() / "hooks" / "dte_enforcement_hook.py"


def hooks_config() -> Path:
    return codex_home() / "hooks.json"


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _command(
    path: Path,
    content_sha256: str,
    pinned_skill_root: Path | None = None,
    python_executable: str | Path | None = None,
) -> str:
    if len(content_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in content_sha256
    ):
        raise ValueError("hook content SHA-256 must be 64 lowercase hexadecimal characters")
    # Codex trusts the hook definition, not the bytes behind a referenced path.
    # Pin both the dispatcher bytes and the exact Skill tree supplying the
    # backend so an old editable install or non-default Codex home cannot win.
    root = (pinned_skill_root or skill_root()).resolve()
    arguments = [
        str(python_executable or sys.executable),
        str(path),
        f"--dte-hook-content-sha256={content_sha256}",
        f"--dte-skill-root={root}",
    ]
    if os.name == "nt":
        quoted = " ".join(
            "'" + argument.replace("'", "''") + "'" for argument in arguments
        )
        return f"& {quoted}"
    return shlex.join(arguments)


def _handler_definition(
    target: Path,
    content_sha256: str,
    pinned_skill_root: Path | None = None,
) -> dict[str, Any]:
    return {
        "type": "command",
        "command": _command(target, content_sha256, pinned_skill_root),
        "timeout": HOOK_TIMEOUT,
        "statusMessage": HOOK_STATUS_MESSAGE,
    }


def _group_definition(
    event: str,
    target: Path,
    content_sha256: str,
    pinned_skill_root: Path | None = None,
) -> dict[str, Any]:
    group: dict[str, Any] = {
        "hooks": [
            dict(
                _handler_definition(
                    target,
                    content_sha256,
                    pinned_skill_root,
                )
            )
        ]
    }
    if event in TOOL_EVENTS:
        # "*" is the documented all-local-tools matcher. The canonical name for
        # the unified exec tool is "Bash", not the former "exec_command" token,
        # but production enforcement must cover tools beyond unified exec too.
        group["matcher"] = "*"
    elif event == "SessionStart":
        group["matcher"] = "startup|resume|clear|compact"
    return group


def _is_dte_handler(handler: Any) -> bool:
    if not isinstance(handler, dict):
        return False
    command = handler.get("command", "")
    return isinstance(command, str) and (
        "dte_prompt_guard.py" in command or "dte_enforcement_hook.py" in command
    )


def _merge_hooks(
    existing: dict[str, Any],
    target: Path,
    content_sha256: str,
    pinned_skill_root: Path | None = None,
) -> dict[str, Any]:
    merged = json.loads(json.dumps(existing))
    hooks = merged.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise ValueError("hooks.json field 'hooks' must be an object")
    for event, groups in list(hooks.items()):
        if not isinstance(groups, list):
            raise ValueError(f"hooks.{event} must be a list")
        retained_groups = []
        for group in groups:
            if not isinstance(group, dict):
                raise ValueError(f"hooks.{event} contains a non-object group")
            handlers = group.get("hooks", [])
            if not isinstance(handlers, list):
                raise ValueError(f"hooks.{event} group handlers must be a list")
            retained = [handler for handler in handlers if not _is_dte_handler(handler)]
            if retained:
                copy = dict(group)
                copy["hooks"] = retained
                retained_groups.append(copy)
        hooks[event] = retained_groups
    for event in EVENTS:
        hooks.setdefault(event, []).append(
            _group_definition(
                event,
                target,
                content_sha256,
                pinned_skill_root,
            )
        )
    return merged


def _same_installed_configuration(existing: dict[str, Any], merged: dict[str, Any]) -> bool:
    return (
        existing == merged
        and installed_hook().is_file()
        and _sha256(installed_hook()) == _sha256(source_hook())
        and not (codex_home() / "hooks" / "dte_prompt_guard.py").is_file()
    )


def _self_test(hook_path: Path) -> None:
    content_sha256 = _sha256(hook_path)
    subprocess.run(
        [sys.executable, "-m", "py_compile", str(hook_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    with tempfile.TemporaryDirectory(prefix="dte-hook-selftest-") as temp:
        payload = {
            "session_id": "installer-self-test",
            "cwd": str(skill_root()),
            "hook_event_name": "SessionStart",
            "source": "startup",
            "permission_mode": "default",
        }
        env = dict(os.environ)
        env["DTE_HOOK_STATE_ROOT"] = str(Path(temp) / "state")
        env.pop("DTE_SKILL_ROOT", None)
        completed = subprocess.run(
            [
                sys.executable,
                str(hook_path),
                f"--dte-hook-content-sha256={content_sha256}",
                f"--dte-skill-root={skill_root()}",
            ],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            env=env,
        )
        if completed.returncode != 0 or completed.stdout.strip():
            raise RuntimeError(
                "hook self-test failed: " + (completed.stderr.strip() or completed.stdout.strip())
            )


def _backend_import_diagnostics() -> dict[str, Any]:
    expected = (skill_root() / "src" / "dte_backend" / "__init__.py").resolve()
    spec = importlib.util.find_spec("dte_backend")
    origin = None if spec is None else spec.origin
    matches = False
    if isinstance(origin, str):
        try:
            matches = Path(origin).resolve() == expected
        except OSError:
            matches = False
    providers = sorted(
        importlib.metadata.packages_distributions().get("dte_backend", [])
    )
    warnings = []
    if not matches:
        warnings.append(
            "plain Python resolves dte_backend outside this Skill; the Hook pins "
            "the Skill source, but direct CLI use remains ambiguous"
        )
    if len(providers) > 1:
        warnings.append(
            "multiple installed distributions expose the dte_backend package: "
            + ", ".join(providers)
        )
    return {
        "expected_backend_origin": str(expected),
        "plain_python_backend_origin": origin,
        "plain_python_origin_matches_skill": matches,
        "backend_package_providers": providers,
        "warnings": warnings,
    }


def _path_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"existed": False, "sha256": None}
    return {"existed": True, "sha256": _sha256(path)}


def _backup(content_sha256: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = codex_home() / "hook-backups" / stamp
    directory.mkdir(parents=True, exist_ok=False)
    legacy = codex_home() / "hooks" / "dte_prompt_guard.py"
    tracked = [
        hooks_config(),
        legacy,
        installed_hook(),
    ]
    records = []
    for source in tracked:
        state = _path_state(source)
        record = {
            "path": str(source),
            **state,
            "backup_name": source.name,
        }
        if source.is_file():
            shutil.copy2(source, directory / source.name)
        records.append(record)
    _atomic_json(
        directory / "backup_manifest.json",
        {
            "schema_version": "dte-hook-backup.v2",
            "files": records,
            "hook_transaction": {
                "installed_handler": _handler_definition(
                    installed_hook(), content_sha256
                ),
                "post_install_files": {
                    str(installed_hook()): {
                        "existed": True,
                        "sha256": content_sha256,
                    },
                    str(legacy): {"existed": False, "sha256": None},
                },
            },
        },
    )
    return directory


def install_user() -> dict[str, Any]:
    _self_test(source_hook())
    content_sha256 = _sha256(source_hook())
    existing: dict[str, Any] = {}
    if hooks_config().is_file():
        existing = json.loads(hooks_config().read_text(encoding="utf-8"))
        if not isinstance(existing, dict):
            raise ValueError("existing hooks.json must contain a JSON object")
    merged = _merge_hooks(existing, installed_hook(), content_sha256)
    if _same_installed_configuration(existing, merged):
        result = verify_user()
        result.update(
            {
                "changed": False,
                "backup_dir": None,
                "trust_required": None,
                "next_action": (
                    "Confirm trust in /hooks, fully restart Codex, and run both "
                    "live event-delivery probes."
                ),
            }
        )
        return result
    backup = _backup(content_sha256)
    temporary_hook = installed_hook().with_name(".dte_enforcement_hook.py.tmp")
    try:
        installed_hook().parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_hook(), temporary_hook)
        _self_test(temporary_hook)
        temporary_hook.replace(installed_hook())
        _atomic_json(hooks_config(), merged)
        legacy = codex_home() / "hooks" / "dte_prompt_guard.py"
        if legacy.is_file():
            legacy.unlink()
        result = verify_user()
    except Exception:
        if temporary_hook.is_file():
            temporary_hook.unlink()
        _restore_backup_exact(backup)
        raise
    result["backup_dir"] = str(backup)
    result["changed"] = True
    result["trust_required"] = True
    result["next_action"] = (
        "Open /hooks and trust the changed content-bound definition, fully "
        "restart Codex, then run both live event-delivery probes."
    )
    return result


def _managed_requirements() -> str:
    windows_dir = r"C:\ProgramData\Codex\DTE\hooks"
    posix_dir = "/opt/codex/dte/hooks"
    content_sha256 = _sha256(source_hook())
    lines = [
        "[features]",
        "hooks = true",
        "",
        "[hooks]",
        f'managed_dir = "{posix_dir}"',
        f"windows_managed_dir = '{windows_dir}'",
        "",
    ]
    for event in EVENTS:
        lines.append(f"[[hooks.{event}]]")
        if event in TOOL_EVENTS:
            lines.append('matcher = "*"')
        elif event == "SessionStart":
            lines.append('matcher = "startup|resume|clear|compact"')
        lines.extend(
            [
                f"[[hooks.{event}.hooks]]",
                'type = "command"',
                f'command = "python3 {posix_dir}/dte_enforcement_hook.py '
                f'--dte-hook-content-sha256={content_sha256} '
                f'--dte-skill-root=/opt/codex/dte"',
                f"command_windows = 'python {windows_dir}\\dte_enforcement_hook.py "
                f"--dte-hook-content-sha256={content_sha256} "
                f"--dte-skill-root=C:\\ProgramData\\Codex\\DTE'",
                f"timeout = {HOOK_TIMEOUT}",
                f'statusMessage = "{HOOK_STATUS_MESSAGE}"',
                "",
            ]
        )
    return "\n".join(lines)


def install_managed_template() -> dict[str, Any]:
    _self_test(source_hook())
    output = skill_root() / "deploy" / "managed-template"
    hooks_dir = output / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_hook(), hooks_dir / "dte_enforcement_hook.py")
    _atomic_text(output / "requirements.toml", _managed_requirements())
    _atomic_text(
        output / "README.md",
        "# Managed DTE hook template\n\n"
        "Deploy `dte_enforcement_hook.py` to `C:\\ProgramData\\Codex\\DTE\\hooks`, "
        "deploy this skill's `src/dte_backend` tree to "
        "`C:\\ProgramData\\Codex\\DTE\\src\\dte_backend`, make the complete "
        "`C:\\ProgramData\\Codex\\DTE` tree read-only for ordinary users, and install "
        "`requirements.toml` through the administrator-managed Codex configuration layer. "
        "The trusted command pins that protected root with `--dte-skill-root`. "
        "This template intentionally does not set `allow_managed_hooks_only`.\n",
    )
    return {
        "scope": "managed-template",
        "output_dir": str(output),
        "administrator_install_performed": False,
    }


def verify_user() -> dict[str, Any]:
    errors: list[str] = []
    diagnostics = _backend_import_diagnostics()
    content_sha256 = _sha256(source_hook())
    if not installed_hook().is_file():
        errors.append("installed enforcement script is missing")
    elif _sha256(installed_hook()) != content_sha256:
        errors.append("installed enforcement script hash differs from the skill source")
    try:
        config = json.loads(hooks_config().read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"hooks.json cannot be read: {exc}")
        config = {}
    hooks = config.get("hooks", {}) if isinstance(config, dict) else {}
    if not isinstance(hooks, dict):
        errors.append("hooks.json field 'hooks' must be an object")
        hooks = {}
    for event in EVENTS:
        expected = _group_definition(event, installed_hook(), content_sha256)
        groups = hooks.get(event, [])
        if not isinstance(groups, list):
            errors.append(f"hooks.{event} must be a list")
            continue
        exact_groups = [group for group in groups if group == expected]
        dte_handlers = []
        malformed = False
        for group in groups:
            if not isinstance(group, dict):
                malformed = True
                continue
            handlers = group.get("hooks", [])
            if not isinstance(handlers, list):
                malformed = True
                continue
            dte_handlers.extend(
                handler for handler in handlers if _is_dte_handler(handler)
            )
        if malformed:
            errors.append(f"hooks.{event} contains a malformed hook group")
        if len(exact_groups) != 1:
            errors.append(
                f"{event} must contain exactly one complete DTE enforcement definition"
            )
        if len(dte_handlers) != 1:
            errors.append(
                f"{event} must contain exactly one DTE enforcement handler reference"
            )
    if installed_hook().is_file():
        try:
            _self_test(installed_hook())
        except Exception as exc:
            errors.append(str(exc))
    if (codex_home() / "hooks" / "dte_prompt_guard.py").is_file():
        errors.append("legacy dte_prompt_guard.py remains installed")
    result = {
        "scope": "user",
        "verified": not errors,
        "static_configuration_verified": not errors,
        "runtime_delivery_status": "requires_post_restart_probe",
        "hooks_config": str(hooks_config()),
        "installed_hook": str(installed_hook()),
        "errors": errors,
        **diagnostics,
    }
    if errors:
        raise RuntimeError(json.dumps(result, ensure_ascii=False))
    return result


def _restore_backup_exact(backup: Path) -> dict[str, Any]:
    """Restore the transaction snapshot after an incomplete install.

    This is intentionally stronger than the public rollback operation: it runs
    only inside the failed install transaction, before control returns to the
    caller, so restoring the exact pre-install snapshot is safe.
    """
    manifest = json.loads((backup / "backup_manifest.json").read_text(encoding="utf-8"))
    restored = []
    removed = []
    for record in manifest["files"]:
        target = Path(record["path"])
        if record["existed"]:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.rollback.tmp")
            shutil.copy2(backup / record["backup_name"], temporary)
            temporary.replace(target)
            restored.append(str(target))
        elif target.is_file():
            target.unlink()
            removed.append(str(target))
    return {"rolled_back_from": str(backup), "restored": restored, "removed": removed}


def _record_for(manifest: dict[str, Any], target: Path) -> dict[str, Any] | None:
    for record in manifest.get("files", []):
        if Path(record["path"]).resolve() == target.resolve():
            return record
    return None


def _backed_up_config(
    backup: Path,
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    record = _record_for(manifest, hooks_config())
    if record is None:
        raise ValueError("backup manifest does not describe hooks.json")
    if not record["existed"]:
        return {}, False
    payload = json.loads(
        (backup / record["backup_name"]).read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise ValueError("backed-up hooks.json must contain a JSON object")
    return payload, True


def _dte_fragments(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    hooks = config.get("hooks", {})
    if not isinstance(hooks, dict):
        raise ValueError("backed-up hooks.json field 'hooks' must be an object")
    fragments: dict[str, list[dict[str, Any]]] = {}
    for event, groups in hooks.items():
        if not isinstance(groups, list):
            raise ValueError(f"backed-up hooks.{event} must be a list")
        for group in groups:
            if not isinstance(group, dict):
                raise ValueError(f"backed-up hooks.{event} contains a non-object group")
            handlers = group.get("hooks", [])
            if not isinstance(handlers, list):
                raise ValueError(
                    f"backed-up hooks.{event} group handlers must be a list"
                )
            managed = [handler for handler in handlers if _is_dte_handler(handler)]
            if managed:
                fragment = {
                    key: json.loads(json.dumps(value))
                    for key, value in group.items()
                    if key != "hooks"
                }
                fragment["hooks"] = json.loads(json.dumps(managed))
                fragments.setdefault(event, []).append(fragment)
    return fragments


def _group_metadata(group: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in group.items() if key != "hooks"}


def _rollback_hooks_config(
    backup: Path,
    manifest: dict[str, Any],
) -> tuple[list[str], list[str], list[str], bool]:
    before, config_existed_before = _backed_up_config(backup, manifest)
    if hooks_config().is_file():
        current = json.loads(hooks_config().read_text(encoding="utf-8"))
        if not isinstance(current, dict):
            raise ValueError("current hooks.json must contain a JSON object")
    else:
        current = {}

    current_had_hooks = "hooks" in current
    current_hooks = current.get("hooks", {})
    if not isinstance(current_hooks, dict):
        raise ValueError(
            "current hooks.json field 'hooks' is not an object; refusing destructive rollback"
        )
    current_hooks = json.loads(json.dumps(current_hooks))

    transaction = manifest.get("hook_transaction", {})
    installed_handler = (
        transaction.get("installed_handler")
        if isinstance(transaction, dict)
        else None
    )
    exact_installed_handler = (
        installed_handler if isinstance(installed_handler, dict) else None
    )
    conflicts: list[str] = []

    for event, groups in list(current_hooks.items()):
        if not isinstance(groups, list):
            conflicts.append(f"hooks.{event} is not a list and was preserved")
            continue
        retained_groups = []
        for group in groups:
            if not isinstance(group, dict):
                conflicts.append(
                    f"hooks.{event} contains a non-object group that was preserved"
                )
                retained_groups.append(group)
                continue
            handlers = group.get("hooks", [])
            if not isinstance(handlers, list):
                conflicts.append(
                    f"hooks.{event} contains malformed handlers that were preserved"
                )
                retained_groups.append(group)
                continue
            if exact_installed_handler is None:
                retained = [
                    handler for handler in handlers if not _is_dte_handler(handler)
                ]
            else:
                retained = [
                    handler
                    for handler in handlers
                    if handler != exact_installed_handler
                ]
            removed_managed_handler = len(retained) != len(handlers)
            if not removed_managed_handler:
                retained_groups.append(group)
            elif retained:
                retained_group = dict(group)
                retained_group["hooks"] = retained
                retained_groups.append(retained_group)
        current_hooks[event] = retained_groups

    before_fragments = _dte_fragments(before)
    unexpected_dte_handler = False
    for event, fragments in before_fragments.items():
        groups = current_hooks.setdefault(event, [])
        if not isinstance(groups, list):
            conflicts.append(
                f"hooks.{event} could not restore the previous DTE definition"
            )
            unexpected_dte_handler = True
            continue
        residual_dte = any(
            _is_dte_handler(handler)
            for group in groups
            if isinstance(group, dict)
            for handler in (
                group.get("hooks", [])
                if isinstance(group.get("hooks", []), list)
                else []
            )
        )
        if residual_dte:
            conflicts.append(
                f"hooks.{event} has a post-install DTE edit; previous DTE hooks were not overlaid"
            )
            unexpected_dte_handler = True
            continue
        for fragment in fragments:
            metadata = _group_metadata(fragment)
            destination = next(
                (
                    group
                    for group in groups
                    if isinstance(group, dict)
                    and isinstance(group.get("hooks", []), list)
                    and _group_metadata(group) == metadata
                ),
                None,
            )
            if destination is None:
                groups.append(json.loads(json.dumps(fragment)))
            else:
                destination["hooks"].extend(
                    handler
                    for handler in fragment["hooks"]
                    if handler not in destination["hooks"]
                )

    # A changed managed definition which did not exist before installation is
    # user-visible post-install state. Preserve it and keep the installed script.
    for event, groups in current_hooks.items():
        if event in before_fragments or not isinstance(groups, list):
            continue
        if any(
            _is_dte_handler(handler)
            for group in groups
            if isinstance(group, dict)
            for handler in (
                group.get("hooks", [])
                if isinstance(group.get("hooks", []), list)
                else []
            )
        ):
            conflicts.append(
                f"hooks.{event} has a post-install DTE edit that was preserved"
            )
            unexpected_dte_handler = True

    before_hooks = before.get("hooks", {})
    if not isinstance(before_hooks, dict):
        raise ValueError("backed-up hooks.json field 'hooks' must be an object")
    for event in EVENTS:
        if event not in before_hooks and current_hooks.get(event) == []:
            current_hooks.pop(event, None)

    if "hooks" in before or current_had_hooks or current_hooks:
        current["hooks"] = current_hooks
    else:
        current.pop("hooks", None)

    restored: list[str] = []
    removed: list[str] = []
    if not config_existed_before and not current:
        if hooks_config().is_file():
            hooks_config().unlink()
            removed.append(str(hooks_config()))
    else:
        _atomic_json(hooks_config(), current)
        restored.append(str(hooks_config()))
    return restored, removed, conflicts, unexpected_dte_handler


def _state_matches(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        bool(left.get("existed")) == bool(right.get("existed"))
        and left.get("sha256") == right.get("sha256")
    )


def _rollback_files(
    backup: Path,
    manifest: dict[str, Any],
    *,
    preserve_installed_hook: bool,
) -> tuple[list[str], list[str], list[str]]:
    transaction = manifest.get("hook_transaction", {})
    post_install_files = (
        transaction.get("post_install_files", {})
        if isinstance(transaction, dict)
        else {}
    )
    if not isinstance(post_install_files, dict):
        post_install_files = {}

    restored: list[str] = []
    removed: list[str] = []
    conflicts: list[str] = []
    for record in manifest.get("files", []):
        target = Path(record["path"])
        if target.resolve() == hooks_config().resolve():
            continue
        if preserve_installed_hook and target.resolve() == installed_hook().resolve():
            conflicts.append(
                f"{target} was preserved because its hook definition was edited after install"
            )
            continue

        before_state = {
            "existed": bool(record.get("existed")),
            "sha256": record.get("sha256"),
        }
        if before_state["existed"] and before_state["sha256"] is None:
            before_state["sha256"] = _sha256(backup / record["backup_name"])
        current_state = _path_state(target)
        if _state_matches(current_state, before_state):
            continue

        expected_state = post_install_files.get(str(target))
        if expected_state is None:
            # A v1 backup did not record the post-install enforcement-script
            # hash.  We therefore cannot distinguish the installer-owned copy
            # from a later user edit and must preserve it rather than guessing.
            if target.resolve() == installed_hook().resolve():
                conflicts.append(
                    f"{target} has no recorded post-install hash and was preserved"
                )
                continue
            else:
                # The legacy prompt guard was removed by every v1 install.
                expected_state = {"existed": False, "sha256": None}
        if not isinstance(expected_state, dict) or not _state_matches(
            current_state, expected_state
        ):
            conflicts.append(f"{target} has post-install changes and was preserved")
            continue

        if before_state["existed"]:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.rollback.tmp")
            shutil.copy2(backup / record["backup_name"], temporary)
            temporary.replace(target)
            restored.append(str(target))
        elif target.is_file():
            target.unlink()
            removed.append(str(target))
    return restored, removed, conflicts


def _rollback_backup(backup: Path) -> dict[str, Any]:
    manifest = json.loads(
        (backup / "backup_manifest.json").read_text(encoding="utf-8")
    )
    restored, removed, config_conflicts, definition_conflict = _rollback_hooks_config(
        backup, manifest
    )
    file_restored, file_removed, conflicts = _rollback_files(
        backup,
        manifest,
        preserve_installed_hook=definition_conflict,
    )
    return {
        "rolled_back_from": str(backup),
        "restored": [*restored, *file_restored],
        "removed": [*removed, *file_removed],
        "preserved_conflicts": [*config_conflicts, *conflicts],
    }


def rollback() -> dict[str, Any]:
    root = codex_home() / "hook-backups"
    candidates = sorted(
        (path for path in root.iterdir() if (path / "backup_manifest.json").is_file()),
        reverse=True,
    ) if root.is_dir() else []
    if not candidates:
        raise FileNotFoundError("no DTE hook backup is available")
    # Idempotent installs create no backup. A content revision is a real new
    # transaction, so rollback safely undoes the newest installed definition.
    return _rollback_backup(candidates[0])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Install or verify deterministic DTE hooks")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scope", choices=["user", "managed-template"])
    group.add_argument("--verify", action="store_true")
    group.add_argument("--rollback", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.scope == "user":
            result = install_user()
        elif args.scope == "managed-template":
            result = install_managed_template()
        elif args.verify:
            result = verify_user()
        else:
            result = rollback()
    except Exception as exc:
        print(json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    print(json.dumps({"success": True, **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

