import hashlib
import json
import os
import re
import runpy
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "scripts" / "install_dte_hooks.py"


def run_installer(codex_home: Path, *args: str):
    env = dict(os.environ)
    env["DTE_CODEX_HOME"] = str(codex_home)
    return subprocess.run(
        [sys.executable, str(INSTALLER), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
    )


def test_user_install_merges_unrelated_hooks_verifies_and_rolls_back(tmp_path):
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    original = {
        "description": "keep me",
        "hooks": {
            "UserPromptSubmit": [
                {
                    "hooks": [
                        {"type": "command", "command": "python unrelated.py"},
                        {"type": "command", "command": "python dte_prompt_guard.py"},
                    ]
                }
            ]
        },
    }
    (codex_home / "hooks.json").write_text(json.dumps(original), encoding="utf-8")
    (codex_home / "hooks").mkdir()
    legacy_hook = codex_home / "hooks" / "dte_prompt_guard.py"
    legacy_hook.write_text("# legacy reminder\n", encoding="utf-8")

    installed = run_installer(codex_home, "--scope", "user")
    assert installed.returncode == 0, installed.stdout + installed.stderr
    config = json.loads((codex_home / "hooks.json").read_text(encoding="utf-8"))
    assert config["description"] == "keep me"
    commands = [
        handler["command"]
        for groups in config["hooks"].values()
        for group in groups
        for handler in group["hooks"]
    ]
    assert "python unrelated.py" in commands
    assert not any("dte_prompt_guard.py" in command for command in commands)
    assert sum("dte_enforcement_hook.py" in command for command in commands) == 5
    content_sha256 = hashlib.sha256(
        (ROOT / "hooks" / "dte_enforcement_hook.py").read_bytes()
    ).hexdigest()
    assert all(
        f"--dte-hook-content-sha256={content_sha256}" in command
        for command in commands
        if "dte_enforcement_hook.py" in command
    )
    assert all(
        f"--dte-skill-root={ROOT.resolve()}" in command
        for command in commands
        if "dte_enforcement_hook.py" in command
    )
    assert config["hooks"]["PreToolUse"][-1]["matcher"] == "*"
    assert config["hooks"]["PostToolUse"][-1]["matcher"] == "*"
    assert (codex_home / "hooks" / "dte_enforcement_hook.py").is_file()
    assert not legacy_hook.exists()

    verified = run_installer(codex_home, "--verify")
    assert verified.returncode == 0, verified.stdout + verified.stderr
    verified_payload = json.loads(verified.stdout)
    assert verified_payload["verified"] is True
    assert verified_payload["static_configuration_verified"] is True
    assert verified_payload["runtime_delivery_status"] == "requires_post_restart_probe"
    assert "plain_python_backend_origin" in verified_payload

    prompt_command = config["hooks"]["UserPromptSubmit"][-1]["hooks"][0]["command"]
    probe_env = dict(os.environ)
    probe_env.pop("DTE_SKILL_ROOT", None)
    probe_env["DTE_HOOK_STATE_ROOT"] = str(tmp_path / "probe-state")
    probe_payload = {
        "session_id": "installed-command-probe",
        "cwd": str(tmp_path),
        "hook_event_name": "UserPromptSubmit",
        "turn_id": "probe-turn",
        "prompt": "/evolving-frontier-research --hook-probe InstallProbe_1234",
    }
    if os.name == "nt":
        invocation = [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            prompt_command,
        ]
    else:
        invocation = ["/bin/sh", "-c", prompt_command]
    probed = subprocess.run(
        invocation,
        input=json.dumps(probe_payload),
        capture_output=True,
        text=True,
        env=probe_env,
        cwd=tmp_path,
    )
    assert probed.returncode == 0, probed.stdout + probed.stderr
    probe_context = json.loads(probed.stdout)["hookSpecificOutput"]["additionalContext"]
    assert "DTE_HOOK_PROBE_ACK:InstallProbe_1234" in probe_context
    _, probe_evidence = probe_context.split(" ", maxsplit=1)
    assert Path(json.loads(probe_evidence)["skill_root"]).resolve() == ROOT.resolve()

    repeated = run_installer(codex_home, "--scope", "user")
    repeated_payload = json.loads(repeated.stdout)
    assert repeated.returncode == 0, repeated.stdout + repeated.stderr
    assert repeated_payload["changed"] is False
    assert repeated_payload["backup_dir"] is None

    rolled_back = run_installer(codex_home, "--rollback")
    assert rolled_back.returncode == 0, rolled_back.stdout + rolled_back.stderr
    restored = json.loads((codex_home / "hooks.json").read_text(encoding="utf-8"))
    assert restored == original
    assert not (codex_home / "hooks" / "dte_enforcement_hook.py").exists()
    assert legacy_hook.read_text(encoding="utf-8") == "# legacy reminder\n"


def test_verify_rejects_partial_or_wrong_managed_definitions(tmp_path):
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    installed = run_installer(codex_home, "--scope", "user")
    assert installed.returncode == 0, installed.stdout + installed.stderr

    config_path = codex_home / "hooks.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["hooks"]["PreToolUse"][-1]["matcher"] = "exec_command"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    wrong_matcher = run_installer(codex_home, "--verify")
    assert wrong_matcher.returncode == 1
    assert "complete DTE enforcement definition" in wrong_matcher.stdout

    config["hooks"]["PreToolUse"][-1]["matcher"] = "*"
    handler = config["hooks"]["PostToolUse"][-1]["hooks"][0]
    versioned_command = handler["command"]
    handler["command"] = re.sub(
        r"""\s+['"]?--dte-hook-content-sha256=[0-9a-f]{64}['"]?""",
        "",
        versioned_command,
        count=1,
    )
    config_path.write_text(json.dumps(config), encoding="utf-8")
    missing_content_hash = run_installer(codex_home, "--verify")
    assert missing_content_hash.returncode == 1
    assert "complete DTE enforcement definition" in missing_content_hash.stdout

    handler["command"] = versioned_command
    config["hooks"]["PostToolUse"].append(
        json.loads(json.dumps(config["hooks"]["PostToolUse"][-1]))
    )
    config_path.write_text(json.dumps(config), encoding="utf-8")
    duplicate_handler = run_installer(codex_home, "--verify")
    assert duplicate_handler.returncode == 1
    assert "exactly one DTE enforcement handler reference" in duplicate_handler.stdout


def test_content_revision_changes_the_trusted_handler_definition(tmp_path):
    namespace = runpy.run_path(str(INSTALLER))
    target = tmp_path / "Codex Home With Spaces" / "hooks" / "dte_enforcement_hook.py"
    pinned_root = tmp_path / "Skill Root With Spaces"
    first_digest = "1" * 64
    second_digest = "2" * 64

    first = namespace["_merge_hooks"]({}, target, first_digest, pinned_root)
    second = namespace["_merge_hooks"]({}, target, second_digest, pinned_root)
    first_command = first["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    second_command = second["hooks"]["PreToolUse"][0]["hooks"][0]["command"]

    assert first_command != second_command
    assert str(target) in first_command
    assert f"--dte-hook-content-sha256={first_digest}" in first_command
    assert f"--dte-hook-content-sha256={second_digest}" in second_command
    assert f"--dte-skill-root={pinned_root.resolve()}" in first_command
    if os.name == "nt":
        spaced_python = Path(r"C:\Program Files\Python 3.14\python.exe")
        powershell_command = namespace["_command"](
            target,
            first_digest,
            pinned_root,
            spaced_python,
        )
        assert powershell_command.startswith("& ")
        assert f"'{spaced_python}'" in powershell_command


def test_rollback_preserves_unrelated_post_install_changes(tmp_path):
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    original = {
        "description": "before install",
        "hooks": {
            "UserPromptSubmit": [
                {
                    "hooks": [
                        {"type": "command", "command": "python unrelated-before.py"},
                        {"type": "command", "command": "python dte_prompt_guard.py"},
                    ]
                }
            ]
        },
    }
    (codex_home / "hooks.json").write_text(json.dumps(original), encoding="utf-8")
    hooks_dir = codex_home / "hooks"
    hooks_dir.mkdir()
    legacy_hook = hooks_dir / "dte_prompt_guard.py"
    legacy_hook.write_text("# legacy\n", encoding="utf-8")

    installed = run_installer(codex_home, "--scope", "user")
    assert installed.returncode == 0, installed.stdout + installed.stderr
    current = json.loads((codex_home / "hooks.json").read_text(encoding="utf-8"))
    current["description"] = "changed after install"
    current["post_install_setting"] = {"preserve": True}
    current["hooks"]["Notification"] = [{"matcher": "*", "hooks": []}]
    current["hooks"]["PreToolUse"].insert(
        0,
        {
            "matcher": "Bash",
            "hooks": [
                {"type": "command", "command": "python unrelated-after.py"}
            ],
        },
    )
    (codex_home / "hooks.json").write_text(json.dumps(current), encoding="utf-8")

    rolled_back = run_installer(codex_home, "--rollback")
    assert rolled_back.returncode == 0, rolled_back.stdout + rolled_back.stderr
    assert json.loads(rolled_back.stdout)["preserved_conflicts"] == []
    restored = json.loads((codex_home / "hooks.json").read_text(encoding="utf-8"))
    assert restored["description"] == "changed after install"
    assert restored["post_install_setting"] == {"preserve": True}
    assert restored["hooks"]["Notification"] == [{"matcher": "*", "hooks": []}]
    commands = [
        handler["command"]
        for groups in restored["hooks"].values()
        for group in groups
        for handler in group["hooks"]
    ]
    assert "python unrelated-before.py" in commands
    assert "python unrelated-after.py" in commands
    assert "python dte_prompt_guard.py" in commands
    assert not any("dte_enforcement_hook.py" in command for command in commands)
    assert not (hooks_dir / "dte_enforcement_hook.py").exists()
    assert legacy_hook.read_text(encoding="utf-8") == "# legacy\n"


def test_managed_template_pins_hooks_without_disabling_unrelated_hooks():
    requirements = runpy.run_path(str(INSTALLER))["_managed_requirements"]()
    assert "[features]\nhooks = true" in requirements
    assert "C:\\ProgramData\\Codex\\DTE\\hooks" in requirements
    assert "allow_managed_hooks_only" not in requirements
    for event in ("UserPromptSubmit", "PreToolUse", "PostToolUse", "Stop", "SessionStart"):
        assert f"[[hooks.{event}]]" in requirements
    assert requirements.count('matcher = "*"') == 2
    assert "exec_command" not in requirements
    content_sha256 = hashlib.sha256(
        (ROOT / "hooks" / "dte_enforcement_hook.py").read_bytes()
    ).hexdigest()
    assert requirements.count(
        f"--dte-hook-content-sha256={content_sha256}"
    ) == 10
    assert requirements.count("--dte-skill-root=/opt/codex/dte") == 5
    assert requirements.count(
        "--dte-skill-root=C:\\ProgramData\\Codex\\DTE"
    ) == 5
    managed = ROOT / "deploy" / "managed-template"
    assert (managed / "requirements.toml").read_text(encoding="utf-8") == requirements
    assert (
        managed / "hooks" / "dte_enforcement_hook.py"
    ).read_bytes() == (ROOT / "hooks" / "dte_enforcement_hook.py").read_bytes()


def test_v1_rollback_preserves_installed_hook_without_post_install_hash(tmp_path):
    codex_home = tmp_path / "codex-home"
    hooks_dir = codex_home / "hooks"
    backup = codex_home / "hook-backups" / "legacy-v1"
    hooks_dir.mkdir(parents=True)
    backup.mkdir(parents=True)

    current_hook = hooks_dir / "dte_enforcement_hook.py"
    current_hook.write_text("# possibly edited after installation\n", encoding="utf-8")
    original_config = {"hooks": {}}
    (codex_home / "hooks.json").write_text(
        json.dumps(original_config),
        encoding="utf-8",
    )
    (backup / "hooks.json").write_text(
        json.dumps(original_config),
        encoding="utf-8",
    )
    (backup / "backup_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "dte-hook-backup.v1",
                "files": [
                    {
                        "path": str(codex_home / "hooks.json"),
                        "existed": True,
                        "backup_name": "hooks.json",
                    },
                    {
                        "path": str(current_hook),
                        "existed": False,
                        "backup_name": "dte_enforcement_hook.py",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    rolled_back = run_installer(codex_home, "--rollback")
    assert rolled_back.returncode == 0, rolled_back.stdout + rolled_back.stderr
    payload = json.loads(rolled_back.stdout)
    assert current_hook.is_file()
    assert current_hook.read_text(encoding="utf-8") == (
        "# possibly edited after installation\n"
    )
    assert any(
        "no recorded post-install hash" in conflict
        for conflict in payload["preserved_conflicts"]
    )

