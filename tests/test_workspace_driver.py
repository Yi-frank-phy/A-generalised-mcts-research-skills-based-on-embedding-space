from __future__ import annotations

import importlib.util
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "workspace_driver.py"
SPEC = importlib.util.spec_from_file_location("workspace_driver", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
workspace_driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workspace_driver)


def test_parse_env_file_accepts_only_supported_provider_keys(tmp_path: Path) -> None:
    path = tmp_path / "secrets.env"
    path.write_text(
        "GEMINI_API_KEY=test-gemini\n"
        "GOOGLE_API_KEY=test-google\n"
        "UNRELATED_SECRET=must-not-load\n",
        encoding="utf-8",
    )
    assert workspace_driver._parse_env_file(path) == {
        "GEMINI_API_KEY": "test-gemini",
        "GOOGLE_API_KEY": "test-google",
    }


def test_load_secrets_does_not_override_existing_environment(monkeypatch, tmp_path: Path) -> None:
    secret_file = tmp_path / "secrets.env"
    secret_file.write_text("GEMINI_API_KEY=file-value\n", encoding="utf-8")
    monkeypatch.setenv("DTE_SECRETS_FILE", str(secret_file))
    monkeypatch.setenv("GEMINI_API_KEY", "environment-value")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    source = workspace_driver.load_secrets()

    assert source == "environment"
    assert os.environ["GEMINI_API_KEY"] == "environment-value"


def test_workspace_attestation_never_claims_host_or_isolation_enforcement() -> None:
    attestation = workspace_driver._attestation(receipt_chain_verified=True)
    assert attestation["backend_receipt_chain_verified"] is True
    assert attestation["host_hook_enforcement"] is False
    assert attestation["wrapper_use_host_enforced"] is False
    assert attestation["context_isolation_verified"] is False
    assert attestation["reasoning_effort_attested"] is False


def test_repository_does_not_contain_committed_gemini_key_pattern() -> None:
    forbidden_prefix = "AI" + "za"
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.name == ".env" or path.suffix in {".pyc", ".zip"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        assert forbidden_prefix not in text, f"possible Gemini API key committed in {path}"
