from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_RUNTIME_DOCS = [
    ROOT / "SKILL.md",
    ROOT / "agents" / "openai.yaml",
    ROOT / "README.md",
    ROOT / "docs" / "CODEX_APP_WORKFLOW.md",
    ROOT / "SPEC.md",
    ROOT / "ARCHITECTURE.md",
    ROOT / "AGENTS.md",
]
ACTIVE_RUNTIME_CODE = list((ROOT / "src" / "dte_backend").glob("*.py"))


def _joined(paths: list[Path]) -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)


def test_production_docs_expose_only_strict_real_entrypoint():
    docs = _joined(ACTIVE_RUNTIME_DOCS)
    metadata = (ROOT / "agents" / "openai.yaml").read_text(encoding="utf-8")
    skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")

    assert "app-orchestrated-real" not in docs
    assert "python -m dte_backend strict-run --mode real" in skill
    assert "strict-run --mode real" in metadata
    assert "main agent is the orchestrator" not in skill.casefold()


def test_active_runtime_has_no_main_agent_synthesis_path():
    active = _joined(ACTIVE_RUNTIME_DOCS + ACTIVE_RUNTIME_CODE)

    assert "main_agent_requested_synthesis" not in active
    assert 'requested_by="main_agent"' not in active
    assert '"requested_by": "main_agent"' not in active
    assert "requested_by: main_agent" not in active


def test_runtime_instructions_state_observation_is_not_authority():
    skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")
    workflow = (ROOT / "docs" / "CODEX_APP_WORKFLOW.md").read_text(encoding="utf-8")

    assert "observation != authority" in skill
    assert "observation != authority" in workflow
