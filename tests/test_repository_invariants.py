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
PUBLIC_SKILL_SURFACES = [
    ROOT / "SKILL.md",
    ROOT / "agents" / "openai.yaml",
    ROOT / "hooks" / "dte_prompt_guard.py",
]


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


def test_public_skill_surfaces_expose_only_canonical_name():
    public_surfaces = _joined(PUBLIC_SKILL_SURFACES)
    old_alias = "dte-" + "extreme-research"

    assert old_alias not in public_surfaces
    assert "name: evolving-frontier-research" in public_surfaces
    assert "$evolving-frontier-research" in public_surfaces
    assert "`evolving-frontier-research` skill" in public_surfaces


def test_active_runtime_routes_main_agent_authority_through_operator_policy():
    active = _joined(ACTIVE_RUNTIME_DOCS + ACTIVE_RUNTIME_CODE)

    assert "main_agent_requested_synthesis" in active
    assert "main_agent_may_request_synthesis" in active
    assert "main-agent synthesis request is disabled by operator_policy" in active


def test_runtime_instructions_state_observation_is_not_authority():
    skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")
    workflow = (ROOT / "docs" / "CODEX_APP_WORKFLOW.md").read_text(encoding="utf-8")

    assert "observation != authority" in skill
    assert "observation != authority" in workflow
    assert "delegation + policy + validated command = authority" in skill
    assert "delegation + policy + validated command = authority" in workflow
