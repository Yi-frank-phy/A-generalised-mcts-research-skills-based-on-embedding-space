from pathlib import Path
from typing import get_args

import dte_backend.episode_adapter as episode_adapter
from dte_backend.episode_models import EpisodeRequest, EpisodeRole
from tests.helpers import completed_node

from dte_backend.app_driver import create_app_run, next_app_episode
from dte_backend.models import BudgetSpec, DTERunSpec


ROOT = Path(__file__).resolve().parents[1]


def _spec() -> DTERunSpec:
    return DTERunSpec(
        problem="p",
        goal="g",
        budget=BudgetSpec(
            max_iterations=2,
            allocation_mass_per_iteration=1,
            max_children_per_iteration=1,
            max_relation_enrichment_pairs=0,
        ),
        embedding_provider="hash",
        embedding_dimension=8,
    )


def test_new_run_never_schedules_generic_judge(tmp_path):
    run_dir = tmp_path / "run"
    create_app_run(
        run_dir,
        _spec(),
        [completed_node(node_id="seed", claim="seed claim")],
        run_id="run-no-judge",
    )

    outcome = next_app_episode(run_dir)

    assert outcome.request is None or outcome.request.role != "judge"


def test_production_episode_contract_has_no_generic_judge_role():
    assert "judge" not in get_args(EpisodeRole)
    assert "judge_payload" not in EpisodeRequest.model_fields
    assert not hasattr(episode_adapter, "build_judge_episode_request")


def test_production_bundle_has_no_generic_judge_files_or_cli():
    for relative in (
        "src/dte_backend/judge.py",
        "scripts/codex_judge_adapter.py",
        "prompts/judge.md",
        "prompts/judge_oracle.md",
    ):
        assert not (ROOT / relative).exists(), relative

    strict_runner = (ROOT / "src/dte_backend/strict_runner.py").read_text(encoding="utf-8")
    cli = (ROOT / "src/dte_backend/__main__.py").read_text(encoding="utf-8")
    skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")

    assert "judge_command" not in strict_runner
    assert "--judge-command" not in cli
    assert "Judge every granted node exactly once" not in skill
