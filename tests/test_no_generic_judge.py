from tests.helpers import completed_node

from dte_backend.app_driver import create_app_run, next_app_episode
from dte_backend.models import BudgetSpec, DTERunSpec


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
