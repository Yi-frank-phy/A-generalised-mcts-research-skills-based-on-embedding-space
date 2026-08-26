from tests.helpers import completed_node
import dte_backend
import dte_backend.app_driver as app_driver

from dte_backend.embedding import HashEmbeddingProvider
from dte_backend.episode_models import EpisodeResult, RuntimeDiagnostics, compute_output_hash
from dte_backend.models import BudgetSpec, DTERunSpec, SearchNode
from dte_backend.telemetry import EpisodeEventLog




def _state_snapshot(run_dir):
    return app_driver.app_run_status(run_dir).model_dump(mode="json")


def test_public_and_app_driver_submission_entrypoints_are_the_same_guard():
    assert app_driver.submit_app_episode_result is dte_backend.submit_app_episode_result




