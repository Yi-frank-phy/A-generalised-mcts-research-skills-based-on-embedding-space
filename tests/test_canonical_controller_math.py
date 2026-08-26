from tests.helpers import completed_node
import math

import pytest

import dte_backend.app_driver as app_driver
from dte_backend.app_driver import create_app_run, next_app_episode, submit_app_episode_result
from dte_backend.embedding import HashEmbeddingProvider
from dte_backend.entropy import temperature_for_target_entropy
from dte_backend.episode_models import EpisodeResult, RuntimeDiagnostics, compute_output_hash
from dte_backend.math_engine import allocate_frontier
from dte_backend.models import BudgetSpec, DTERunSpec, SearchNode




