"""DTE Codex skill backend prototype."""

from . import app_driver as _app_driver
from .executor_adapter import ExpansionRequest, SubprocessExecutorAdapter
from .app_driver import (
    app_run_status,
    cancel_app_episode,
    create_app_run,
    fail_app_episode,
    next_app_episode,
    retry_app_episode,
)
from .app_submit_guard import build_fail_closed_submit
from .episode_adapter import (
    AgentEpisodeAdapter,
    CommandAgentEpisodeAdapter,
    NativeStubEpisodeAdapter,
    run_and_commit_episode,
)
from .episode_commit import EpisodeGraph, commit_episode_result
from .episode_models import EpisodeRequest, EpisodeResult
from .models import AllocationResult, BudgetSpec, DTERunSpec, SearchNode
from .runner import RunResult, run_frontier_search

# Install the fail-closed public submission boundary on the already-loaded
# app_driver module.  This also covers `python -m dte_backend`, whose __main__
# imports the function from app_driver after package initialization.
submit_app_episode_result = build_fail_closed_submit(
    original_submit=_app_driver.submit_app_episode_result,
    load_state=_app_driver.load_app_run,
    event_log_factory=_app_driver._event_log,
    submit_outcome_type=_app_driver.SubmitEpisodeOutcome,
)
_app_driver.submit_app_episode_result = submit_app_episode_result

__all__ = [
    "AllocationResult",
    "AgentEpisodeAdapter",
    "app_run_status",
    "BudgetSpec",
    "DTERunSpec",
    "CommandAgentEpisodeAdapter",
    "create_app_run",
    "cancel_app_episode",
    "EpisodeGraph",
    "EpisodeRequest",
    "EpisodeResult",
    "ExpansionRequest",
    "RunResult",
    "SearchNode",
    "NativeStubEpisodeAdapter",
    "next_app_episode",
    "SubprocessExecutorAdapter",
    "run_frontier_search",
    "run_and_commit_episode",
    "retry_app_episode",
    "fail_app_episode",
    "submit_app_episode_result",
    "commit_episode_result",
]
