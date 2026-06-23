"""Backward-compatible executor adapter API.

The canonical implementation lives in :mod:`dte_backend.adapter`. This module
keeps the older class/protocol names working for callers that still import
``dte_backend.executor_adapter``.
"""

from __future__ import annotations

from typing import Protocol

from .adapter import run_subprocess_executor, validate_adapter_output
from .models import ExpansionRequest, SearchNode


class ExecutorAdapter(Protocol):
    """Adapter protocol used by the DTE expansion role."""

    def expand(self, request: ExpansionRequest) -> list[SearchNode]:
        """Return child SearchNodes for the given parent and budget."""


def validate_executor_children(parent: SearchNode, count: int, children: list[SearchNode]) -> list[SearchNode]:
    """Validate already-parsed children through the canonical adapter rules."""

    try:
        return validate_adapter_output(parent, count, {"nodes": [child.model_dump() for child in children]})
    except ValueError as exc:
        message = str(exc)
        if "DTE metric" in message:
            raise ValueError("Executor adapters cannot pre-fill Judge/Evolution metrics") from exc
        if "expansion_budget" in message:
            raise ValueError("Executor adapters cannot pre-fill expansion budgets") from exc
        raise


class SubprocessExecutorAdapter:
    """Class wrapper around the canonical subprocess adapter function."""

    def __init__(self, command: list[str], timeout_seconds: float = 120.0) -> None:
        self.command = list(command)
        self.timeout_seconds = timeout_seconds

    def expand(self, request: ExpansionRequest) -> list[SearchNode]:
        """Run the subprocess adapter and validate returned children."""

        try:
            return run_subprocess_executor(self.command, request, timeout=self.timeout_seconds)
        except ValueError as exc:
            if "JSON list or {'nodes':" in str(exc):
                raise ValueError("Executor stdout must be a JSON list or an object with a nodes list") from exc
            raise
