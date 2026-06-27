"""Shared DTE boundary guard checks."""

from __future__ import annotations

from .models import DTERunSpec


def enforce_run_spec_guard(spec: DTERunSpec) -> None:
    """Fail before a non-compliant run spec can enter a DTE workflow."""

    if spec.mode != "mandatory_frontier":
        raise SystemExit("DTE guard failed: mode must be mandatory_frontier")
    if spec.embedding_provider == "gemini-embedding-2" and spec.embedding_dimension != 3072:
        raise SystemExit("DTE guard failed: Gemini geometry must use embedding_dimension=3072")
