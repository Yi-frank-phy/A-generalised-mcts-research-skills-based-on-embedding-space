"""Validation helpers for DTE machine-facing artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def load_json_model(path: str | Path, model_type: type[T]) -> T:
    """Load a JSON file and validate it against a Pydantic model."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return model_type.model_validate(data)


def load_json_list(path: str | Path, model_type: type[T]) -> list[T]:
    """Load a JSON list and validate each item."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return [model_type.model_validate(item) for item in data]
