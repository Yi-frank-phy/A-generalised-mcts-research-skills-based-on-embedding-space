"""Canonical context envelopes for cache-friendly DTE nodes.

The old context-engineering style tried to minimize prompt length by aggressively
compiling local context. That can accidentally make cache keys unstable: tiny
formatting changes, reordered bullet lists, temporary logs, and parent ids cause
otherwise equivalent nodes to miss embedding/Judge caches.

This module separates:

- semantic envelope: stable enough for embedding geometry;
- evaluation envelope: slightly richer, used for Judge caching.

The goal is not to make stale caches. Real semantic changes in claim/evidence/
risk still change the key. Unstable transcript/log/context noise should not.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from .models import SearchNode


_WS_RE = re.compile(r"\s+")
_VOLATILE_LINE_RE = re.compile(
    r"^(debug|trace|log|stdout|stderr|tool|timestamp|time|uuid|id|run|commit|sha|cwd|path)\s*[:=]",
    re.IGNORECASE,
)


def normalize_text(text: str, max_chars: int = 2000) -> str:
    """Normalize text for cache keys and embedding input.

    This intentionally removes obvious run-local noise while preserving formulas,
    assumptions, evidence, and risk statements.
    """

    lines: list[str] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _VOLATILE_LINE_RE.match(line):
            continue
        line = _WS_RE.sub(" ", line)
        lines.append(line)
    normalized = "\n".join(lines).strip().casefold()
    if len(normalized) > max_chars:
        normalized = normalized[:max_chars]
    return normalized


def normalize_items(items: Iterable[str], max_items: int = 12, max_chars_each: int = 500) -> list[str]:
    """Normalize, deduplicate, and sort list-like context fields."""

    seen: set[str] = set()
    values: list[str] = []
    for item in items:
        normalized = normalize_text(str(item), max_chars=max_chars_each)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        values.append(normalized)
    return sorted(values)[:max_items]


def semantic_embedding_text(node: SearchNode) -> str:
    """Stable node summary used as embedding input.

    Parent ids, confidence, status, score, and UCB are intentionally excluded.
    Evidence and risks are included because they change the semantic position.
    """

    parts = [
        "claim:\n" + normalize_text(node.claim, max_chars=1200),
        "assumptions:\n" + "\n".join(normalize_items(node.assumptions)),
        "evidence:\n" + "\n".join(normalize_items(node.evidence)),
        "risks:\n" + "\n".join(normalize_items(node.risks)),
    ]
    # Rationale is useful but often contains volatile local reasoning. Keep a
    # short normalized version only.
    rationale = normalize_text(node.rationale, max_chars=800)
    if rationale:
        parts.append("rationale_hint:\n" + rationale)
    return "\n\n".join(part for part in parts if part.strip())


def semantic_embedding_payload(node: SearchNode) -> dict[str, object]:
    """Payload for embedding cache key."""

    return {
        "node_type": node.node_type,
        "claim": normalize_text(node.claim, max_chars=1200),
        "assumptions": normalize_items(node.assumptions),
        "evidence": normalize_items(node.evidence),
        "risks": normalize_items(node.risks),
        "rationale_hint": normalize_text(node.rationale, max_chars=800),
    }


def evaluation_payload(node: SearchNode) -> dict[str, object]:
    """Payload for Judge cache key.

    This includes confidence because it is part of the stated candidate quality,
    but still excludes parent ids and controller metrics.
    """

    payload = semantic_embedding_payload(node)
    payload["confidence"] = round(float(node.confidence), 4)
    return payload
