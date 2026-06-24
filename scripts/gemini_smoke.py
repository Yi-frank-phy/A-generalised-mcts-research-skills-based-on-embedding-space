"""Manual Gemini Embedding 2 smoke test.

This script is intentionally not part of normal CI. It only runs when a Gemini
API key is present and verifies that the provider returns the requested max
geometry dimension.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dte_backend.embedding import GeminiEmbedding2Provider  # noqa: E402


def main() -> None:
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        print("Skipping Gemini smoke: set GEMINI_API_KEY or GOOGLE_API_KEY to run it.")
        return

    provider = GeminiEmbedding2Provider(dim=3072)
    vectors = provider.embed_texts(["DTE geometry smoke test node summary."])
    if len(vectors) != 1:
        raise SystemExit("expected one vector")
    if len(vectors[0]) != 3072:
        raise SystemExit(f"expected 3072 dimensions, got {len(vectors[0])}")
    print("Gemini Embedding 2 smoke ok: 3072 dimensions")


if __name__ == "__main__":
    main()
