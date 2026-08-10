"""Content-pinned entrypoint for hook-enforced DTE driver commands."""

from __future__ import annotations

import sys
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
SRC = SKILL_ROOT / "src"
sys.path.insert(0, str(SRC))

from dte_backend.__main__ import main  # noqa: E402


if __name__ == "__main__":
    main()
