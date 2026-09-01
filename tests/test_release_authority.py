from pathlib import Path
import hashlib

EXPECTED = {
    "docs/PHYSICS.md": "d8e2028a9b8bf2d93490ca1e2b13be19ccabfd6c4de2adfded10644077d32d3d",
    "docs/DESIGN.md": "ee69051d16852731d7dea3a79030cdc3f88042c2c4b9dafe5bfea94f0df07603",
}

def test_formal_release_authority_hashes() -> None:
    root = Path(__file__).resolve().parents[1]
    observed = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in EXPECTED}
    assert observed == EXPECTED
