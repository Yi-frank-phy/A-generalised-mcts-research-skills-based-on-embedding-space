from pathlib import Path
import hashlib

EXPECTED = {
    "docs/PHYSICS.md": "f7c8879ed7de1f03c987dbc412469b05a752aa7df4a05cc047529cf2bd2896e7",
    "docs/DESIGN.md": "11f6d7016bb4af486e39ef75fa5936c794e98b0bb870d5ea512e043d685f1c69",
}

def test_formal_release_authority_hashes() -> None:
    root = Path(__file__).resolve().parents[1]
    observed = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in EXPECTED}
    assert observed == EXPECTED
