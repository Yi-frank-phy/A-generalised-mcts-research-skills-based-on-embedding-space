from pathlib import Path
import hashlib

EXPECTED = {
    "docs/PHYSICS.md": "d972b8f7242451a3114ad030b0396c277855ee5e65d5df020c8a1967b5642a42",
    "docs/DESIGN.md": "346ddd03584753ce2b0cbd647089651ea0a89fe2f69b8150e9f7aa8bf10a2b7f",
}

def test_formal_release_authority_hashes() -> None:
    root = Path(__file__).resolve().parents[1]
    observed = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in EXPECTED}
    assert observed == EXPECTED
