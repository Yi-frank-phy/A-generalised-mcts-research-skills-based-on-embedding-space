from dte_backend.cache import embedding_cache_key, judge_cache_key
from dte_backend.context_envelope import normalize_items, normalize_text, semantic_embedding_text
from dte_backend.models import SearchNode


def test_normalize_text_drops_volatile_lines():
    text = "Claim line\nstdout: temporary log\nTimestamp: 2026-01-01\nReal evidence"
    normalized = normalize_text(text)
    assert "stdout" not in normalized
    assert "timestamp" not in normalized
    assert "real evidence" in normalized


def test_normalize_items_deduplicates_and_sorts():
    assert normalize_items([" B ", "a", "b", "A"]) == ["a", "b"]


def test_embedding_key_ignores_parent_confidence_and_order():
    first = SearchNode(
        node_id="a",
        claim="Same Claim",
        assumptions=["b", "a"],
        evidence=["e1", "e2"],
        risks=["r"],
        parent_ids=["p1"],
        confidence=0.2,
    )
    second = SearchNode(
        node_id="b",
        claim=" same   claim ",
        assumptions=["a", "b"],
        evidence=["e2", "e1"],
        risks=["r"],
        parent_ids=["p2"],
        confidence=0.9,
    )
    assert embedding_cache_key(first) == embedding_cache_key(second)


def test_judge_key_tracks_confidence_but_not_parent_id():
    first = SearchNode(node_id="a", claim="same", parent_ids=["p1"], confidence=0.2)
    second = SearchNode(node_id="b", claim="same", parent_ids=["p2"], confidence=0.2)
    third = SearchNode(node_id="c", claim="same", parent_ids=["p2"], confidence=0.9)
    assert judge_cache_key(first) == judge_cache_key(second)
    assert judge_cache_key(first) != judge_cache_key(third)


def test_semantic_embedding_text_is_stable_envelope():
    node = SearchNode(claim="C", node_id="n", evidence=["E"], risks=["R"])
    text = semantic_embedding_text(node)
    assert "claim:" in text
    assert "evidence:" in text
    assert "risks:" in text
