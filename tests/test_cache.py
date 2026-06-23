from dte_backend.cache import DTECache, stable_node_hash
from dte_backend.judge import batch_judge
from dte_backend.models import SearchNode
from dte_backend.novelty import ensure_embeddings


def test_stable_node_hash_ignores_metrics():
    node = SearchNode(node_id="n", claim="same", score=0.1, uncertainty=0.2)
    before = stable_node_hash(node)
    node.score = 0.9
    node.uncertainty = 0.8
    node.ucb_score = 1.7
    assert stable_node_hash(node) == before


def test_judge_cache_hits_on_second_eval():
    cache = DTECache()
    node = SearchNode(node_id="n", claim="route", confidence=0.5)
    batch_judge([node], cache=cache)
    batch_judge([node], cache=cache)
    assert cache.stats.judge_hits == 1
    assert cache.stats.judge_misses == 1


def test_embedding_cache_hits_for_equivalent_content():
    cache = DTECache()
    first = SearchNode(node_id="a", claim="same claim")
    second = SearchNode(node_id="b", claim="same claim")
    ensure_embeddings([first], cache=cache)
    ensure_embeddings([second], cache=cache)
    assert cache.stats.embedding_hits == 1
    assert cache.stats.embedding_misses == 1
    assert first.local_embedding == second.local_embedding
