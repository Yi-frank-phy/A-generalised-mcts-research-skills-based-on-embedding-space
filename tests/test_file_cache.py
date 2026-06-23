from dte_backend.file_cache import FileDTECache
from dte_backend.models import SearchNode


def test_file_cache_persists_embedding(tmp_path):
    path = tmp_path / "dte_cache.json"
    node = SearchNode(node_id="n", claim="claim")
    cache = FileDTECache(path)
    cache.set_embedding(node, [0.1, 0.2])
    cache2 = FileDTECache(path)
    assert cache2.get_embedding(node) == [0.1, 0.2]


def test_file_cache_persists_score(tmp_path):
    path = tmp_path / "dte_cache.json"
    node = SearchNode(node_id="n", claim="claim")
    cache = FileDTECache(path)
    cache.set_judge(node, 0.7, "reason")
    cache2 = FileDTECache(path)
    entry = cache2.get_judge(node)
    assert entry is not None
    assert entry.score == 0.7
