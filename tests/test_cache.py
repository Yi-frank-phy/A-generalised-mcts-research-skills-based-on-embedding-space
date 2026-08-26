from dte_backend.cache import DTECache, EmbeddingCacheNamespace, embedding_cache_key
from dte_backend.models import SearchNode






def test_embedding_cache_key_changes_with_provider_namespace():
    node = SearchNode(node_id="n", claim="same")
    first = EmbeddingCacheNamespace("hash", "hash-v1", 64, "embedding-v1")
    second = EmbeddingCacheNamespace("gemini", "snapshot-1", 64, "embedding-v1")
    assert embedding_cache_key(node, namespace=first) != embedding_cache_key(node, namespace=second)


