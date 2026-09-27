from app.schemas.search import SearchResult
from app.services.search_cache import SearchCache


class FakeRedis:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self.data.get(key)

    def set(
        self,
        key: str,
        value: str,
        nx: bool = False,
    ) -> bool:
        if nx and key in self.data:
            return False

        self.data[key] = value
        return True

    def setex(
        self,
        key: str,
        _ttl: int,
        value: str,
    ) -> bool:
        self.data[key] = value
        return True

    def incr(self, key: str) -> int:
        value = int(self.data.get(key, "0")) + 1
        self.data[key] = str(value)
        return value

    def delete(self, key: str) -> int:
        return int(
            self.data.pop(key, None) is not None,
        )


def test_search_cache_key_is_scoped_to_user() -> None:
    cache = SearchCache(FakeRedis())

    user_a_key = cache.build_key(
        user_id="user-a",
        query="什么是 JWT？",
        top_k=5,
        mode="vector",
        min_score=0.0,
        min_rerank_score=0.0,
    )

    user_b_key = cache.build_key(
        user_id="user-b",
        query="什么是 JWT？",
        top_k=5,
        mode="vector",
        min_score=0.0,
        min_rerank_score=0.0,
    )

    assert user_a_key != user_b_key


def _result(
    chunk_id: str,
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        filename="test.txt",
        chunk_index=0,
        content=f"内容 {chunk_id}",
        score=0.9,
    )


def test_search_cache_returns_stored_results() -> None:
    cache = SearchCache(FakeRedis())

    cache_key = cache.build_key(
        user_id="user-a",
        query="机器学习是什么？",
        top_k=5,
        mode="vector",
        min_score=0.0,
        min_rerank_score=0.0,
    )

    expected = [_result("chunk-1")]

    cache.set(
        cache_key,
        expected,
    )

    assert cache.get(cache_key) == expected


def test_search_cache_invalidation_changes_user_key() -> None:
    cache = SearchCache(FakeRedis())

    old_key = cache.build_key(
        user_id="user-a",
        query="报销制度",
        top_k=5,
        mode="vector",
        min_score=0.0,
        min_rerank_score=0.0,
    )

    cache.set(
        old_key,
        [_result("old-chunk")],
    )

    cache.invalidate("user-a")

    new_key = cache.build_key(
        user_id="user-a",
        query="报销制度",
        top_k=5,
        mode="vector",
        min_score=0.0,
        min_rerank_score=0.0,
    )

    assert old_key != new_key
    assert cache.get(new_key) is None
