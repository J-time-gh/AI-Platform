import sys
from types import SimpleNamespace

from redis.exceptions import RedisError

from app.services.embedding import BgeM3Embedder, CachedEmbedder


class CountingEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(index)] for index, _ in enumerate(texts)]


class FakeRedis:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self.data.get(key)

    def setex(
        self,
        key: str,
        _ttl: int,
        value: str,
    ) -> bool:
        self.data[key] = value
        return True

    def delete(self, key: str) -> int:
        return int(self.data.pop(key, None) is not None)


def test_cached_embedder_reuses_cached_vector() -> None:
    inner = CountingEmbedder()
    redis_client = FakeRedis()

    embedder = CachedEmbedder(
        inner,
        redis_client,
    )

    first = embedder.embed_texts(
        ["相同文本"],
    )
    second = embedder.embed_texts(
        ["相同文本"],
    )

    assert first == second
    assert inner.calls == [["相同文本"]]


def test_cached_embedder_deduplicates_batch_texts() -> None:
    inner = CountingEmbedder()
    redis_client = FakeRedis()

    embedder = CachedEmbedder(
        inner,
        redis_client,
    )

    vectors = embedder.embed_texts(
        [
            "苹果",
            "香蕉",
            "苹果",
        ],
    )

    assert inner.calls == [
        [
            "苹果",
            "香蕉",
        ],
    ]
    assert len(vectors) == 3
    assert vectors[0] == vectors[2]


class UnavailableRedis:
    def get(self, _key: str) -> str:
        raise RedisError("Redis 不可用")


def test_cached_embedder_falls_back_when_redis_is_unavailable() -> None:
    inner = CountingEmbedder()

    embedder = CachedEmbedder(
        inner,
        UnavailableRedis(),
    )

    vectors = embedder.embed_texts(
        ["Redis 故障时仍可向量化"],
    )

    assert vectors == [[0.0]]
    assert inner.calls == [
        ["Redis 故障时仍可向量化"],
    ]


def test_bge_embedder_loads_model_only_when_embedding_is_needed(
    monkeypatch,
) -> None:
    class FakeVector(list[float]):
        def tolist(self) -> list[float]:
            return list(self)

    class FakeSentenceTransformer:
        load_count = 0

        def __init__(self, _model: str, *, device: str) -> None:
            self.device = device
            type(self).load_count += 1

        def encode(self, texts: list[str], **_kwargs) -> list[FakeVector]:
            return [FakeVector([0.25]) for _ in texts]

    FakeSentenceTransformer.load_count = 0
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    embedder = BgeM3Embedder()

    assert FakeSentenceTransformer.load_count == 0
    assert embedder.embed_texts([]) == []
    assert FakeSentenceTransformer.load_count == 0

    assert embedder.embed_texts(["需要向量化的文本"]) == [[0.25]]
    assert embedder.embed_texts(["第二段文本"]) == [[0.25]]
    assert FakeSentenceTransformer.load_count == 1
