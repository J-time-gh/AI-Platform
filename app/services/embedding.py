"""文本向量化：真实实现（bge-m3 + GPU）与测试替身（FakeEmbedder）。"""

import hashlib
import json
import os
from functools import lru_cache
from typing import Annotated, Protocol

from fastapi import Depends
from redis import Redis
from redis.exceptions import RedisError

from app.core.config import settings
from app.db.models.chunk import EMBEDDING_DIM
from app.services.redis_client import get_redis_client


class Embedder(Protocol):
    """向量化接口：把一批文本转成向量列表。"""

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


class BgeM3Embedder:
    """bge-m3 实现：懒加载模型，跑在 GPU 上。"""

    def __init__(self) -> None:
        os.environ.setdefault("HF_ENDPOINT", settings.hf_endpoint)
        os.environ.setdefault("HF_HOME", settings.hf_home)
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import (
                SentenceTransformer,  # 懒导入：普通对话不需要加载向量模型
            )

            self._model = SentenceTransformer(
                settings.embedding_model,
                device=settings.embedding_device,
            )
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._load_model().encode(
            texts,
            batch_size=settings.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]


class FakeEmbedder:
    """测试替身：由文本哈希生成确定性向量，无需模型/GPU。"""

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self._dim = dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[i % len(digest)] / 255.0 - 0.5 for i in range(self._dim)]


# Embedding Redis缓存部分新加入
class CachedEmbedder:
    """为真实 Embedder 增加 Redis 缓存的装饰器。"""

    def __init__(
        self,
        inner: Embedder,
        redis_client: Redis,
    ) -> None:
        self._inner = inner
        self._redis = redis_client

    @staticmethod
    def _cache_key(text: str) -> str:
        """根据模型和文本内容生成稳定、不暴露原文的 Redis Key。"""
        model_digest = hashlib.sha256(
            settings.embedding_model.encode(),
        ).hexdigest()[:16]

        text_digest = hashlib.sha256(
            text.encode(),
        ).hexdigest()

        return f"embedding:v1:{model_digest}:{text_digest}"

    def embed_texts(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        if not texts:
            return []

        unique_texts = list(dict.fromkeys(texts))
        vectors_by_text: dict[str, list[float]] = {}
        missing_texts: list[str] = []
        # 缓存命中流程
        try:
            for text in unique_texts:
                cached_value = self._redis.get(
                    self._cache_key(text),
                )

                if cached_value is None:
                    missing_texts.append(text)
                    continue

                try:
                    vectors_by_text[text] = json.loads(
                        cached_value,
                    )
                except json.JSONDecodeError:
                    self._redis.delete(
                        self._cache_key(text),
                    )
                    missing_texts.append(text)

        except RedisError:
            # Redis 故障时不影响主功能，直接调用真实模型。
            return self._inner.embed_texts(texts)

        if missing_texts:
            new_vectors = self._inner.embed_texts(
                missing_texts,
            )

            for text, vector in zip(
                missing_texts,
                new_vectors,
                # 如果文本数量和向量数量不一致→ 立即抛异常
                strict=True,
            ):
                vectors_by_text[text] = vector
            # 写入 Redis 缓存
            try:
                for text in missing_texts:
                    # setex=写入 Key→ 同时设置过期时间
                    self._redis.setex(
                        self._cache_key(text),
                        settings.redis_embedding_ttl_seconds,
                        json.dumps(vectors_by_text[text]),
                    )
            except RedisError:
                # 向 Redis 写缓存失败不影响本次已生成的向量。
                pass

        return [vectors_by_text[text] for text in texts]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """FastAPI 依赖：真实 Embedder 外包一层 Redis 缓存。"""
    return CachedEmbedder(
        BgeM3Embedder(),
        get_redis_client(),
    )


EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
