import hashlib
import json
from functools import lru_cache

from pydantic import ValidationError
from redis import Redis
from redis.exceptions import RedisError

from app.core.config import settings
from app.schemas.search import SearchResult
from app.services.redis_client import get_redis_client


class SearchCache:
    """按用户范围缓存 SearchResult 列表。"""

    def __init__(
        self,
        redis_client: Redis,
    ) -> None:
        self._redis = redis_client

    # 不同用户的缓存版本完全独立
    @staticmethod
    def _version_key(user_id: str) -> str:
        # v2 起缓存载荷包含 highlight_terms，避免复用旧结构的搜索结果。
        return f"search-version:v2:{user_id}"

    def _get_version(
        self,
        user_id: str,
    ) -> str:
        """读取用户当前缓存版本；首次访问默认初始化为 1。"""
        version_key = self._version_key(user_id)
        version = self._redis.get(version_key)
        # 初始化
        if version is None:
            self._redis.set(
                version_key,
                "1",
                # 只有 Key 不存在时才写入
                nx=True,
            )
            version = self._redis.get(version_key)

        return version or "1"

    def build_key(
        self,
        *,
        user_id: str,
        query: str,
        top_k: int,
        mode: str,
        min_score: float,
        min_rerank_score: float,
    ) -> str | None:
        """生成当前检索参数对应的用户私有缓存 Key。"""
        try:
            version = self._get_version(user_id)
        except RedisError:
            return None

        parameters = {
            "embedding_model": settings.embedding_model,
            "min_rerank_score": min_rerank_score,
            "min_score": min_score,
            "mode": mode,
            "query": query,
            "rerank_model": settings.rerank_model,
            "top_k": top_k,
        }

        fingerprint = hashlib.sha256(
            json.dumps(
                parameters,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode(),
        ).hexdigest()

        return f"search:v2:{user_id}:{version}:{fingerprint}"

    def get(
        self,
        cache_key: str | None,
    ) -> list[SearchResult] | None:
        """命中缓存时返回结果；缓存未命中或 Redis 故障时返回 None。"""
        if cache_key is None:
            return None

        try:
            raw_value = self._redis.get(cache_key)
        except RedisError:
            return None

        if raw_value is None:
            return None

        try:
            payload = json.loads(raw_value)
            return [SearchResult.model_validate(item) for item in payload]
        except (
            TypeError,
            ValidationError,
            json.JSONDecodeError,
        ):
            try:
                self._redis.delete(cache_key)
            except RedisError:
                pass

            return None

    def set(
        self,
        cache_key: str | None,
        results: list[SearchResult],
    ) -> None:
        """写入 SearchResult 缓存；Redis 故障不影响主流程。"""
        if cache_key is None:
            return

        payload = [result.model_dump(mode="json") for result in results]

        try:
            self._redis.setex(
                cache_key,
                settings.redis_search_ttl_seconds,
                json.dumps(payload),
            )
        except RedisError:
            pass

    def invalidate(
        self,
        user_id: str,
    ) -> None:
        """递增用户缓存版本，使其旧 Search 缓存自然失效。"""
        try:
            self._redis.incr(
                self._version_key(user_id),
            )
        except RedisError:
            pass


@lru_cache(maxsize=1)
def get_search_cache() -> SearchCache:
    """返回进程内复用的 SearchCache。"""
    return SearchCache(
        get_redis_client(),
    )


def invalidate_search_cache(
    user_id: str,
) -> None:
    """供上传、删除文档等数据变更操作调用。"""
    get_search_cache().invalidate(user_id)
