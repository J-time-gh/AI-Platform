"""创建、复用 Redis Client"""

from functools import lru_cache

from redis import Redis

from app.core.config import settings


# 整个 FastAPI 进程 共用一个 Redis Client（最多缓存一个返回结果）
# Search Cache 的 JSON Embedding Cache 的 JSON 缓存版本号、字符串 Key
@lru_cache(maxsize=1)
def get_redis_client() -> Redis:
    """返回进程内复用的 Redis Client。"""
    return Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_connect_timeout=settings.redis_socket_timeout_seconds,
        socket_timeout=settings.redis_socket_timeout_seconds,
    )


# 包含二进制序列化数据，由 RQ 自己按 bytes 处理和反序列化
# 避免缓存层的 UTF-8 自动解码影响 RQ Job
@lru_cache(maxsize=1)
def get_rq_redis_client() -> Redis:
    """供 RQ 读写二进制 Job 数据使用。"""
    return Redis.from_url(
        settings.redis_url,
        decode_responses=False,
        socket_connect_timeout=settings.redis_socket_timeout_seconds,
        socket_timeout=settings.redis_socket_timeout_seconds,
    )
