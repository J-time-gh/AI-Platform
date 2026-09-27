from functools import lru_cache
from typing import Annotated, Protocol

from fastapi import Depends
from redis.exceptions import RedisError
from rq import Queue

from app.core.config import settings
from app.services.redis_client import get_rq_redis_client


class QueueUnavailableError(Exception):
    """Redis / RQ 不可用，任务无法投递。"""


class IngestQueue(Protocol):
    """文档入库任务队列的抽象接口。"""

    def enqueue(self, document_id: str) -> str:
        """投递文档入库任务，返回 RQ job ID。"""


# 生产实现
class RqIngestQueue:
    """生产环境的 RQ 队列实现。"""

    def __init__(self) -> None:
        self._queue = Queue(
            settings.ingest_queue_name,
            connection=get_rq_redis_client(),
            default_timeout=settings.ingest_job_timeout_seconds,
        )

    def enqueue(self, document_id: str) -> str:
        try:
            job = self._queue.enqueue(
                "app.workers.document_ingest.process_document",
                document_id,
            )
        except RedisError as exc:
            raise QueueUnavailableError(
                "文档处理队列暂时不可用",
            ) from exc

        return job.id


# FastAPI 依赖注入
@lru_cache(maxsize=1)
def get_ingest_queue() -> IngestQueue:
    return RqIngestQueue()


IngestQueueDep = Annotated[IngestQueue, Depends(get_ingest_queue)]
