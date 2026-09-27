from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# 数据库文档状态定义
DocumentStatus = Literal[
    "pending",
    "processing",
    "ready",
    "failed",
]

# 阶段名是 API 与前端流程图之间的稳定契约。
# 前端依据这些值点亮节点；unknown 仅供升级前已失败、没有历史阶段信息的旧记录使用。
IngestStage = Literal[
    "upload_received",
    "queued",
    "worker_start",
    "parsing",
    "chunking",
    "embedding",
    "saving",
    "ready",
    "unknown",
]


class DocumentInfo(BaseModel):
    id: str
    filename: str
    size: int
    content_type: str
    uploaded_at: datetime
    status: DocumentStatus
    # 前端流程图当前节点与粗粒度进度；这里不暴露文档正文或 Worker 内部敏感数据。
    ingest_stage: IngestStage = "upload_received"
    ingest_progress: int = Field(default=5, ge=0, le=100)
    ingest_error: str | None
