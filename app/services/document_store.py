"""documents 表的数据访问层。"""

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.db.models.document import Document
from app.schemas.documents import DocumentInfo, IngestStage


def _to_info(row: Document) -> DocumentInfo:
    """ORM 对象转换为接口返回模型。"""
    return DocumentInfo(
        id=row.id,
        filename=row.filename,
        size=row.size,
        content_type=row.content_type,
        uploaded_at=row.uploaded_at,
        status=row.status,
        ingest_stage=row.ingest_stage,
        ingest_progress=row.ingest_progress,
        ingest_error=row.ingest_error,
    )


def add(
    db: Session,
    document: DocumentInfo,
    *,
    user_id: str,
    stored_path: str | None = None,
    status: str = "pending",
) -> DocumentInfo:
    """创建属于指定用户的文档记录。"""
    row = Document(
        id=document.id,
        user_id=user_id,
        filename=document.filename,
        size=document.size,
        content_type=document.content_type,
        uploaded_at=document.uploaded_at,
        stored_path=stored_path,
        status=status,
        # 初始节点记录：文件已保存并创建记录；API 随后会尝试把任务投递到队列。
        ingest_stage=document.ingest_stage,
        ingest_progress=document.ingest_progress,
        ingest_error=None,
    )

    db.add(row)
    db.commit()
    db.refresh(row)

    return _to_info(row)


def list_all(db: Session, user_id: str) -> list[DocumentInfo]:
    """列出指定用户的全部文档。"""
    rows = db.scalars(
        select(Document).where(Document.user_id == user_id).order_by(Document.uploaded_at),
    ).all()

    return [_to_info(row) for row in rows]


def _get_row(
    db: Session,
    doc_id: str,
    user_id: str,
) -> Document | None:
    """按文档 ID 和所属用户查询记录。"""
    return db.scalar(
        select(Document).where(
            Document.id == doc_id,
            Document.user_id == user_id,
        ),
    )


def get(
    db: Session,
    doc_id: str,
    user_id: str,
) -> DocumentInfo | None:
    """获取指定用户拥有的文档。"""
    row = _get_row(db, doc_id, user_id)

    return _to_info(row) if row is not None else None


def get_stored_path(
    db: Session,
    doc_id: str,
    user_id: str,
) -> str | None:
    """获取指定用户拥有的文档存储路径。"""
    row = _get_row(db, doc_id, user_id)

    return row.stored_path if row is not None else None


def delete(
    db: Session,
    doc_id: str,
    user_id: str,
) -> bool:
    """删除指定用户拥有的文档。"""
    row = _get_row(db, doc_id, user_id)

    if row is None:
        return False

    db.delete(row)
    db.commit()

    return True


# 以下方法由 API/Worker 使用，专门维护每份文档的实时流程图节点。
def mark_queued(
    db: Session,
    document_id: str,
) -> bool:
    """记录任务已进入 RQ 队列；不覆盖已被快速 Worker 领取的阶段。"""
    # Worker 可能在 enqueue() 返回前就已领取任务，因此用带状态条件的 UPDATE
    # 原子写入：只有数据库里仍为 pending 时才能设置 queued，避免覆盖 worker_start。
    result = db.execute(
        update(Document)
        .where(
            Document.id == document_id,
            Document.status == "pending",
        )
        .values(
            ingest_stage="queued",
            ingest_progress=10,
        )
        .execution_options(synchronize_session=False),
    )
    db.commit()
    # 上传请求的 Session 可能缓存了旧记录；过期缓存后才能读到 Worker 实际的最新阶段。
    db.expire_all()
    return result.rowcount > 0


def claim_for_ingest(
    db: Session,
    document_id: str,
) -> Document | None:
    """原子认领一个 pending 文档，避免重复 Worker 同时处理。"""
    row = db.scalar(
        # with_for_update() 会给这一行加数据库行锁
        # 只允许 pending 状态被认领
        # 避免了两个 Worker 同时处理同一个 Document
        select(Document).where(Document.id == document_id).with_for_update(),
    )

    if row is None or row.status != "pending":
        return None

    row.status = "processing"
    row.ingest_error = None
    # 流程图进入独立 Worker 分支，标记 Worker 已领取该文档任务。
    row.ingest_stage = "worker_start"
    row.ingest_progress = 20
    db.commit()
    db.refresh(row)

    return row


def update_ingest_progress(
    db: Session,
    document_id: str,
    stage: IngestStage,
    progress: int,
) -> bool:
    """持久化 Worker 当前运行节点，供文档状态 API 和前端流程图轮询。"""
    if not 0 <= progress <= 99:
        raise ValueError("处理中进度必须在 0 到 99 之间")

    row = db.get(Document, document_id)
    if row is None or row.status != "processing":
        return False

    # 仅接受单调递增的进度，避免旧/重复更新让流程图倒退。
    row.ingest_stage = stage
    row.ingest_progress = max(row.ingest_progress, progress)
    db.commit()
    return True


def mark_ready(
    db: Session,
    document_id: str,
) -> str | None:
    """将 processing 文档标记 ready，返回所属 user_id。"""
    row = db.get(Document, document_id)

    if row is None:
        return None
    # 只有成功后才使 Search Cache 失效，因为此刻知识库内容才真正发生变化
    row.status = "ready"
    row.ingest_error = None
    # 最终流程图节点：Worker 已成功写入所有块和向量。
    row.ingest_stage = "ready"
    row.ingest_progress = 100
    db.commit()

    return row.user_id


def mark_failed(
    db: Session,
    document_id: str,
    error_message: str,
    *,
    stage: IngestStage | None = None,
) -> str | None:
    """记录失败状态；保留失败节点，让 UI 能把红色错误标在准确流程步骤。"""
    row = db.get(Document, document_id)

    if row is None:
        return None

    row.status = "failed"
    row.ingest_error = error_message[:1000]
    # Worker 失败默认保留正在执行的 ingest_stage；API 可显式标注入队失败节点。
    if stage is not None:
        row.ingest_stage = stage
    db.commit()

    return row.user_id
