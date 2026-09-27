from datetime import UTC, datetime
from uuid import uuid4

from fastapi import APIRouter, HTTPException, UploadFile, status

from app.api.dependencies.auth import CurrentUserDep
from app.db.session import DbSession
from app.schemas.documents import DocumentInfo
from app.services import document_store, file_store
from app.services.ingest_queue import (
    IngestQueueDep,
    QueueUnavailableError,
)
from app.services.search_cache import invalidate_search_cache

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md", ".xlsx"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB


def _cleanup_failed_upload(
    db: DbSession,
    document_id: str,
    user_id: str,
    stored_path: str,
) -> None:
    """入库失败时清理数据库记录与文件。"""
    document_store.delete(db, document_id, user_id)
    file_store.delete_file(stored_path)


@router.post(
    "/upload",
    response_model=DocumentInfo,
    # HTTP_202_ACCEPTED服务端已经接受请求，但处理尚未完成。
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_document(
    file: UploadFile,
    db: DbSession,
    current_user: CurrentUserDep,
    ingest_queue: IngestQueueDep,
) -> DocumentInfo:
    #  API 层处理轻量校验
    filename = file.filename or "unnamed"
    suffix = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"不支持的文件类型：{suffix or '无扩展名'}",
        )

    content = await file.read()

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="文件超过 20MB 限制",
        )
    # 生成 UUID
    # 此时文件已经落到服务器磁盘；即使 Worker 尚未启动，文件也不会丢失。
    document_id = str(uuid4())
    stored_path = file_store.save_file(
        document_id,
        suffix,
        content,
    )

    document = DocumentInfo(
        id=document_id,
        filename=filename,
        size=len(content),
        content_type=file.content_type or "application/octet-stream",
        uploaded_at=datetime.now(UTC),
        status="pending",
        # 流程图的起点：API 已将文件落盘并创建文档记录，尚未确认已入 RQ 队列。
        ingest_stage="upload_received",
        ingest_progress=5,
        ingest_error=None,
    )

    result = document_store.add(
        db,
        document,
        user_id=current_user.id,
        stored_path=stored_path,
        status="pending",
    )

    try:
        # 队列只传 document_id，不传文件二进制、数据库 Session、用户对象或 Embedding 模型。
        # Worker 按 ID 从数据库和磁盘读取数据，让任务轻量且便于重试、排查。
        ingest_queue.enqueue(result.id)
        # Worker 可能极快地先开始执行；mark_queued 只更新仍为 pending 的记录，防止流程倒退。
        document_store.mark_queued(db, result.id)
    # Redis/RQ 不可用时：把错误定位到图中的“排队”节点。
    except QueueUnavailableError:
        document_store.mark_failed(
            db,
            result.id,
            "文档已接收，但暂时无法进入处理队列，请稍后重试。",
            stage="queued",
        )

    return document_store.get(
        db,
        result.id,
        current_user.id,
    )


@router.get("", response_model=list[DocumentInfo])
async def list_documents(
    db: DbSession,
    current_user: CurrentUserDep,
) -> list[DocumentInfo]:
    return document_store.list_all(db, current_user.id)


@router.get("/{doc_id}", response_model=DocumentInfo)
async def get_document(
    doc_id: str,
    db: DbSession,
    current_user: CurrentUserDep,
) -> DocumentInfo:
    document = document_store.get(
        db,
        doc_id,
        current_user.id,
    )

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文档不存在",
        )

    return document


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    doc_id: str,
    db: DbSession,
    current_user: CurrentUserDep,
) -> None:
    stored_path = document_store.get_stored_path(
        db,
        doc_id,
        current_user.id,
    )

    if not document_store.delete(
        db,
        doc_id,
        current_user.id,
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文档不存在",
        )
    # 删除文档后让缓存失效
    invalidate_search_cache(
        current_user.id,
    )
    if stored_path:
        file_store.delete_file(stored_path)
