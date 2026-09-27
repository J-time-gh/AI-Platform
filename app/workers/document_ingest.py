from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.schemas.documents import IngestStage
from app.services import document_store
from app.services.embedding import Embedder, get_embedder
from app.services.ingest import IngestError, ingest_document
from app.services.search_cache import invalidate_search_cache


def _process_document(
    db: Session,
    document_id: str,
    embedder: Embedder,
) -> None:
    """可单元测试的 Worker 核心逻辑。"""
    document = document_store.claim_for_ingest(
        db,
        document_id,
    )

    # 已删除、已处理或已被其他 Worker 认领。
    if document is None:
        return

    try:
        # 把 Worker 内部的阶段变化写回 DB；UI 的文档卡片会轮询并点亮流程图节点。
        def report_progress(stage: IngestStage, progress: int) -> None:
            document_store.update_ingest_progress(
                db,
                document.id,
                stage,
                progress,
            )

        ingest_document(
            db,
            document.id,
            document.stored_path,
            embedder,
            progress_reporter=report_progress,
        )

    except IngestError as exc:
        document_store.mark_failed(
            db,
            document.id,
            str(exc),
        )
        return

    except Exception:
        db.rollback()

        document_store.mark_failed(
            db,
            document.id,
            "文档处理失败，请稍后重试。",
        )
        raise

    user_id = document_store.mark_ready(
        db,
        document.id,
    )

    if user_id:
        invalidate_search_cache(user_id)


# 生产入口
# RQ 调用的函数
# Worker 是独立进程，不能复用 FastAPI 请求中的数据库 Session
def process_document(document_id: str) -> None:
    """RQ 调用的生产环境入口。"""
    db = SessionLocal()

    try:
        _process_document(
            db,
            document_id,
            get_embedder(),
        )
    finally:
        db.close()
