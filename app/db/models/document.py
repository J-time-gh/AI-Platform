from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


# 数据库模型
class Document(Base):
    """文档元数据表。"""

    __tablename__ = "documents"

    __table_args__ = (
        # CheckConstraint数据库自身限制状态值，避免代码写入拼错的状态
        CheckConstraint(
            "status IN ('pending', 'processing', 'ready', 'failed')",
            name="ck_documents_status",
        ),
        Index(
            "ix_documents_user_id_status",
            "user_id",
            "status",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    # JWT 关联用户
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    filename: Mapped[str] = mapped_column(String(255))
    size: Mapped[int] = mapped_column(Integer)
    content_type: Mapped[str] = mapped_column(String(100))
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    stored_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    # 状态
    # pending processing ready failed
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        server_default="pending",
    )

    # 入库流程图节点：记录这份文档当前走到哪个 Worker/API 阶段。
    # UI 根据这个字段高亮节点；它是持久化状态，因此刷新页面后仍能恢复进度。
    ingest_stage: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="upload_received",
        server_default="upload_received",
    )
    # 粗粒度进度里程碑，配合 ingest_stage 让跳过轮询间隔的前端也能画出已完成节点。
    ingest_progress: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=5,
        server_default="5",
    )

    ingest_error: Mapped[str | None] = mapped_column(
        String(1000),
        nullable=True,
    )
