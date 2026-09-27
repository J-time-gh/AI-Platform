from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class ChatSession(Base):
    """一个用户拥有的一段持久化聊天会话。"""

    __tablename__ = "chat_sessions"
    # 数据库性能优化
    # 为数据库额外建立了一份目录，数据库级别的额外规则 例如索引 唯一约束 检查约束 联合约束
    __table_args__ = (
        Index(
            "ix_chat_sessions_user_id_updated_at",
            "user_id",
            "updated_at",
        ),
    )
    # 会话自己的UUID，非数据库自增ID。用于在前端和后端之间传递会话标识。
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
    )
    # 用户的UUID，非数据库自增ID。用于在前端和后端之间传递用户标识。
    user_id: Mapped[str] = mapped_column(
        String(36),
        # ondelete删除用户时删除其会话和消息；删除会话时删除其消息。
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    # 会话标题，用户可自定义。可为空，表示未命名会话。
    title: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    # 当前会话选择的 Skill；None 表示普通对话。
    active_skill_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    # 会话创建时间，UTC时间。
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    # 会话更新时间，UTC时间。每次会话或消息更新时都会更新此字段。
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )


# 这是一个 SQLAlchemy ORM 模型
# 继承 Base（来自 base.py）
# 会映射为数据库表
class ChatMessage(Base):
    """属于某个聊天会话的一条用户或助手消息。"""

    # 一条消息对应数据库一行
    __tablename__ = "chat_messages"
    # 表示“这个数据表额外需要的数据库规则”
    __table_args__ = (
        # 角色约束
        # 在 PostgreSQL 中形成类似规则 数据库只允许
        CheckConstraint(
            # 数据库只允许这两个类
            "role IN ('user', 'assistant')",
            # 数据库约束包括：
            # 数据库报错时可识别
            # Alembic 迁移可管理
            # 以后修改 / 删除该约束时可准确指定名称
            name="ck_chat_messages_role",
        ),
        # 消息索引
        # 服务于 chat_session_store.py 中的消息读取
        Index(
            "ix_chat_messages_session_id_created_at",
            "session_id",
            "created_at",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
    )
    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
    )
    # 保存用户问题或助手回答文本
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    # Assistant messages retain retrieval chunks so evidence survives session reloads.
    sources: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
