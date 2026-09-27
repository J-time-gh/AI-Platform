"""会话数据访问层。

职责：
- 创建会话
- 列出会话
- 读取会话
- 读取消息
- 写入一轮消息
- 删除会话
"""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.chat import ChatMessage, ChatSession
from app.schemas.chat_session import (
    ChatSessionResponse,
    PersistedChatMessageResponse,
)


def _to_session_response(
    row: ChatSession,
) -> ChatSessionResponse:
    return ChatSessionResponse(
        id=row.id,
        title=row.title,
        active_skill_id=row.active_skill_id,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


# 把数据库 JSON 来源返回给 API
def _to_message_response(
    row: ChatMessage,
) -> PersistedChatMessageResponse:
    return PersistedChatMessageResponse(
        id=row.id,
        role=row.role,
        content=row.content,
        sources=row.sources or [],
        created_at=row.created_at,
    )


def _get_session_row(
    db: Session,
    session_id: str,
    user_id: str,
) -> ChatSession | None:
    """只查询当前用户拥有的会话。"""
    return db.scalar(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == user_id,
        ),
    )


def create(
    db: Session,
    *,
    user_id: str,
    title: str | None,
) -> ChatSessionResponse:
    now = datetime.now(UTC)

    row = ChatSession(
        id=str(uuid4()),
        user_id=user_id,
        title=title,
        active_skill_id=None,
        created_at=now,
        updated_at=now,
    )
    # → 加入当前数据库会话
    db.add(row)
    # → 真实写入 PostgreSQL
    db.commit()
    # → 从数据库重新读取该对象
    db.refresh(row)

    return _to_session_response(row)


# 用用户 ID 和会话 ID 找行，更新 Skill 并提交事务
def set_active_skill(
    db: Session,
    *,
    session_id: str,
    user_id: str,
    active_skill_id: str | None,
) -> ChatSessionResponse | None:
    """更新当前用户会话的 Skill；None 表示会话不存在或不属于用户。"""
    row = _get_session_row(
        db,
        session_id,
        user_id,
    )
    if row is None:
        return None

    row.active_skill_id = active_skill_id
    row.updated_at = datetime.now(UTC)
    db.commit()
    db.refresh(row)

    return _to_session_response(row)


def list_all(
    db: Session,
    user_id: str,
) -> list[ChatSessionResponse]:
    rows = db.scalars(
        select(ChatSession)
        .where(ChatSession.user_id == user_id)
        # desc降序排序 （最近的会话排在最前面）
        .order_by(ChatSession.updated_at.desc()),
    ).all()

    return [_to_session_response(row) for row in rows]


def get(
    db: Session,
    session_id: str,
    user_id: str,
) -> ChatSessionResponse | None:
    row = _get_session_row(
        db,
        session_id,
        user_id,
    )

    return _to_session_response(row) if row else None


def list_messages(
    db: Session,
    session_id: str,
    user_id: str,
) -> list[PersistedChatMessageResponse] | None:
    """返回 None 表示会话不属于当前用户或不存在。"""
    session = _get_session_row(
        db,
        session_id,
        user_id,
    )
    if session is None:
        return None

    rows = db.scalars(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at),
    ).all()

    return [_to_message_response(row) for row in rows]


# 写入一轮消息
def add_turn(
    db: Session,
    *,
    session_id: str,
    user_id: str,
    user_content: str,
    assistant_content: str,
    assistant_sources: list[dict[str, Any]] | None = None,
) -> bool:
    """一次写入用户消息和助手消息，确保它们属于同一会话。"""
    # 用户范围验证
    session = _get_session_row(
        db,
        session_id,
        user_id,
    )
    if session is None:
        return False

    now = datetime.now(UTC)
    # 一次加入两条记录
    db.add_all(
        [
            ChatMessage(
                id=str(uuid4()),
                session_id=session.id,
                role="user",
                content=user_content,
                sources=[],
                created_at=now,
            ),
            ChatMessage(
                id=str(uuid4()),
                session_id=session.id,
                role="assistant",
                content=assistant_content,
                sources=assistant_sources or [],
                created_at=now,
            ),
        ],
    )

    session.updated_at = now

    db.commit()

    return True


def delete(
    db: Session,
    session_id: str,
    user_id: str,
) -> bool:
    row = _get_session_row(
        db,
        session_id,
        user_id,
    )
    if row is None:
        return False

    db.delete(row)
    db.commit()

    return True
