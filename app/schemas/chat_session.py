from datetime import datetime

from pydantic import BaseModel, Field

from app.schemas.search import SearchResult


class CreateChatSessionRequest(BaseModel):
    """创建会话的请求。"""

    title: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
    )


class UpdateChatSessionSkillRequest(BaseModel):
    """更新会话当前激活 Skill 的请求。"""

    active_skill_id: str | None = Field(
        ...,
        max_length=64,
    )


class ChatSessionResponse(BaseModel):
    """会话列表或创建会话时的返回结构。"""

    id: str
    title: str | None
    active_skill_id: str | None = None
    created_at: datetime
    updated_at: datetime


class PersistedChatMessageResponse(BaseModel):
    """数据库中已持久化的一条聊天消息。"""

    id: str
    role: str
    content: str
    sources: list[SearchResult] = Field(default_factory=list)
    created_at: datetime
