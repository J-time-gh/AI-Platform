from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.schemas.search import SearchResult


# 一条对话消息
class ChatMessage(BaseModel):
    # 谁说的
    role: Literal["user", "assistant", "system"]
    # 内容是
    content: str = Field(min_length=1)


# 用户发来的请求
class ChatRequest(BaseModel):
    # 对话长度限制
    message: str = Field(min_length=1, max_length=2000)
    # 历史
    # 执行过程：每来一个请求 → 创建新的 ChatRequest 实例 → Pydantic 发现没传 history
    # → 调用一次 list() → 得到一个崭新的空列表。请求 A 和请求 B 拿到的永远是各不相同的列表，
    # 互不影响。
    # default_factory=list每次都会调用list()，而不是在类定义时就创建一个空列表，
    # 这样可以避免多个实例共享同一个列表的问题。
    history: list[ChatMessage] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=20)
    # 用哪个大模型：不传则用 settings.llm_model；传了走指定模型（Gateway 雏形）
    model: str | None = None
    # 用哪种检索器：vector = 向量语义检索，bm25 = 关键词检索
    # 加 "hybrid"
    mode: Literal["vector", "bm25", "hybrid", "hybrid_rerank"] = "vector"
    session_id: str | None = None

    @model_validator(mode="after")
    def reject_history_with_session_id(self) -> "ChatRequest":
        if self.session_id and self.history:
            raise ValueError(
                "使用 session_id 时不能同时传 history",
            )

        return self


# 接口返回的数据
class ChatResponse(BaseModel):
    # 回答内容
    reply: str
    model: str
    sources: list[SearchResult] = Field(default_factory=list)
