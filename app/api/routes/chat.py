"""聊天接口：普通对话直连模型，Skill 对话由 Agent 执行。"""

import asyncio
import json
import logging
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.dependencies.auth import CurrentUserDep
from app.core.config import settings
from app.db.session import DbSession
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.search import SearchResult
from app.services import chat_session_store
from app.services.agent import AgentProtocolError, AgentService
from app.services.embedding import EmbedderDep
from app.services.llm import LLMClientDep, LLMUnavailable
from app.services.reranker import RerankerDep
from app.services.skill_registry import SkillNotFound, skill_registry
from app.services.tools.python_sandbox import PythonSandboxTool
from app.services.tools.rag_search import RagSearchTool
from app.services.tools.registry import ToolRegistry
from app.services.tools.sql_query import SqlQueryTool

# 余弦相似度低于此值视为"没检索到相关资料"（仅用于 vector 模式）
# MIN_VECTOR_SCORE = 0.45

router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)

# 前端流程图接收的结构化事件；detail 只放安全摘要，不放隐藏推理或工具原始输出。
ChatTraceCallback = Callable[[dict[str, Any]], None]

GENERAL_CHAT_SYSTEM_PROMPT = (
    "你是一个可靠的 AI 助手。根据用户问题和对话历史直接回答。"
    "不要声称检索过用户的文件或知识库；不确定时请明确说明。"
)


def _collect_rag_sources(agent_result: dict[str, Any]) -> list[SearchResult]:
    sources_by_chunk: dict[str, SearchResult] = {}

    for tool_call in agent_result.get("tool_calls", []):
        if tool_call.get("tool_name") != "rag_search" or not tool_call.get("ok"):
            continue

        output = tool_call.get("output")
        if not isinstance(output, dict):
            continue

        for item in output.get("results", []):
            source = SearchResult.model_validate(item)
            sources_by_chunk.setdefault(source.chunk_id, source)

    return list(sources_by_chunk.values())


async def _run_chat_request(
    request: ChatRequest,
    *,
    db: DbSession,
    embedder: EmbedderDep,
    llm: LLMClientDep,
    reranker: RerankerDep,
    current_user: CurrentUserDep,
    event_callback: ChatTraceCallback | None = None,
) -> ChatResponse:
    def emit_flow(
        node_id: str,
        node_status: str,
        label: str,
        detail: str | None = None,
    ) -> None:
        """记录后端实际到达的聊天节点，供 /chat/stream 实时绘图。"""
        if event_callback is not None:
            event_callback(
                {
                    "kind": "flow",
                    "id": node_id,
                    "status": node_status,
                    "label": label,
                    "detail": detail,
                },
            )

    emit_flow("chat-request", "complete", "收到用户问题")
    history_messages = [message.model_dump() for message in request.history]
    active_skill_id: str | None = None
    emit_flow("chat-context", "active", "读取会话与历史记录")

    if request.session_id:
        session = chat_session_store.get(
            db,
            request.session_id,
            current_user.id,
        )
        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="会话不存在",
            )

        active_skill_id = session.active_skill_id
        persisted_messages = chat_session_store.list_messages(
            db,
            request.session_id,
            current_user.id,
        )
        if persisted_messages is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="会话不存在",
            )

        history_messages = [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in persisted_messages
        ]

    emit_flow("chat-context", "complete", "会话上下文已准备")
    skill = None
    if active_skill_id is not None:
        try:
            skill = skill_registry.get(active_skill_id)
        except SkillNotFound as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"会话绑定的 Skill 已不可用：{active_skill_id}",
            ) from exc

    mode_label = "普通对话（不检索知识库）" if skill is None else f"Skill：{skill.label}"
    emit_flow("chat-mode", "complete", "识别本轮处理模式", mode_label)

    if skill is None:
        messages = [
            {
                "role": "system",
                "content": GENERAL_CHAT_SYSTEM_PROMPT,
            },
            *history_messages,
            {
                "role": "user",
                "content": request.message,
            },
        ]

        emit_flow("chat-llm-final", "active", "请求大模型生成回答")
        try:
            reply = await llm.chat(messages, model=request.model)
        except LLMUnavailable as exc:
            emit_flow("chat-llm-final", "failed", "大模型请求失败")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc

        emit_flow("chat-llm-final", "complete", "大模型已返回回答")
        response = ChatResponse(
            reply=reply,
            model=request.model or settings.llm_model,
            sources=[],
        )
    else:
        registry = ToolRegistry()
        registry.register(
            RagSearchTool(
                db=db,
                embedder=embedder,
                reranker=reranker,
                user_id=current_user.id,
                mode=request.mode,
                default_top_k=request.top_k,
            ),
        )
        registry.register(
            SqlQueryTool(
                db=db,
                user_id=current_user.id,
            ),
        )
        registry.register(PythonSandboxTool())

        agent = AgentService(
            llm=llm,
            registry=registry,
            max_steps=settings.agent_max_steps,
            skill=skill,
            event_callback=event_callback,
        )

        try:
            result = await agent.run(
                question=request.message,
                model=request.model,
                history=history_messages,
            )
        except LLMUnavailable as exc:
            emit_flow("chat-agent", "failed", "Agent 模型调用失败")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        except AgentProtocolError as exc:
            emit_flow("chat-agent", "failed", "Agent 决策格式错误")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"模型返回格式错误：{exc}",
            ) from exc

        emit_flow("chat-answer", "complete", "整理回答与引用来源")
        response = ChatResponse(
            reply=result["answer"],
            model=request.model or settings.llm_model,
            sources=_collect_rag_sources(result),
        )

    if request.session_id:
        emit_flow("chat-persist", "active", "保存本轮问答到会话")
        chat_session_store.add_turn(
            db,
            session_id=request.session_id,
            user_id=current_user.id,
            user_content=request.message,
            assistant_content=response.reply,
            assistant_sources=[source.model_dump(mode="json") for source in response.sources],
        )
        emit_flow("chat-persist", "complete", "问答与引用已保存")

    return response


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    *,
    db: DbSession,
    embedder: EmbedderDep,
    llm: LLMClientDep,
    reranker: RerankerDep,
    current_user: CurrentUserDep,
) -> ChatResponse:
    """兼容原有 JSON 聊天接口，不要求 API 客户端改动。"""
    return await _run_chat_request(
        request,
        db=db,
        embedder=embedder,
        llm=llm,
        reranker=reranker,
        current_user=current_user,
    )


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    *,
    db: DbSession,
    embedder: EmbedderDep,
    llm: LLMClientDep,
    reranker: RerankerDep,
    current_user: CurrentUserDep,
) -> StreamingResponse:
    """以 SSE 推送真实聊天节点；最终回答仍使用原 ChatResponse 数据结构。"""
    loop = asyncio.get_running_loop()
    event_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    def publish(event: dict[str, Any]) -> None:
        # Agent 同步工具节点可能在 LangGraph 的工作线程执行；必须线程安全地写入 asyncio 队列。
        loop.call_soon_threadsafe(event_queue.put_nowait, event)

    async def execute_request() -> None:
        try:
            response = await _run_chat_request(
                request,
                db=db,
                embedder=embedder,
                llm=llm,
                reranker=reranker,
                current_user=current_user,
                event_callback=publish,
            )
            publish(
                {
                    "kind": "complete",
                    "response": response.model_dump(mode="json"),
                },
            )
        except HTTPException as exc:
            publish(
                {
                    "kind": "flow",
                    "id": "chat-failure",
                    "status": "failed",
                    "label": "聊天请求未完成",
                    "detail": "请查看下方错误信息",
                },
            )
            publish(
                {
                    "kind": "error",
                    "detail": str(exc.detail),
                    "status_code": exc.status_code,
                },
            )
        except Exception:
            logger.exception("Unexpected error in streamed chat request")
            publish(
                {
                    "kind": "flow",
                    "id": "chat-failure",
                    "status": "failed",
                    "label": "聊天流程异常中断",
                    "detail": "请查看服务端日志",
                },
            )
            publish(
                {
                    "kind": "error",
                    "detail": "聊天处理失败，请查看服务端日志",
                    "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                },
            )
        finally:
            loop.call_soon_threadsafe(event_queue.put_nowait, None)

    async def event_stream():
        task = asyncio.create_task(execute_request())
        try:
            while True:
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=15)
                except TimeoutError:
                    # 发送注释行维持连接，避免代理把长时间模型调用误判为空闲。
                    yield ": keep-alive\n\n"
                    continue

                if event is None:
                    break
                yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
