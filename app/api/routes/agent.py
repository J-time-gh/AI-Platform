from fastapi import APIRouter, HTTPException, status

from app.api.dependencies.auth import CurrentUserDep
from app.core.config import settings
from app.db.session import DbSession
from app.schemas.agent import AgentRequest, AgentResponse
from app.services.agent import AgentProtocolError, AgentService
from app.services.embedding import EmbedderDep
from app.services.llm import LLMClientDep, LLMUnavailable
from app.services.reranker import RerankerDep
from app.services.tools.python_sandbox import PythonSandboxTool
from app.services.tools.rag_search import RagSearchTool
from app.services.tools.registry import ToolRegistry
from app.services.tools.sql_query import SqlQueryTool

router = APIRouter(tags=["agent"])


@router.post("/agent", response_model=AgentResponse)
async def agent(
    request: AgentRequest,
    *,
    db: DbSession,
    embedder: EmbedderDep,
    reranker: RerankerDep,
    llm: LLMClientDep,
    current_user: CurrentUserDep,
) -> AgentResponse:
    registry = ToolRegistry()

    registry.register(
        RagSearchTool(
            db=db,
            embedder=embedder,
            reranker=reranker,
            user_id=current_user.id,
        ),
    )
    registry.register(
        SqlQueryTool(
            db=db,
            user_id=current_user.id,
        ),
    )
    registry.register(PythonSandboxTool())

    service = AgentService(
        llm=llm,
        registry=registry,
        max_steps=settings.agent_max_steps,
    )

    try:
        result = await service.run(
            question=request.message,
            model=request.model,
        )

    except LLMUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    except AgentProtocolError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"模型返回格式错误：{exc}",
        ) from exc

    return AgentResponse.model_validate(result)
