from fastapi import APIRouter

from app.api.dependencies.auth import CurrentUserDep
from app.core.config import settings
from app.db.session import DbSession
from app.schemas.search import SearchRequest, SearchResult
from app.services import retrieval
from app.services.embedding import EmbedderDep
from app.services.reranker import RerankerDep

router = APIRouter(tags=["search"])


@router.post("/search", response_model=list[SearchResult])
async def search(
    request: SearchRequest,
    db: DbSession,
    embedder: EmbedderDep,
    reranker: RerankerDep,
    current_user: CurrentUserDep,
) -> list[SearchResult]:
    """只检索当前登录用户的知识库。"""
    return retrieval.search_with_mode(
        db,
        embedder,
        request.query,
        request.top_k,
        request.mode,
        user_id=current_user.id,
        min_score=settings.min_vector_score,
        reranker=reranker,
        min_rerank_score=settings.min_rerank_score,
    )
