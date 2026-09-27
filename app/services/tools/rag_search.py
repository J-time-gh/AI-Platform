from typing import Any

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.services import retrieval
from app.services.embedding import Embedder
from app.services.reranker import Reranker
from app.services.tools.base import ToolError


# 参数校验
class RagSearchArguments(BaseModel):
    # 必填
    query: str = Field(min_length=1, max_length=2000)
    # 可选
    top_k: int | None = Field(default=None, ge=1, le=20)


class RagSearchTool:
    name = "rag_search"
    description = "检索企业知识库。适用于询问项目文档、规章制度、上传资料等知识库内容。"
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "要检索的问题",
            },
            "top_k": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    # 依赖注入
    def __init__(
        self,
        db: Session,
        embedder: Embedder,
        reranker: Reranker,
        # 传给检索服务，限定检索范围
        user_id: str,
        # 控制 vector/BM25/hybrid 等算法
        mode: str = "hybrid_rerank",
        default_top_k: int = 5,
    ) -> None:
        self._db = db
        self._embedder = embedder
        self._reranker = reranker
        self._user_id = user_id
        self._mode = mode
        self._default_top_k = default_top_k

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            args = RagSearchArguments.model_validate(arguments)
        except ValidationError as exc:
            raise ToolError(f"rag_search 参数错误：{exc}") from exc

        results = retrieval.search_with_mode(
            db=self._db,
            embedder=self._embedder,
            query=args.query,
            top_k=(args.top_k if args.top_k is not None else self._default_top_k),
            mode=self._mode,
            user_id=self._user_id,
            min_score=settings.min_vector_score,
            reranker=self._reranker,
            min_rerank_score=settings.min_rerank_score,
        )

        return {
            "results": [result.model_dump() for result in results],
            "count": len(results),
        }
