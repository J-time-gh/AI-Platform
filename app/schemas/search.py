from typing import Literal

from pydantic import BaseModel, Field


# 用户输入什么
class SearchRequest(BaseModel):
    """检索请求。"""

    query: str = Field(min_length=1, max_length=2000)
    # 返回最相关的结果数量           #至少1  #至多20
    top_k: int = Field(default=5, ge=1, le=20)
    # ← 加 "hybrid"
    # # 默认向量，保证旧测试/旧调用不受影响
    mode: Literal["vector", "bm25", "hybrid", "hybrid_rerank"] = "vector"


# 单条检索结果模型
# 系统返回什么
class SearchResult(BaseModel):
    """单条检索结果。"""

    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    content: str
    score: float  # 相似度（1 - 余弦距离），越接近 1 越相关
    # 后端依据 jieba 分词得出的、同时出现在查询和正文中的字面词元。
    # 仅供前端展示高亮，不参与检索排序。
    highlight_terms: list[str] = Field(default_factory=list)
