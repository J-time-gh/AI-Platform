"""语义检索：把查询文本向量化，在 chunks 表里找余弦距离最近的块。"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.chunk import Chunk
from app.db.models.document import Document
from app.schemas.search import SearchResult
from app.services.bm25 import get_bm25_index
from app.services.embedding import Embedder
from app.services.fusion import merge_candidates, rrf_fuse
from app.services.highlight import matched_query_terms
from app.services.reranker import Reranker, rerank
from app.services.search_cache import get_search_cache

RECALL_K = 20


def search_chunks(
    db: Session,
    query_vector: list[float],
    top_k: int,
    *,
    user_id: str,
) -> list[SearchResult]:
    """按余弦距离升序返回当前用户最相似的 top_k 个块。"""
    distance = Chunk.embedding.cosine_distance(query_vector)

    rows = db.execute(
        select(Chunk, Document.filename, distance)
        .join(Document, Chunk.document_id == Document.id)
        .where(
            Document.user_id == user_id,
            Document.status == "ready",
        )
        .order_by(distance)
        .limit(top_k)
    ).all()

    return [
        SearchResult(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            filename=filename,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            score=round(1.0 - float(distance_value), 4),
        )
        for chunk, filename, distance_value in rows
    ]


def search(
    db: Session,
    embedder: Embedder,
    query: str,
    top_k: int,
    user_id: str,
) -> list[SearchResult]:
    """把 Query 向量化后，在当前用户自己的 Chunk 中做向量检索。"""
    query_vector = embedder.embed_texts([query])[0]

    return search_chunks(
        db,
        query_vector,
        top_k,
        user_id=user_id,
    )


def search_with_mode(
    db: Session,
    embedder: Embedder,
    query: str,
    top_k: int,
    mode: str = "vector",
    *,
    user_id: str,
    min_score: float = 0.0,
    reranker: Reranker | None = None,
    min_rerank_score: float = 0.0,
) -> list[SearchResult]:
    """按模式检索当前用户的知识库，并缓存最终 SearchResult。

    - vector：只走向量检索
    - bm25：只走关键词检索
    - hybrid：向量与 BM25 分别召回，再做 RRF 融合
    - hybrid_rerank：向量与 BM25 合并候选，再由 CrossEncoder 精排

    Redis 缓存 Key 含 user_id、模式、Query、阈值和用户缓存版本，
    因此不会跨用户复用 SearchResult。
    """
    allowed_modes = {
        "vector",
        "bm25",
        "hybrid",
        "hybrid_rerank",
    }

    if mode not in allowed_modes:
        raise ValueError(f"未知检索模式：{mode}")

    if mode == "hybrid_rerank" and reranker is None:
        raise ValueError("hybrid_rerank 模式必须注入 reranker")

    # 1. 先构建当前用户私有的 Search 缓存 Key。
    search_cache = get_search_cache()

    cache_key = search_cache.build_key(
        user_id=user_id,
        query=query,
        top_k=top_k,
        mode=mode,
        min_score=min_score,
        min_rerank_score=min_rerank_score,
    )

    # 2. 缓存命中时，直接返回最终 SearchResult。
    # cached_results 是 None：缓存未命中、Redis 故障或缓存内容损坏。
    # cached_results 是 []：缓存命中，但没有相关资料。
    cached_results = search_cache.get(cache_key)

    if cached_results is not None:
        return cached_results

    # 3. 缓存未命中时，执行原有检索逻辑。
    if mode == "bm25":
        results = get_bm25_index(user_id).search(
            db,
            query,
            top_k,
        )

    else:
        recall_k = top_k if mode == "vector" else RECALL_K

        vector_results = search(
            db,
            embedder,
            query,
            recall_k,
            user_id,
        )

        # 只对向量腿执行余弦相似度门控。
        if min_score > 0:
            vector_results = [result for result in vector_results if result.score >= min_score]

        if mode == "vector":
            results = vector_results

        else:
            bm25_results = get_bm25_index(user_id).search(
                db,
                query,
                RECALL_K,
            )

            if mode == "hybrid":
                results = rrf_fuse(
                    [vector_results, bm25_results],
                    top_k,
                )

            else:
                # 到这里 mode 只能是 hybrid_rerank；
                # 前面已经保证 reranker 不会是 None。
                assert reranker is not None

                candidates = merge_candidates(
                    [vector_results, bm25_results],
                )

                results = rerank(
                    query,
                    candidates,
                    top_k,
                    reranker,
                    min_rerank_score,
                )

    # 4. 检索完成后才生成展示用高亮词；不改变召回、排序或门控。
    results = [
        result.model_copy(
            update={
                "highlight_terms": matched_query_terms(
                    query,
                    result.content,
                ),
            },
        )
        for result in results
    ]

    # 5. Redis 写入失败不会影响本次正常检索结果。
    search_cache.set(
        cache_key,
        results,
    )

    return results
