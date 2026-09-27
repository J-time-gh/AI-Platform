"""RRF（Reciprocal Rank Fusion）融合：把多路检索结果按"排名"合成一份。

为什么用排名而不是分数：向量检索返回余弦相似度（0~1），BM25 返回词频分
（0~几十），两者量纲不同，**不能相加、不能比大小**。RRF 只用每个结果在各自
列表里的"第几名"来投票，天然规避了量纲问题。

公式：score(块) = Σ 1 / (k + rank)，k 默认 60（原论文推荐值，起平滑作用）。
"""

from app.schemas.search import SearchResult

RRF_K = 60  # 平滑常数：越大则"排名靠后"的惩罚越轻


def rrf_fuse(
    result_lists: list[list[SearchResult]],
    top_k: int,
    k: int = RRF_K,
) -> list[SearchResult]:
    """把多路结果按 RRF 融合，返回前 top_k 条（score 字段会被替换成 RRF 分数）。"""
    scores: dict[str, float] = {}
    first_seen: dict[str, SearchResult] = {}

    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            chunk_id = result.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(chunk_id, result)

    ordered = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:top_k]
    fused: list[SearchResult] = []
    for chunk_id, score in ordered:
        fused.append(first_seen[chunk_id].model_copy(update={"score": round(score, 6)}))
    return fused


def merge_candidates(result_lists: list[list[SearchResult]]) -> list[SearchResult]:
    """把多路结果**只做去重合并**（不重新排序），交给下游精排。

    与 rrf_fuse 的分工：rrf_fuse 的职责是"产出排名"，所以必须打分；
    这里只要求"两路的候选进同一个池子、同一个块不重复"——排序交给
    CrossEncoder，因为它比 RRF 看得准。
    """
    seen: set[str] = set()
    merged: list[SearchResult] = []
    for results in result_lists:
        for result in results:
            if result.chunk_id in seen:
                continue
            seen.add(result.chunk_id)
            merged.append(result)
    return merged
