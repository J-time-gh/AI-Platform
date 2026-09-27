"""CrossEncoder 精排：把「问题 ↔ 候选块」拼在一起逐对打分，再重排取 top_k。

为什么需要它（阶段 4.2 的数据结论）：
    hybrid 两路 top-20 的并集召回 = 15/15 = 100%，说明**召回已经满分**，
    瓶颈在"从 40 条里挑 5 条"。而 RRF 只看名次不看内容，会因"双票必胜"把
    向量独有的正确答案挤出前 5。精排器逐条读内容，正好补这个缺口。

Bi-Encoder 与 CrossEncoder 的区别：
    Embedder（bge-m3）是 Bi-Encoder：query 和块**分别**编码成向量再算余弦，
    两者从不"见面"，所以能预先算好、检索极快，但精度有限。
    Reranker 是 CrossEncoder：把 (query, 块) **拼成一段**送进模型，能看到词级
    交互，精度高得多；代价是每对都要跑一次前向，没法预先计算——所以它只用于
    精排少量候选，不能拿来扫全库。

顺带白赚的一件事：精排分是"这问题和这块有多相关"的**绝对分数**，天生可比，
于是门控（判断库里到底有没有资料）也能用同一把尺子——顺手修掉 4.2 里
"BM25 腿没有可比阈值、导致拒答率掉到 60%"的毛病。
"""

import logging
import os
from functools import lru_cache
from typing import Annotated, Protocol

from fastapi import Depends

from app.core.config import settings
from app.schemas.search import SearchResult

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    """精排接口：给一批候选文本打分，分数越高越相关。"""

    def score(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """bge-reranker 实现：模型懒加载，跑在 GPU 上。"""

    def __init__(self) -> None:
        os.environ.setdefault("HF_ENDPOINT", settings.hf_endpoint)
        os.environ.setdefault("HF_HOME", settings.hf_home)
        # 注意：这里**不**加载模型。构造对象必须足够便宜——否则 vector / bm25
        # 模式也会因为依赖注入而被白白占掉几个 G 显存。
        self._encoder = None

    def _load(self):
        """第一次打分时才真正加载模型（首次约 10~60 秒，取决于是否已缓存）。"""
        if self._encoder is None:
            # 懒导入：CI 不装 torch 也能 import 本模块
            from sentence_transformers import CrossEncoder

            logger.info("加载精排模型 %s（首次较慢，请稍候）", settings.rerank_model)
            self._encoder = CrossEncoder(
                settings.rerank_model,
                max_length=settings.rerank_max_length,
            )
        return self._encoder

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        encoder = self._load()
        pairs = [(query, passage) for passage in passages]
        return [float(value) for value in encoder.predict(pairs)]


class FakeReranker:
    """测试替身：按「查询字的命中比例」打分，确定性、不加载模型。"""

    def __init__(self) -> None:
        self.call_count = 0
        self.last_passages: list[str] = []

    def score(self, query: str, passages: list[str]) -> list[float]:
        self.call_count += 1
        self.last_passages = list(passages)
        chars = {char for char in query if char.strip()}
        if not chars:
            return [0.0 for _ in passages]
        return [sum(1 for char in chars if char in passage) / len(chars) for passage in passages]


def rerank(
    query: str,
    results: list[SearchResult],
    top_k: int,
    reranker: Reranker,
    min_score: float = 0.0,
) -> list[SearchResult]:
    """用精排分数重排候选块，返回前 top_k 条（score 字段被替换成精排分）。

    min_score > 0 时做门控：低于阈值的块直接丢掉；全丢掉就等于"库里没有相关
    资料"，/chat 会据此拒答——这就是 4.2 里缺的那把统一尺子。
    """
    if not results:
        return []

    scores = reranker.score(query, [item.content for item in results])
    scored = [
        (item, float(score))
        for item, score in zip(results, scores, strict=True)
        if score >= min_score
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [item.model_copy(update={"score": round(score, 6)}) for item, score in scored[:top_k]]


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    """FastAPI 依赖：进程内单例（构造很便宜，模型在首次打分时才加载）。"""
    return CrossEncoderReranker()


RerankerDep = Annotated[Reranker, Depends(get_reranker)]
