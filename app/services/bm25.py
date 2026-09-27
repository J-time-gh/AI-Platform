"""BM25 关键词检索：jieba 分词 + rank_bm25，补齐向量检索的字面匹配短板。

注意：不同 mode 下 SearchResult.score 的量纲不同——
向量模式是余弦相似度（0~1），BM25 模式是词频分数（任意正数）。
两者不可直接比较，也**不能相加**（融合用 RRF，只看排名）。
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache

import jieba
from rank_bm25 import BM25Okapi
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models.chunk import Chunk
from app.db.models.document import Document
from app.schemas.search import SearchResult

jieba.setLogLevel(logging.WARNING)  # 关掉 "Building prefix dict" 之类的噪音日志

# 中文停用词 / 疑问词：它们几乎出现在每个块里，只会制造"假命中"
# （"公司的年假有多少天？"里的"的/有/多少/天"就是典型）
STOPWORDS = {
    "的",
    "了",
    "是",
    "有",
    "和",
    "在",
    "就",
    "都",
    "而",
    "及",
    "与",
    "或",
    "对",
    "为",
    "以",
    "于",
    "其",
    "之",
    "也",
    "很",
    "把",
    "被",
    "这",
    "那",
    "一个",
    "我们",
    "他们",
    "什么",
    "怎么",
    "怎样",
    "多少",
    "哪些",
    "哪个",
    "为什么",
    "如何",
    "是否",
    "可以",
    "应该",
    "吗",
    "呢",
    "吧",
    "啊",
    "天",
    "用",
    "？",
    "，",
    "。",
    "、",
    "；",
    "：",
    "！",
}


def tokenize(text: str) -> list[str]:
    """中文分词：BM25 比的是"词"；同时滤掉停用词，避免"的/是/多少"制造假命中。"""
    return [token for token in jieba.lcut(text) if token.strip() and token not in STOPWORDS]


@dataclass
class IndexedChunk:
    """索引里保存的一行——检索时不需要再回数据库。"""

    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    content: str
    tokens: set[str]  # 预分词结果，用于判断"是否真的含查询词"


class Bm25Index:
    """进程内的 BM25 索引；数据库内容变化时自动重建。"""

    def __init__(self, user_id: str) -> None:
        self._user_id = user_id
        self._version: tuple[int, datetime | None] | None = None
        self._rows: list[IndexedChunk] = []
        self._bm25: BM25Okapi | None = None

    def _read_version(
        self,
        db: Session,
    ) -> tuple[int, datetime | None]:
        """读取指定用户的索引版本信息。"""
        total, latest = db.execute(
            select(
                func.count(Chunk.id),
                func.max(Chunk.created_at),
            )
            .join(Document, Chunk.document_id == Document.id)
            .where(
                Document.user_id == self._user_id,
                Document.status == "ready",
            )
        ).one()

        return int(total), latest

    def _rebuild(self, db: Session) -> None:
        """只重建当前用户的 BM25 索引。"""
        rows = db.execute(
            select(Chunk, Document.filename)
            .join(Document, Chunk.document_id == Document.id)
            .where(
                Document.user_id == self._user_id,
                Document.status == "ready",
            )
        ).all()

        self._rows = []
        corpus: list[list[str]] = []

        for chunk, filename in rows:
            tokens = tokenize(chunk.content)
            corpus.append(tokens)
            self._rows.append(
                IndexedChunk(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    filename=filename,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    tokens=set(tokens),
                ),
            )

        self._bm25 = BM25Okapi(corpus) if corpus else None
        self._version = self._read_version(db)

    def search(self, db: Session, query: str, top_k: int) -> list[SearchResult]:
        """按 BM25 分数降序返回 top_k；与查询词无交集的块会被过滤掉。

        刻意不用 "score > 0" 过滤：rank_bm25 的 IDF 在"某词出现在恰好一半文档里"时
        会算出 0，在"某词很常见"时 eps 还可能为负——两种情况都会把真正含查询词的块
        误过滤掉。所以改用分词交集判断，分数只负责排序。
        """
        if self._read_version(db) != self._version:
            self._rebuild(db)
        if self._bm25 is None:
            return []

        query_tokens = tokenize(query)
        query_token_set = set(query_tokens)
        scores = self._bm25.get_scores(query_tokens)
        candidates = [
            (row, float(score))
            for row, score in zip(self._rows, scores, strict=True)
            if row.tokens & query_token_set
        ]
        candidates.sort(key=lambda pair: pair[1], reverse=True)
        return [
            SearchResult(
                chunk_id=row.chunk_id,
                document_id=row.document_id,
                filename=row.filename,
                chunk_index=row.chunk_index,
                content=row.content,
                score=round(score, 4),
            )
            for row, score in candidates[:top_k]
        ]


@lru_cache(maxsize=128)
def get_bm25_index(user_id: str) -> Bm25Index:
    """进程内单例：只保留一份索引。"""
    return Bm25Index(user_id)
