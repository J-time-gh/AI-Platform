"""精排测试：用假实现，不加载模型、不连数据库。"""

from fastapi.testclient import TestClient

from app.schemas.search import SearchResult
from app.services.reranker import FakeReranker, rerank


class _FixedReranker:
    """按预设分数打分，用来精确控制排序结果。"""

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.passages: list[str] = []

    def score(self, query: str, passages: list[str]) -> list[float]:
        self.passages = list(passages)
        return self._scores


def _result(chunk_id: str, content: str = "") -> SearchResult:
    """造一条假的检索结果。"""
    return SearchResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        filename="fake.txt",
        chunk_index=0,
        content=content or f"内容 {chunk_id}",
        score=1.0,
    )


def test_rerank_reorders_by_score() -> None:
    results = [_result("a"), _result("b"), _result("c")]
    reranked = rerank("q", results, top_k=3, reranker=_FixedReranker([0.1, 0.9, 0.5]))
    assert [item.chunk_id for item in reranked] == ["b", "c", "a"]


def test_rerank_respects_top_k() -> None:
    results = [_result(str(index)) for index in range(5)]
    scores = [0.1, 0.2, 0.3, 0.4, 0.5]
    reranked = rerank("q", results, top_k=2, reranker=_FixedReranker(scores))
    assert [item.chunk_id for item in reranked] == ["4", "3"]


def test_rerank_applies_score_gate() -> None:
    results = [_result("a"), _result("b")]
    reranked = rerank("q", results, top_k=5, reranker=_FixedReranker([0.9, 0.1]), min_score=0.5)
    assert [item.chunk_id for item in reranked] == ["a"]


def test_rerank_gate_can_filter_everything() -> None:
    """全被门控滤掉 = 库里没有相关资料 → /chat 会据此拒答。"""
    results = [_result("a")]
    assert rerank("q", results, top_k=5, reranker=_FixedReranker([0.2]), min_score=0.5) == []


def test_rerank_replaces_score_but_keeps_content() -> None:
    results = [_result("a", content="原文一个字都不能丢")]
    reranked = rerank("q", results, top_k=1, reranker=_FixedReranker([0.77]))
    assert reranked[0].score == 0.77
    assert reranked[0].content == "原文一个字都不能丢"
    assert results[0].score == 1.0  # 原对象没被改动（model_copy 的价值）


def test_rerank_sends_content_to_reranker() -> None:
    reranker = _FixedReranker([0.5])
    rerank("问题", [_result("a", content="第一段")], top_k=1, reranker=reranker)
    assert reranker.passages == ["第一段"]


def test_rerank_handles_empty_input() -> None:
    assert rerank("q", [], top_k=5, reranker=_FixedReranker([])) == []


def test_fake_reranker_is_deterministic() -> None:
    reranker = FakeReranker()
    first = reranker.score("苹果", ["苹果很好吃", "香蕉"])
    assert first == reranker.score("苹果", ["苹果很好吃", "香蕉"])
    assert first[0] > first[1]


def test_hybrid_rerank_mode_is_accepted(
    client: TestClient,
    fake_reranker: FakeReranker,
    authenticated_headers: dict[str, str],
    upload_ready_document,
) -> None:
    """API 层接受 hybrid_rerank，并且真的调用精排。"""
    upload_ready_document(
        authenticated_headers,
        "rerank.txt",
        "苹果香蕉橘子",
    )

    search = client.post(
        "/search",
        headers=authenticated_headers,
        json={
            "query": "苹果",
            "top_k": 5,
            "mode": "hybrid_rerank",
        },
    )
    assert search.status_code == 200
    assert fake_reranker.call_count == 1
