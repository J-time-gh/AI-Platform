import pytest

from app.services import retrieval
from app.services.tools.rag_search import RagSearchTool


@pytest.mark.parametrize(
    ("arguments", "expected_top_k"),
    [
        ({"query": "关键词"}, 2),
        ({"query": "关键词", "top_k": 4}, 4),
    ],
)
def test_rag_tool_uses_configured_mode_and_top_k(
    monkeypatch,
    arguments: dict[str, object],
    expected_top_k: int,
) -> None:
    captured: dict[str, object] = {}

    def fake_search_with_mode(**kwargs: object) -> list[object]:
        captured.update(kwargs)
        return []

    monkeypatch.setattr(retrieval, "search_with_mode", fake_search_with_mode)

    tool = RagSearchTool(
        db=object(),
        embedder=object(),
        reranker=object(),
        user_id="test-user",
        mode="bm25",
        default_top_k=2,
    )
    result = tool.run(arguments)

    assert result == {"results": [], "count": 0}
    assert captured["mode"] == "bm25"
    assert captured["top_k"] == expected_top_k
    assert captured["user_id"] == "test-user"
