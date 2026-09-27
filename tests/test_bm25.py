"""BM25 关键词检索测试：纯 CPU、无需 GPU，CI 可直接跑。"""

from fastapi.testclient import TestClient

from app.services.bm25 import tokenize


def _search(
    client: TestClient,
    headers: dict[str, str],
    query: str,
    mode: str = "bm25",
    top_k: int = 5,
) -> list[dict]:
    response = client.post(
        "/search",
        headers=headers,
        json={
            "query": query,
            "top_k": top_k,
            "mode": mode,
        },
    )
    assert response.status_code == 200

    return response.json()


def test_tokenize_splits_chinese_into_words() -> None:
    text = "科学实验在认识过程中的作用"
    tokens = tokenize(text)
    assert tokens
    assert len(tokens) < len(text)  # 按词切了，不是按字


def test_bm25_finds_chunk_by_keyword(
    client: TestClient,
    authenticated_headers: dict[str, str],
    upload_ready_document,
) -> None:
    upload_ready_document(
        authenticated_headers,
        "a.txt",
        "科学实验是认识过程中的重要环节。",
    )
    upload_ready_document(
        authenticated_headers,
        "b.txt",
        "商品的价值量由社会必要劳动时间决定。",
    )
    upload_ready_document(
        authenticated_headers,
        "c.txt",
        "资本的有机构成随着积累而提高。",
    )

    results = _search(
        client,
        authenticated_headers,
        "科学实验",
    )

    assert results
    assert results[0]["filename"] == "a.txt"


def test_bm25_returns_empty_for_unrelated_query(
    client: TestClient,
    authenticated_headers: dict[str, str],
    upload_ready_document,
) -> None:
    upload_ready_document(
        authenticated_headers,
        "a.txt",
        "科学实验是认识过程中的重要环节。",
    )

    assert _search(client, authenticated_headers, "zzzqqq") == []


def test_bm25_index_rebuilds_after_upload(
    client: TestClient,
    authenticated_headers: dict[str, str],
    upload_ready_document,
) -> None:
    upload_ready_document(
        authenticated_headers,
        "a.txt",
        "科学实验是认识过程中的重要环节。",
    )
    upload_ready_document(
        authenticated_headers,
        "b.txt",
        "商品的价值量由社会必要劳动时间决定。",
    )

    assert _search(client, authenticated_headers, "唯物论") == []

    upload_ready_document(
        authenticated_headers,
        "c.txt",
        "唯物论和唯心论的根本分歧。",
    )

    results = _search(
        client,
        authenticated_headers,
        "唯物论",
    )

    assert results
    assert results[0]["filename"] == "c.txt"


def test_vector_mode_remains_default(
    client: TestClient,
    authenticated_headers: dict[str, str],
    upload_ready_document,
) -> None:
    upload_ready_document(
        authenticated_headers,
        "a.txt",
        "苹果是一种水果。",
    )

    response = client.post(
        "/search",
        headers=authenticated_headers,
        json={"query": "苹果是一种水果。"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body
    assert body[0]["filename"] == "a.txt"


def test_search_rejects_unknown_mode(
    client: TestClient,
    authenticated_headers: dict[str, str],
) -> None:
    response = client.post(
        "/search",
        headers=authenticated_headers,
        json={
            "query": "x",
            "mode": "bm42",
        },
    )

    assert response.status_code == 422


def test_tokenize_filters_stopwords() -> None:
    tokens = tokenize("公司的年假有多少天？")
    assert "的" not in tokens
    assert "有" not in tokens
    assert "多少" not in tokens
