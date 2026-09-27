from fastapi.testclient import TestClient

from app.services.highlight import matched_query_terms


def test_matched_query_terms_uses_real_chinese_word() -> None:
    terms = matched_query_terms(
        "什么是资本论？",
        "《资本论》讨论商品、货币和资本的关系。",
    )

    assert terms == ["资本论"]


def test_matched_query_terms_excludes_question_words_and_non_matches() -> None:
    terms = matched_query_terms(
        "年薪是多少？",
        "员工连续工作满一年后享有带薪年假。",
    )

    assert terms == []


def test_search_returns_backend_highlight_terms(
    client: TestClient,
    authenticated_headers: dict[str, str],
    upload_ready_document,
) -> None:
    upload_ready_document(
        authenticated_headers,
        "capital.txt",
        "《资本论》讨论商品、货币和资本的关系。",
    )

    response = client.post(
        "/search",
        headers=authenticated_headers,
        json={
            "query": "什么是资本论？",
            "top_k": 5,
            "mode": "bm25",
        },
    )

    assert response.status_code == 200
    assert response.json()[0]["highlight_terms"] == ["资本论"]
