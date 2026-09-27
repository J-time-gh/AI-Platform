import json
from typing import Any

from fastapi.testclient import TestClient

from app.services.llm import LLMToolCall, LLMToolResponse


def _tool_reply(
    call_id: str,
    name: str,
    arguments: dict[str, Any],
) -> LLMToolResponse:
    arguments_json = json.dumps(arguments, ensure_ascii=False)
    assistant_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments_json,
                },
            },
        ],
    }

    return LLMToolResponse(
        assistant_message=assistant_message,
        content=None,
        tool_calls=(
            LLMToolCall(
                id=call_id,
                name=name,
                arguments_json=arguments_json,
            ),
        ),
        finish_reason="tool_calls",
    )


def _final_reply(content: str) -> LLMToolResponse:
    assistant_message = {
        "role": "assistant",
        "content": content,
    }

    return LLMToolResponse(
        assistant_message=assistant_message,
        content=content,
        tool_calls=(),
        finish_reason="stop",
    )


def _auth_headers(
    client: TestClient,
    email: str,
) -> dict[str, str]:
    """注册并登录一个测试用户，返回 Bearer Token Header。"""
    password = "correct-horse-battery-staple"
    register_response = client.post(
        "/auth/register",
        json={
            "email": email,
            "password": password,
        },
    )
    assert register_response.status_code == 201

    login_response = client.post(
        "/auth/login",
        json={
            "email": email,
            "password": password,
        },
    )
    assert login_response.status_code == 200

    return {
        "Authorization": f"Bearer {login_response.json()['access_token']}",
    }


def _create_rag_session(
    client: TestClient,
    headers: dict[str, str],
) -> str:
    create_response = client.post(
        "/chat/sessions",
        headers=headers,
        json={"title": "RAG 隔离测试"},
    )
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]

    skill_response = client.patch(
        f"/chat/sessions/{session_id}/skill",
        headers=headers,
        json={"active_skill_id": "knowledge_base_qa"},
    )
    assert skill_response.status_code == 200
    return session_id


def _search(
    client: TestClient,
    headers: dict[str, str],
    query: str,
    mode: str = "vector",
) -> list[dict]:
    """以指定用户身份调用 Search。"""
    response = client.post(
        "/search",
        headers=headers,
        json={
            "query": query,
            "top_k": 5,
            "mode": mode,
        },
    )
    assert response.status_code == 200

    return response.json()


def test_vector_search_isolated_by_user(
    client: TestClient,
    upload_ready_document,
) -> None:
    """用户 B 不能通过向量检索读取用户 A 的私有资料。"""
    user_a_headers = _auth_headers(client, "vector-user-a@example.com")
    user_b_headers = _auth_headers(client, "vector-user-b@example.com")

    private_text = "用户A独有的向量检索机密关键词"
    upload_ready_document(
        user_a_headers,
        "vector-private.txt",
        private_text,
    )

    user_a_results = _search(
        client,
        user_a_headers,
        private_text,
    )
    assert user_a_results
    assert user_a_results[0]["content"] == private_text

    user_b_results = _search(
        client,
        user_b_headers,
        private_text,
    )
    assert user_b_results == []


def test_bm25_search_isolated_by_user(
    client: TestClient,
    upload_ready_document,
) -> None:
    """用户 B 不能通过 BM25 索引读取用户 A 的私有资料。"""
    user_a_headers = _auth_headers(client, "bm25-user-a@example.com")
    user_b_headers = _auth_headers(client, "bm25-user-b@example.com")

    private_text = "用户A独有的BM25关键词量子报销"
    upload_ready_document(
        user_a_headers,
        "bm25-private.txt",
        private_text,
    )

    user_a_results = _search(
        client,
        user_a_headers,
        "量子报销",
        mode="bm25",
    )
    assert user_a_results
    assert user_a_results[0]["content"] == private_text

    user_b_results = _search(
        client,
        user_b_headers,
        "量子报销",
        mode="bm25",
    )
    assert user_b_results == []


def test_chat_rag_skill_does_not_use_another_users_document(
    client: TestClient,
    fake_llm,
    upload_ready_document,
) -> None:
    """用户 B 的 RAG Skill 不得检索或返回用户 A 的资料。"""
    user_a_headers = _auth_headers(client, "chat-user-a@example.com")
    user_b_headers = _auth_headers(client, "chat-user-b@example.com")

    private_text = "用户A私有的Chat机密资料"
    upload_ready_document(
        user_a_headers,
        "chat-private.txt",
        private_text,
    )
    session_id = _create_rag_session(client, user_b_headers)

    fake_llm.tool_replies = [
        _tool_reply(
            "call_chat_rag_1",
            "rag_search",
            {"query": private_text, "top_k": 5},
        ),
        _final_reply("没有检索到资料"),
    ]

    response = client.post(
        "/chat",
        headers=user_b_headers,
        json={
            "message": private_text,
            "session_id": session_id,
        },
    )

    assert response.status_code == 200

    body = response.json()
    assert body["reply"] == "没有检索到资料"
    assert body["sources"] == []
    assert fake_llm.call_count == 2
    tool_result = json.loads(fake_llm.last_messages[-1]["content"])
    assert tool_result["output"]["count"] == 0
    assert private_text not in fake_llm.last_messages[-1]["content"]


def test_agent_rag_search_isolated_by_user(
    client: TestClient,
    fake_llm,
    upload_ready_document,
) -> None:
    """用户 B 的 Agent rag_search 不得读取用户 A 的资料。"""
    user_a_headers = _auth_headers(client, "agent-user-a@example.com")
    user_b_headers = _auth_headers(client, "agent-user-b@example.com")

    private_text = "用户A私有的Agent检索资料"
    upload_ready_document(
        user_a_headers,
        "agent-private.txt",
        private_text,
    )

    fake_llm.tool_replies = [
        _tool_reply(
            "call_rag_1",
            "rag_search",
            {
                "query": private_text,
                "top_k": 5,
            },
        ),
        _final_reply("没有检索到资料"),
    ]

    response = client.post(
        "/agent",
        headers=user_b_headers,
        json={
            "message": f"请检索：{private_text}",
        },
    )

    assert response.status_code == 200

    body = response.json()
    assert body["tool_calls"][0]["tool_name"] == "rag_search"
    assert body["tool_calls"][0]["ok"] is True
    assert body["tool_calls"][0]["output"]["count"] == 0
