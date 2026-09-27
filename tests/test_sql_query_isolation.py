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
    """注册并登录一个独立测试用户，返回该用户的 JWT Header。"""
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


def test_agent_sql_query_can_read_own_chunks(
    client: TestClient,
    fake_llm,
    upload_ready_document,
) -> None:
    """用户 A 的 Agent 可以读取用户 A 自己文档的 Chunk。"""
    user_a_headers = _auth_headers(
        client,
        "sql-own-user@example.com",
    )

    private_text = "用户A自己的SQL资料"
    document_id = upload_ready_document(
        user_a_headers,
        "own-document.txt",
        private_text,
    )

    fake_llm.tool_replies = [
        _tool_reply(
            "call_sql_own_1",
            "sql_query",
            {
                "query_name": "chunks_for_document",
                "document_id": document_id,
                "limit": 20,
            },
        ),
        _final_reply("已读取自己的文档"),
    ]

    response = client.post(
        "/agent",
        headers=user_a_headers,
        json={
            "message": "读取我的文档文本块",
        },
    )
    assert response.status_code == 200, response.text

    body = response.json()
    tool_call = body["tool_calls"][0]
    output = tool_call["output"]

    assert tool_call["tool_name"] == "sql_query"
    assert tool_call["ok"] is True
    assert output["count"] > 0
    assert any(private_text in row["content"] for row in output["rows"])


def test_agent_sql_query_cannot_read_another_users_chunks(
    client: TestClient,
    fake_llm,
    upload_ready_document,
) -> None:
    """用户 B 的 Agent 不能读取用户 A 文档的 Chunk。"""
    user_a_headers = _auth_headers(
        client,
        "sql-user-a@example.com",
    )
    user_b_headers = _auth_headers(
        client,
        "sql-user-b@example.com",
    )

    document_id = upload_ready_document(
        user_a_headers,
        "private.txt",
        "用户A的SQL私有资料",
    )

    fake_llm.tool_replies = [
        _tool_reply(
            "call_sql_other_1",
            "sql_query",
            {
                "query_name": "chunks_for_document",
                "document_id": document_id,
                "limit": 20,
            },
        ),
        _final_reply("没有权限读取该文档"),
    ]

    response = client.post(
        "/agent",
        headers=user_b_headers,
        json={
            "message": "读取这份文档的文本块",
        },
    )
    assert response.status_code == 200, response.text

    body = response.json()
    tool_call = body["tool_calls"][0]
    output = tool_call["output"]

    assert tool_call["tool_name"] == "sql_query"
    assert tool_call["ok"] is True
    assert output["rows"] == []
    assert output["count"] == 0
