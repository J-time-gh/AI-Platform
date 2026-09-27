from fastapi.testclient import TestClient

from app.services.llm import LLMToolCall, LLMToolResponse


def tool_reply(
    call_id: str,
    name: str,
    arguments_json: str,
) -> LLMToolResponse:
    assistant_message = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "test reasoning",
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


def final_reply(content: str) -> LLMToolResponse:
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


def empty_reply() -> LLMToolResponse:
    assistant_message = {
        "role": "assistant",
        "content": None,
    }

    return LLMToolResponse(
        assistant_message=assistant_message,
        content=None,
        tool_calls=(),
        finish_reason="stop",
    )


def test_agent_calls_python_then_returns_final(
    client: TestClient,
    fake_llm,
    authenticated_headers: dict[str, str],
) -> None:
    fake_llm.tool_replies = [
        tool_reply(
            "call_python_1",
            "python_sandbox",
            '{"expression":"(12 + 8) / 2"}',
        ),
        final_reply("计算结果是 10"),
    ]

    response = client.post(
        "/agent",
        headers=authenticated_headers,
        json={"message": "请计算 (12 + 8) / 2"},
    )

    assert response.status_code == 200

    body = response.json()
    assert body["answer"] == "计算结果是 10"
    assert fake_llm.call_count == 2
    assert body["tool_calls"][0]["tool_name"] == "python_sandbox"
    assert body["tool_calls"][0]["arguments"] == {
        "expression": "(12 + 8) / 2",
    }
    assert body["tool_calls"][0]["ok"] is True
    assert body["tool_calls"][0]["output"]["value"] == 10.0

    # 第二轮模型请求包含原始调用和真实工具结果。
    assert fake_llm.last_messages[-2]["role"] == "assistant"
    assert fake_llm.last_messages[-2]["reasoning_content"] == "test reasoning"
    assert fake_llm.last_messages[-1]["role"] == "tool"
    assert fake_llm.last_messages[-1]["tool_call_id"] == "call_python_1"


def test_agent_rejects_empty_message(
    client: TestClient,
    authenticated_headers: dict[str, str],
) -> None:
    response = client.post(
        "/agent",
        headers=authenticated_headers,
        json={"message": ""},
    )

    assert response.status_code == 422


def test_agent_rejects_too_long_message(
    client: TestClient,
    authenticated_headers: dict[str, str],
) -> None:
    response = client.post(
        "/agent",
        headers=authenticated_headers,
        json={"message": "a" * 2001},
    )

    assert response.status_code == 422


def test_agent_returns_503_when_llm_unavailable(
    client: TestClient,
    fake_llm,
    authenticated_headers: dict[str, str],
) -> None:
    fake_llm.fail = True

    response = client.post(
        "/agent",
        headers=authenticated_headers,
        json={"message": "测试模型不可用"},
    )

    assert response.status_code == 503
    assert "模拟模型服务不可用" in response.json()["detail"]


def test_agent_returns_502_for_empty_llm_response(
    client: TestClient,
    fake_llm,
    authenticated_headers: dict[str, str],
) -> None:
    fake_llm.tool_replies = [empty_reply()]

    response = client.post(
        "/agent",
        headers=authenticated_headers,
        json={"message": "测试模型空响应"},
    )

    assert response.status_code == 502
    assert "模型返回格式错误" in response.json()["detail"]
