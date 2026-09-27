import json
from unittest.mock import AsyncMock

import httpx
import pytest

from app.services.llm import LmStudioClient


@pytest.mark.asyncio
async def test_chat_with_tools_parses_native_tool_call(monkeypatch):
    llm = LmStudioClient()

    response = httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "test reasoning",
                        "tool_calls": [
                            {
                                "id": "call_test_1",
                                "type": "function",
                                "function": {
                                    "name": "python_sandbox",
                                    "arguments": ('{"expression":"(12+8)/2"}'),
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                },
            ],
        },
        request=httpx.Request(
            "POST",
            "http://test/v1/chat/completions",
        ),
    )

    mock_post = AsyncMock(return_value=response)
    monkeypatch.setattr(llm._client, "post", mock_post)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "python_sandbox",
                "description": "计算数学表达式",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"},
                    },
                    "required": ["expression"],
                    "additionalProperties": False,
                },
            },
        },
    ]

    result = await llm.chat_with_tools(
        messages=[{"role": "user", "content": "计算 (12+8)/2"}],
        tools=tools,
        model="ornith-1.0-9b",
    )

    assert result.content is None
    assert result.finish_reason == "tool_calls"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "call_test_1"
    assert result.tool_calls[0].name == "python_sandbox"
    assert json.loads(result.tool_calls[0].arguments_json) == {
        "expression": "(12+8)/2",
    }

    assert result.assistant_message["reasoning_content"] == "test reasoning"
    assert result.assistant_message["tool_calls"][0]["id"] == "call_test_1"

    sent_payload = mock_post.await_args.kwargs["json"]
    assert sent_payload["tools"] == tools
    assert sent_payload["tool_choice"] == "auto"
    assert "response_format" not in sent_payload

    await llm.aclose()


@pytest.mark.asyncio
async def test_chat_with_tools_forwards_required_tool_choice(monkeypatch):
    llm = LmStudioClient()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "sql_query",
                "description": "执行预定义的只读数据库查询",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query_name": {
                            "type": "string",
                            "enum": ["document_count"],
                        },
                    },
                    "required": ["query_name"],
                    "additionalProperties": False,
                },
            },
        },
    ]

    response = httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_sql_1",
                                "type": "function",
                                "function": {
                                    "name": "sql_query",
                                    "arguments": ('{"query_name":"document_count"}'),
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                },
            ],
        },
        request=httpx.Request(
            "POST",
            "http://test/v1/chat/completions",
        ),
    )

    mock_post = AsyncMock(return_value=response)
    monkeypatch.setattr(llm._client, "post", mock_post)

    result = await llm.chat_with_tools(
        messages=[
            {
                "role": "user",
                "content": "查询文档数量",
            },
        ],
        tools=tools,
        model="ornith-1.0-9b",
        tool_choice="required",
    )

    assert result.finish_reason == "tool_calls"
    assert result.tool_calls[0].name == "sql_query"
    assert result.tool_calls[0].arguments_json == ('{"query_name":"document_count"}')

    sent_payload = mock_post.await_args.kwargs["json"]
    assert sent_payload["tools"] == tools
    assert sent_payload["tool_choice"] == "required"
    assert "response_format" not in sent_payload

    await llm.aclose()
