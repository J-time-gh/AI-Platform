import json
from typing import Any

import pytest

from app.services.agent import AgentProtocolError, AgentService
from app.services.llm import LLMToolCall, LLMToolResponse
from app.services.skill_registry import KNOWLEDGE_BASE_QA
from app.services.tools.base import ToolError
from app.services.tools.registry import ToolRegistry


class ScriptedLLM:
    def __init__(self, replies: list[LLMToolResponse]) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str | None = None,
        tool_choice: str = "auto",
    ) -> LLMToolResponse:
        self.calls.append(
            {
                "messages": list(messages),
                "tools": tools,
                "model": model,
                "tool_choice": tool_choice,
            },
        )
        return self.replies.pop(0)


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


class EchoTool:
    name = "echo"
    description = "返回输入内容"
    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
        },
        "required": ["text"],
    }

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return {"text": arguments["text"]}


class RagStubTool:
    name = "rag_search"
    description = "测试用 RAG 检索"
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return {
            "results": [
                {
                    "filename": "policy.txt",
                    "content": "测试知识库结果",
                    "chunk_index": 0,
                },
            ],
            "count": 1,
        }


class SpyTool:
    name = "spy_tool"
    description = "用于确认未授权工具不会执行"
    parameters = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        self.called = False

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self.called = True
        return {"called": True}


class FailingTool:
    name = "failing_tool"
    description = "始终失败"
    parameters = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raise ToolError("模拟工具失败")


@pytest.mark.asyncio
async def test_agent_calls_tool_then_returns_final_answer() -> None:
    llm = ScriptedLLM(
        [
            tool_reply(
                "call_echo_1",
                "echo",
                '{"text":"hello"}',
            ),
            final_reply("工具返回了 hello"),
        ],
    )

    registry = ToolRegistry()
    registry.register(EchoTool())

    service = AgentService(
        llm=llm,
        registry=registry,
        max_steps=4,
    )

    result = await service.run("测试工具")

    assert result["answer"] == "工具返回了 hello"
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["tool_name"] == "echo"
    assert result["tool_calls"][0]["arguments"] == {"text": "hello"}
    assert result["tool_calls"][0]["ok"] is True
    assert result["tool_calls"][0]["output"] == {"text": "hello"}
    assert llm.calls[0]["tool_choice"] == "auto"
    assert [item["function"]["name"] for item in llm.calls[0]["tools"]] == ["echo"]

    # 下一轮模型请求应包含原始 assistant 调用和对应的工具结果。
    messages = llm.calls[1]["messages"]
    assert messages[-2]["role"] == "assistant"
    assert messages[-2]["reasoning_content"] == "test reasoning"
    assert messages[-1]["role"] == "tool"
    assert messages[-1]["tool_call_id"] == "call_echo_1"
    assert json.loads(messages[-1]["content"]) == {
        "ok": True,
        "output": {"text": "hello"},
    }


@pytest.mark.asyncio
async def test_agent_returns_final_answer_without_calling_tool() -> None:
    llm = ScriptedLLM([final_reply("你好")])
    registry = ToolRegistry()

    service = AgentService(llm=llm, registry=registry)
    result = await service.run("打个招呼")

    assert result["answer"] == "你好"
    assert result["tool_calls"] == []
    assert result["steps"][0]["action"] == "final"
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_agent_records_unknown_tool_error() -> None:
    llm = ScriptedLLM(
        [
            tool_reply(
                "call_unknown_1",
                "unknown_tool",
                "{}",
            ),
            final_reply("该工具不可用"),
        ],
    )
    registry = ToolRegistry()

    service = AgentService(
        llm=llm,
        registry=registry,
        max_steps=4,
    )
    result = await service.run("测试未知工具")

    assert result["tool_calls"][0]["tool_name"] == "unknown_tool"
    assert result["tool_calls"][0]["ok"] is False
    assert "未知工具" in result["tool_calls"][0]["error"]

    tool_message = llm.calls[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_unknown_1"
    assert json.loads(tool_message["content"])["ok"] is False


@pytest.mark.asyncio
async def test_agent_rejects_invalid_tool_arguments_json() -> None:
    llm = ScriptedLLM(
        [
            tool_reply(
                "call_invalid_1",
                "echo",
                "这不是 JSON",
            ),
        ],
    )

    registry = ToolRegistry()
    registry.register(EchoTool())

    service = AgentService(llm=llm, registry=registry)

    with pytest.raises(AgentProtocolError, match="不是合法 JSON"):
        await service.run("测试非法工具参数")


@pytest.mark.asyncio
async def test_agent_rejects_empty_model_response() -> None:
    llm = ScriptedLLM([empty_reply()])
    registry = ToolRegistry()

    service = AgentService(llm=llm, registry=registry)

    with pytest.raises(AgentProtocolError, match="未返回最终回答"):
        await service.run("测试空回答")


@pytest.mark.asyncio
async def test_agent_stops_at_max_tool_calls() -> None:
    llm = ScriptedLLM(
        [
            tool_reply(
                "call_echo_1",
                "echo",
                '{"text":"again"}',
            ),
        ],
    )

    registry = ToolRegistry()
    registry.register(EchoTool())

    service = AgentService(
        llm=llm,
        registry=registry,
        max_steps=1,
    )
    result = await service.run("测试最大工具调用次数")

    assert len(result["tool_calls"]) == 1
    assert result["answer"] == "Agent 未能在限制步数内完成任务。"
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_agent_records_tool_error_and_continues() -> None:
    llm = ScriptedLLM(
        [
            tool_reply(
                "call_failing_1",
                "failing_tool",
                "{}",
            ),
            final_reply("工具执行失败"),
        ],
    )

    registry = ToolRegistry()
    registry.register(FailingTool())

    service = AgentService(
        llm=llm,
        registry=registry,
        max_steps=4,
    )
    result = await service.run("测试工具失败")

    assert result["answer"] == "工具执行失败"
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["tool_name"] == "failing_tool"
    assert result["tool_calls"][0]["ok"] is False
    assert result["tool_calls"][0]["error"] == "模拟工具失败"

    tool_message = llm.calls[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert json.loads(tool_message["content"]) == {
        "ok": False,
        "error": "模拟工具失败",
    }


@pytest.mark.asyncio
async def test_agent_injects_skill_and_filters_tools() -> None:
    llm = ScriptedLLM(
        [
            tool_reply(
                "call_rag_1",
                "rag_search",
                '{"query":"报销制度"}',
            ),
            final_reply("根据检索到的资料回答"),
        ],
    )

    registry = ToolRegistry()
    registry.register(RagStubTool())
    registry.register(SpyTool())

    service = AgentService(
        llm=llm,
        registry=registry,
        skill=KNOWLEDGE_BASE_QA,
    )
    result = await service.run("请查找报销制度")

    assert result["answer"] == "根据检索到的资料回答"
    first_call = llm.calls[0]
    assert first_call["tool_choice"] == "required"
    assert [item["function"]["name"] for item in first_call["tools"]] == ["rag_search"]
    assert KNOWLEDGE_BASE_QA.instructions in first_call["messages"][0]["content"]
    assert llm.calls[1]["tool_choice"] == "auto"
    assert result["tool_calls"][0]["tool_name"] == "rag_search"
    assert result["tool_calls"][0]["ok"] is True


@pytest.mark.asyncio
async def test_required_skill_rejects_final_answer_without_tool_call() -> None:
    llm = ScriptedLLM([final_reply("我直接回答，不检索")])
    registry = ToolRegistry()
    registry.register(RagStubTool())

    service = AgentService(
        llm=llm,
        registry=registry,
        skill=KNOWLEDGE_BASE_QA,
    )

    with pytest.raises(AgentProtocolError, match="要求调用工具"):
        await service.run("请检索知识库")


@pytest.mark.asyncio
async def test_skill_blocks_unlisted_tool_before_execution() -> None:
    llm = ScriptedLLM(
        [
            tool_reply(
                "call_spy_1",
                "spy_tool",
                "{}",
            ),
        ],
    )
    registry = ToolRegistry()
    registry.register(RagStubTool())
    spy_tool = SpyTool()
    registry.register(spy_tool)

    service = AgentService(
        llm=llm,
        registry=registry,
        skill=KNOWLEDGE_BASE_QA,
    )

    with pytest.raises(AgentProtocolError, match="不允许调用工具"):
        await service.run("请检索知识库")

    assert spy_tool.called is False


@pytest.mark.asyncio
async def test_skill_rejects_unregistered_allowed_tool() -> None:
    llm = ScriptedLLM([])
    registry = ToolRegistry()
    service = AgentService(
        llm=llm,
        registry=registry,
        skill=KNOWLEDGE_BASE_QA,
    )

    with pytest.raises(AgentProtocolError, match="未注册的工具"):
        await service.run("请检索知识库")

    assert llm.calls == []


@pytest.mark.asyncio
async def test_agent_includes_prior_history_before_current_question() -> None:
    llm = ScriptedLLM([final_reply("基于历史的回答")])
    registry = ToolRegistry()
    service = AgentService(
        llm=llm,
        registry=registry,
    )

    await service.run(
        question="当前问题",
        history=[
            {"role": "user", "content": "之前的问题"},
            {"role": "assistant", "content": "之前的回答"},
        ],
    )

    messages = llm.calls[0]["messages"]
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert messages[1]["content"] == "之前的问题"
    assert messages[2]["content"] == "之前的回答"
    assert messages[3]["content"] == "当前问题"
