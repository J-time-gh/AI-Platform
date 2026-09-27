"""LLM 客户端：OpenAI 兼容接口（LM Studio / vLLM / 云端 API）。"""

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated, Any, Literal, Protocol

import httpx
from fastapi import Depends

from app.core.config import settings


class LLMUnavailable(Exception):
    """模型服务不可用：连接失败、超时、模型不存在等。"""


# 模型提出的一次工具调用
@dataclass(frozen=True)
class LLMToolCall:
    id: str
    # 工具名
    name: str
    # 模型给工具的 JSON 参数。
    arguments_json: str


# OpenAI 兼容接口返回的信息转为项目内部类型
@dataclass(frozen=True)
class LLMToolResponse:
    # 保留模型原始消息
    assistant_message: dict[str, Any]
    # 普通文本
    content: str | None
    tool_calls: tuple[LLMToolCall, ...]
    # 示模型为什么结束这轮生成
    finish_reason: str | None


class LLMClient(Protocol):
    """LLM 客户端接口。"""

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str: ...

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str | None = None,
        tool_choice: Literal["none", "auto", "required"] = "auto",
    ) -> LLMToolResponse: ...


class LmStudioClient:
    """OpenAI 兼容客户端：支持 LM Studio、vLLM 和云端 API。"""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=settings.llm_base_url,
            headers={
                "Authorization": f"Bearer {settings.llm_api_key}",
            },
            timeout=settings.llm_timeout,
        )

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        payload = {
            "model": model or settings.llm_model,
            "messages": messages,
            "temperature": settings.llm_temperature,
        }

        if response_format is not None:
            payload["response_format"] = response_format

        try:
            response = await self._client.post(
                "/chat/completions",
                json=payload,
            )
            response.raise_for_status()

        except httpx.HTTPStatusError as exc:
            raise LLMUnavailable(
                f"模型服务返回 HTTP {exc.response.status_code}："
                f"请检查 LLM_MODEL={payload['model']} 是否正确",
            ) from exc

        except httpx.HTTPError as exc:
            raise LLMUnavailable(
                f"无法连接模型服务 {settings.llm_base_url}：{exc}",
            ) from exc

        try:
            return response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMUnavailable(
                "模型返回格式不符合 OpenAI Chat Completions 格式",
            ) from exc

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str | None = None,
        tool_choice: Literal["none", "auto", "required"] = "auto",
    ) -> LLMToolResponse:
        payload = {
            "model": model or settings.llm_model,
            "messages": messages,
            # 使用当前配置
            "temperature": settings.llm_temperature,
            # 本轮可用工具的 JSON Schema
            "tools": tools,
            # 告诉模型是否必须、可以或不能调用工具
            "tool_choice": tool_choice,
        }

        try:
            response = await self._client.post(
                "/chat/completions",
                json=payload,
            )
            response.raise_for_status()

        except httpx.HTTPStatusError as exc:
            raise LLMUnavailable(
                f"模型服务返回 HTTP {exc.response.status_code}："
                f"请检查 LLM_MODEL={payload['model']} 是否正确",
            ) from exc

        except httpx.HTTPError as exc:
            raise LLMUnavailable(
                f"无法连接模型服务 {settings.llm_base_url}：{exc}",
            ) from exc

        try:
            data = response.json()
            choice = data["choices"][0]
            message = choice["message"]
            raw_tool_calls = message.get("tool_calls") or []

            if not isinstance(raw_tool_calls, list):
                raise TypeError("tool_calls 必须是列表")

            parsed_tool_calls: list[LLMToolCall] = []

            for raw_call in raw_tool_calls:
                function = raw_call["function"]
                arguments = function["arguments"]

                if isinstance(arguments, dict):
                    arguments = json.dumps(arguments, ensure_ascii=False)

                call_id = raw_call["id"]
                name = function["name"]

                if not isinstance(call_id, str) or not isinstance(name, str):
                    raise TypeError(
                        "tool call id 和 function name 必须是字符串",
                    )

                if not isinstance(arguments, str):
                    raise TypeError(
                        "function arguments 必须是 JSON 字符串",
                    )

                parsed_tool_calls.append(
                    LLMToolCall(
                        id=call_id,
                        name=name,
                        arguments_json=arguments,
                    ),
                )

            content = message.get("content")
            if content is not None and not isinstance(content, str):
                raise TypeError("assistant content 必须是字符串或 null")

            finish_reason = choice.get("finish_reason")
            if finish_reason is not None and not isinstance(finish_reason, str):
                raise TypeError("finish_reason 必须是字符串或 null")

            return LLMToolResponse(
                assistant_message=dict(message),
                content=content,
                tool_calls=tuple(parsed_tool_calls),
                finish_reason=finish_reason,
            )

        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMUnavailable(
                "模型返回的原生工具调用格式无效",
            ) from exc

    async def aclose(self) -> None:
        """关闭 HTTP 连接池。"""
        await self._client.aclose()


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    """FastAPI 依赖：进程内单例。"""
    return LmStudioClient()


LLMClientDep = Annotated[LLMClient, Depends(get_llm_client)]
