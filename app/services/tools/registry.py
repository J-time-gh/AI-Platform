from typing import Any

from app.services.tools.base import Tool, ToolNotFound


class ToolRegistry:
    """Agent 工具注册表。"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"工具已注册：{tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFound(f"未知工具：{name}") from exc

    def definitions(self) -> list[dict[str, Any]]:
        """提供给 LLM 的工具说明。"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in self._tools.values()
        ]

    # 逐个读取项目已注册工具的名称、说明和参数 Schema，包装为模型 API 需要的 type=function 结构
    def openai_tools(self) -> list[dict[str, Any]]:
        """生成 OpenAI-compatible Function Calling 工具定义。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]
