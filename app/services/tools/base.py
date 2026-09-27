from typing import Any, Protocol


class ToolError(Exception):
    """工具调用失败或被安全策略拒绝。"""


class ToolNotFound(ToolError):
    """找不到指定工具。"""


class Tool(Protocol):
    """所有 Agent 工具必须实现的协议。"""

    # 工具名称，供模型选择
    name: str
    # 工具用途说明，提供给 LLM
    description: str
    # 工具参数的 JSON Schema
    parameters: dict[str, Any]

    def run(self, arguments: dict[str, Any]) -> Any:
        """执行工具并返回可序列化结果。"""
        ...
