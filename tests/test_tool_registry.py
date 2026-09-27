from typing import Any

from app.services.tools.registry import ToolRegistry


class ExampleTool:
    name = "echo"
    description = "返回输入内容"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
        },
        "required": ["text"],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return {"text": arguments["text"]}


def test_openai_tools_returns_function_call_schema() -> None:
    registry = ToolRegistry()
    tool = ExampleTool()
    registry.register(tool)

    assert registry.openai_tools() == [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "返回输入内容",
                "parameters": tool.parameters,
            },
        },
    ]


def test_legacy_definitions_format_is_unchanged() -> None:
    registry = ToolRegistry()
    tool = ExampleTool()
    registry.register(tool)

    assert registry.definitions() == [
        {
            "name": "echo",
            "description": "返回输入内容",
            "parameters": tool.parameters,
        },
    ]
