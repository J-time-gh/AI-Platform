import pytest

from app.services.tools.base import ToolError
from app.services.tools.python_sandbox import PythonSandboxTool


@pytest.fixture
def tool() -> PythonSandboxTool:
    return PythonSandboxTool()


def test_calculates_arithmetic(tool: PythonSandboxTool):
    result = tool.run(
        {
            "expression": "(12 + 8) / 2",
        },
    )

    assert result["value"] == 10


def test_calculates_math_function(tool: PythonSandboxTool):
    result = tool.run(
        {
            "expression": "sqrt(16)",
        },
    )

    assert result["value"] == 4


def test_supports_constants(tool: PythonSandboxTool):
    result = tool.run(
        {
            "expression": "pi * 2",
        },
    )

    assert result["value"] > 6.28
    assert result["value"] < 6.29


def test_rejects_empty_expression(tool: PythonSandboxTool):
    with pytest.raises(ToolError, match="不能为空"):
        tool.run(
            {
                "expression": "",
            },
        )


def test_rejects_import(tool: PythonSandboxTool):
    with pytest.raises(ToolError):
        tool.run(
            {
                "expression": "__import__('os')",
            },
        )


def test_rejects_file_access(tool: PythonSandboxTool):
    with pytest.raises(ToolError):
        tool.run(
            {
                "expression": "open('secret.txt')",
            },
        )


def test_rejects_system_call(tool: PythonSandboxTool):
    with pytest.raises(ToolError):
        tool.run(
            {
                "expression": "exec('print(1)')",
            },
        )


def test_rejects_large_exponent(tool: PythonSandboxTool):
    with pytest.raises(ToolError, match="指数过大"):
        tool.run(
            {
                "expression": "2 ** 100",
            },
        )


def test_rejects_division_by_zero(tool: PythonSandboxTool):
    with pytest.raises(ToolError):
        tool.run(
            {
                "expression": "1 / 0",
            },
        )
