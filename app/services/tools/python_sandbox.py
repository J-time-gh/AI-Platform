import ast
import math
import operator
from typing import Any

from app.core.config import settings
from app.services.tools.base import ToolError

# 定义允许的二元运算
_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
# 定义允许的函数和常量
_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "min": min,
    "max": max,
}
# 定义允许的常量
_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}


# Agent 使用的工具对象
class PythonSandboxTool:
    """安全执行数学表达式的工具。"""

    name = "python_sandbox"
    description = (
        "执行受限数学表达式，适用于计算、统计和简单数值问题。"
        "不支持文件、网络、系统命令或任意 Python 代码。"
    )
    # 定义工具参数的 JSON Schema
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": ("数学表达式，例如 '(12 + 8) / 2' 或 'sqrt(16)'"),
            },
        },
        "required": ["expression"],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """校验并执行安全数学表达式。"""
        # 读取表达式
        expression = arguments.get("expression")

        if not isinstance(expression, str):
            raise ToolError("expression 必须是字符串")

        expression = expression.strip()

        if not expression:
            raise ToolError("expression 不能为空")

        if len(expression) > settings.python_max_expression_length:
            raise ToolError(
                "expression 超过最大长度限制",
            )
        # 把表达式解析成 AST
        try:
            # ast.parse() 不会直接执行表达式，而是先把表达式解析成语法树。
            # Expression
            # └── BinOp /
            # ├── BinOp +
            # │   ├── Constant 12
            # │   └── Constant 8
            # └── Constant 2
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ToolError(
                f"表达式语法错误：{exc.msg}",
            ) from exc
        # 检查 AST 节点数量，防止过于复杂的表达式
        node_count = sum(1 for _ in ast.walk(tree))

        if node_count > settings.python_max_ast_nodes:
            raise ToolError(
                "表达式过于复杂",
            )

        try:
            value = self._evaluate(tree.body)
        except ToolError:
            raise
        except (ArithmeticError, ValueError, OverflowError) as exc:
            raise ToolError(
                f"表达式执行失败：{exc}",
            ) from exc

        return {
            "expression": expression,
            "value": value,
        }
        # or

    def _evaluate(self, node: ast.AST) -> int | float:
        """递归计算经过白名单验证的 AST 节点。"""
        # 处理数字常量
        if isinstance(node, ast.Constant):
            value = node.value

            if isinstance(value, bool):
                raise ToolError("不允许布尔值")

            if not isinstance(value, int | float):
                raise ToolError("只允许数字常量")

            return self._check_number(value)
            # 处理名称常量
        if isinstance(node, ast.Name):
            if node.id not in _CONSTANTS:
                raise ToolError(
                    f"不允许使用名称：{node.id}",
                )

            return _CONSTANTS[node.id]
        # 处理正负号
        if isinstance(node, ast.UnaryOp):
            value = self._evaluate(node.operand)

            if isinstance(node.op, ast.USub):
                return self._check_number(-value)

            if isinstance(node.op, ast.UAdd):
                return self._check_number(value)

            raise ToolError("不允许的单目运算")
        # 处理二元运算
        if isinstance(node, ast.BinOp):
            left = self._evaluate(node.left)
            right = self._evaluate(node.right)

            if isinstance(node.op, ast.Pow):
                if abs(right) > settings.python_max_exponent:
                    raise ToolError("乘方指数过大")

            operation = _BINARY_OPERATORS.get(type(node.op))

            if operation is None:
                raise ToolError(
                    f"不允许的二元运算：{type(node.op).__name__}",
                )

            result = operation(left, right)
            return self._check_number(result)
        # 处理函数调用
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ToolError("不允许的方法调用")
            # 只允许调用白名单中的函数
            function = _FUNCTIONS.get(node.func.id)

            if function is None:
                raise ToolError(
                    f"不允许调用函数：{node.func.id}",
                )

            if node.keywords:
                raise ToolError("不允许使用关键字参数")

            args = [self._evaluate(argument) for argument in node.args]

            result = function(*args)
            return self._check_number(result)

        raise ToolError(
            f"不允许的语法节点：{type(node).__name__}",
        )

    # 负责检查计算结果
    @staticmethod
    def _check_number(value: Any) -> int | float:
        if isinstance(value, bool):
            raise ToolError("结果不能是布尔值")

        if not isinstance(value, int | float):
            raise ToolError("结果必须是数字")

        if isinstance(value, float) and not math.isfinite(value):
            raise ToolError("结果不是有限数字")

        if abs(value) > settings.python_max_abs_value:
            raise ToolError("结果数值过大")

        return value
