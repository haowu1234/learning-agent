"""计算器工具

安全地执行数学表达式，使用 ast 模块进行安全解析，
防止代码注入攻击。
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from src.tools.base import Tool

# 支持的运算符
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# 支持的数学函数
_MATH_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "pi": math.pi,
    "e": math.e,
}


def _safe_eval(node: ast.AST) -> float:
    """递归地安全计算 AST 节点的值。"""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"不支持的常量类型: {type(node.value)}")
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _OPERATORS:
            raise ValueError(f"不支持的运算符: {op_type.__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _OPERATORS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _OPERATORS:
            raise ValueError(f"不支持的一元运算符: {op_type.__name__}")
        operand = _safe_eval(node.operand)
        return _OPERATORS[op_type](operand)
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _MATH_FUNCTIONS:
            func = _MATH_FUNCTIONS[node.func.id]
            if callable(func):
                args = [_safe_eval(arg) for arg in node.args]
                return float(func(*args))
            return float(func)
        raise ValueError(f"不支持的函数: {ast.dump(node.func)}")
    elif isinstance(node, ast.Name):
        if node.id in _MATH_FUNCTIONS:
            val = _MATH_FUNCTIONS[node.id]
            if not callable(val):
                return float(val)
        raise ValueError(f"不支持的变量: {node.id}")
    else:
        raise ValueError(f"不支持的表达式类型: {type(node).__name__}")


class CalculatorTool(Tool):
    """安全的数学计算器工具。"""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return (
            "数学计算器，可以安全地计算数学表达式。"
            "支持加减乘除、幂运算、取模，以及 sqrt、sin、cos、tan、log 等函数。"
            "示例：'2 + 3 * 4'、'sqrt(16)'、'sin(pi/2)'"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "要计算的数学表达式，例如 '2 + 3 * 4' 或 'sqrt(16)'",
                }
            },
            "required": ["expression"],
        }

    def run(self, expression: str, **_: Any) -> str:
        try:
            tree = ast.parse(expression, mode="eval")
            result = _safe_eval(tree)
            if result == int(result):
                return str(int(result))
            return str(round(result, 10))
        except ZeroDivisionError:
            return "错误：除以零"
        except (ValueError, TypeError, SyntaxError) as e:
            return f"错误：无法计算表达式 '{expression}'：{e}"
