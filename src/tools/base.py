"""工具基类与注册机制

定义 Tool 抽象基类和 ToolRegistry 工具注册中心。
每个工具需要：
  - name: 工具名称
  - description: 工具描述（给 LLM 看）
  - parameters: 参数的 JSON Schema 定义
  - run(): 执行方法
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """工具抽象基类。所有工具必须继承此类并实现 run 方法。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称，用于 LLM 调用时标识。"""

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述，告诉 LLM 这个工具能做什么。"""

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """工具参数的 JSON Schema 定义。"""

    @abstractmethod
    def run(self, **kwargs: Any) -> str:
        """执行工具逻辑。

        Args:
            **kwargs: 工具参数，与 parameters 中定义的参数对应。

        Returns:
            工具执行结果的字符串表示。
        """

    def to_openai_tool(self) -> dict[str, Any]:
        """转换为 OpenAI Function Calling 工具格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def __repr__(self) -> str:
        return f"Tool(name={self.name!r})"


class ToolRegistry:
    """工具注册中心，管理所有可用工具。"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """注册一个工具。"""
        if tool.name in self._tools:
            raise ValueError(f"工具 '{tool.name}' 已注册，请勿重复注册。")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """根据名称获取工具。"""
        return self._tools.get(name)

    def execute(self, name: str, arguments: str | dict[str, Any]) -> str:
        """执行指定工具。

        Args:
            name: 工具名称。
            arguments: 工具参数，可以是 JSON 字符串或字典。

        Returns:
            工具执行结果。

        Raises:
            ValueError: 工具不存在时抛出。
        """
        tool = self.get(name)
        if tool is None:
            return f"错误：未找到工具 '{name}'。可用工具：{list(self._tools.keys())}"

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return f"错误：无法解析工具参数 JSON：{arguments}"

        try:
            return tool.run(**arguments)
        except Exception as e:
            return f"错误：工具 '{name}' 执行失败：{e}"

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """将所有工具转换为 OpenAI Function Calling 格式列表。"""
        return [tool.to_openai_tool() for tool in self._tools.values()]

    def get_tools_description(self) -> str:
        """生成工具描述文本，用于纯文本模式的 Prompt。"""
        lines = []
        for tool in self._tools.values():
            params = json.dumps(tool.parameters, ensure_ascii=False, indent=2)
            lines.append(f"- **{tool.name}**: {tool.description}\n  参数: {params}")
        return "\n".join(lines)

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={self.tool_names})"
