"""网页搜索工具。

默认返回 Mock 结果，便于学习和测试 Agent 工具调用流程。
当检测到 MCP 配置后，可自动切换到基于 Python MCP SDK 的真实搜索后端。
"""

from __future__ import annotations

import os
from typing import Any

from src.tools.base import Tool
from src.tools.mcp_client import MCPStdioClient

# 模拟搜索结果
_MOCK_RESULTS: dict[str, list[dict[str, str]]] = {
    "python": [
        {"title": "Python 官方文档", "url": "https://docs.python.org", "snippet": "Python 是一种解释型、面向对象的高级编程语言。"},
        {"title": "Python 教程 - 菜鸟教程", "url": "https://www.runoob.com/python3", "snippet": "Python3 基础教程，从入门到精通。"},
    ],
    "react agent": [
        {"title": "ReAct: Synergizing Reasoning and Acting in LLMs", "url": "https://arxiv.org/abs/2210.03629", "snippet": "ReAct 论文提出了一种将推理和行动结合的方法。"},
        {"title": "Building a ReAct Agent from Scratch", "url": "https://example.com/react-agent", "snippet": "手把手教你从零构建一个 ReAct Agent。"},
    ],
    "大语言模型": [
        {"title": "什么是大语言模型 (LLM)", "url": "https://example.com/llm-intro", "snippet": "大语言模型是基于 Transformer 架构的大规模预训练语言模型。"},
        {"title": "LLM 应用开发指南", "url": "https://example.com/llm-dev", "snippet": "介绍如何使用大语言模型构建智能应用。"},
    ],
}


class SearchTool(Tool):
    """网页搜索工具，支持 mock 与 MCP 两种后端。"""

    def __init__(
        self,
        *,
        backend: str = "mock",
        mcp_client: MCPStdioClient | None = None,
        mcp_tool_name: str | None = None,
        query_argument: str = "query",
    ) -> None:
        normalized_backend = backend.strip().lower() if backend else "mock"
        if normalized_backend not in {"mock", "mcp"}:
            raise ValueError(f"不支持的 search backend: {backend}")
        if not query_argument.strip():
            raise ValueError("query_argument 不能为空。")

        self.backend = normalized_backend
        self.mcp_client = mcp_client
        self.mcp_tool_name = mcp_tool_name.strip() if mcp_tool_name else None
        self.query_argument = query_argument.strip()

    @classmethod
    def from_env(cls, *, mcp_client: MCPStdioClient | None = None) -> SearchTool:
        """根据环境变量选择 mock 或 MCP 后端。"""
        provider = os.getenv("SEARCH_PROVIDER", "").strip().lower()
        resolved_mcp_client = mcp_client or MCPStdioClient.from_env()
        mcp_tool_name = os.getenv("MCP_SEARCH_TOOL_NAME", "").strip() or None
        query_argument = os.getenv("MCP_SEARCH_QUERY_ARGUMENT", "query").strip() or "query"

        if provider == "mock":
            return cls()
        if provider == "mcp":
            return cls(
                backend="mcp",
                mcp_client=resolved_mcp_client,
                mcp_tool_name=mcp_tool_name,
                query_argument=query_argument,
            )
        if resolved_mcp_client is not None:
            return cls(
                backend="mcp",
                mcp_client=resolved_mcp_client,
                mcp_tool_name=mcp_tool_name,
                query_argument=query_argument,
            )
        return cls()

    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return "搜索互联网获取相关信息。当你需要查找最新资讯、事实性知识或不确定的信息时使用。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词",
                }
            },
            "required": ["query"],
        }

    def backend_label(self) -> str:
        if self.backend == "mock":
            return "mock"
        if self.mcp_client is None:
            return "mcp(unconfigured)"
        label = self.mcp_client.describe()
        if self.mcp_tool_name:
            return f"{label}#{self.mcp_tool_name}"
        return label

    def run(self, query: str, **_: Any) -> str:
        if self.backend == "mcp":
            return self._run_mcp(query)
        return self._run_mock(query)

    def _run_mcp(self, query: str) -> str:
        if self.mcp_client is None:
            return (
                "错误：search 已切换到 MCP 后端，但未检测到 MCP server 配置。"
                "请设置 MCP_SEARCH_SERVER_COMMAND，必要时再补充 "
                "MCP_SEARCH_SERVER_ARGS / MCP_SEARCH_TOOL_NAME。"
            )

        try:
            result = self.mcp_client.call_tool(
                tool_name=self.mcp_tool_name,
                arguments={self.query_argument: query},
            )
        except Exception as exc:
            return f"错误：MCP 搜索失败：{exc}"

        result = result.strip()
        if result.startswith(f"搜索 '{query}'"):
            return result
        return f"搜索 '{query}' 的结果：\n{result}"

    @staticmethod
    def _run_mock(query: str) -> str:
        query_lower = query.lower()
        for keyword, results in _MOCK_RESULTS.items():
            if keyword in query_lower:
                lines = [f"搜索 '{query}' 的结果："]
                for i, result in enumerate(results, 1):
                    lines.append(f"  {i}. [{result['title']}]({result['url']})")
                    lines.append(f"     {result['snippet']}")
                return "\n".join(lines)

        return (
            f"搜索 '{query}' 的结果：\n"
            f"  1. [相关文章] - 找到了一些关于 '{query}' 的信息。\n"
            f"  2. [维基百科] - '{query}' 的百科介绍。"
        )
