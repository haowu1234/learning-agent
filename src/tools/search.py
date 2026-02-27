"""网页搜索工具（Mock 版本）

返回模拟的搜索结果，用于学习和测试 Agent 工具调用流程。
生产环境中可替换为真实搜索 API（如 Bing Search、SerpAPI 等）。
"""

from __future__ import annotations

from typing import Any

from src.tools.base import Tool

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
    """网页搜索工具（Mock 数据）。"""

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

    def run(self, query: str, **_: Any) -> str:
        # 在 mock 数据中查找匹配结果
        query_lower = query.lower()
        for keyword, results in _MOCK_RESULTS.items():
            if keyword in query_lower:
                lines = [f"搜索 '{query}' 的结果："]
                for i, r in enumerate(results, 1):
                    lines.append(f"  {i}. [{r['title']}]({r['url']})")
                    lines.append(f"     {r['snippet']}")
                return "\n".join(lines)

        # 无匹配时返回通用结果
        return (
            f"搜索 '{query}' 的结果：\n"
            f"  1. [相关文章] - 找到了一些关于 '{query}' 的信息。\n"
            f"  2. [维基百科] - '{query}' 的百科介绍。"
        )
