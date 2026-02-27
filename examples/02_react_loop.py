"""第 2 课：实现 ReAct 循环

本课目标：
- 使用 ReActAgent 完成完整的 Thought → Action → Observation 循环
- 对比 function_calling 和 text_parsing 两种模式
- 观察 Agent 的推理过程

运行方式：
    python -m examples.02_react_loop
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm import LLMClient
from src.tools.base import ToolRegistry
from src.tools.calculator import CalculatorTool
from src.agent.react import ReActAgent


def main():
    print("=" * 60)
    print("第 2 课：ReAct 循环")
    print("=" * 60)

    llm = LLMClient()

    # 注册工具
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    # ---- 模式 1：Function Calling ----
    print("\n\n📌 模式 1：OpenAI Function Calling")
    print("-" * 40)

    agent_fc = ReActAgent(
        llm=llm,
        tool_registry=registry,
        mode="function_calling",
        verbose=True,
    )

    query = "一个圆的半径是 7cm，请计算它的面积（使用 pi * r^2）"
    result = agent_fc.run(query)
    print(f"\n✅ 最终结果: {result}")

    # ---- 模式 2：纯文本解析 ----
    print("\n\n📌 模式 2：纯文本解析")
    print("-" * 40)

    agent_tp = ReActAgent(
        llm=llm,
        tool_registry=registry,
        mode="text_parsing",
        verbose=True,
    )

    result2 = agent_tp.run(query)
    print(f"\n✅ 最终结果: {result2}")

    # ---- 对比总结 ----
    print("\n\n📝 两种模式对比：")
    print("┌──────────────────┬──────────────────────────────────┐")
    print("│ Function Calling │ 纯文本解析                        │")
    print("├──────────────────┼──────────────────────────────────┤")
    print("│ 结构化输出       │ 依赖正则提取                      │")
    print("│ 需要模型支持     │ 兼容所有 LLM                     │")
    print("│ 解析可靠         │ 可能格式不规范                    │")
    print("│ 生产推荐         │ 适合学习理解                      │")
    print("└──────────────────┴──────────────────────────────────┘")


if __name__ == "__main__":
    main()
