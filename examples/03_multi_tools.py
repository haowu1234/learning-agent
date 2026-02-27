"""第 3 课：多工具协作

本课目标：
- 注册多个工具，观察 Agent 如何自主选择
- 体验 Agent 分解复杂问题、多步推理的能力
- 理解工具选择的决策过程

运行方式：
    python -m examples.03_multi_tools
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm import LLMClient
from src.tools.base import ToolRegistry
from src.tools.calculator import CalculatorTool
from src.tools.weather import WeatherTool
from src.tools.search import SearchTool
from src.agent.react import ReActAgent


def main():
    print("=" * 60)
    print("第 3 课：多工具协作")
    print("=" * 60)

    llm = LLMClient()

    # 注册多个工具
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(WeatherTool())
    registry.register(SearchTool())

    print(f"\n已注册工具: {registry.tool_names}")

    agent = ReActAgent(
        llm=llm,
        tool_registry=registry,
        mode="function_calling",
        verbose=True,
    )

    # ---- 测试 1：需要天气工具 ----
    print("\n\n🌤️  测试 1：天气查询")
    print("-" * 40)
    result1 = agent.run("北京和上海今天的天气怎么样？哪个城市更暖和？")
    print(f"\n✅ 结果: {result1}")
    agent.reset()

    # ---- 测试 2：需要搜索 + 计算 ----
    print("\n\n🔍 测试 2：搜索 + 推理")
    print("-" * 40)
    result2 = agent.run("帮我搜索一下 Python 的相关信息")
    print(f"\n✅ 结果: {result2}")
    agent.reset()

    # ---- 测试 3：复合问题，可能需要多个工具 ----
    print("\n\n🧮 测试 3：复合问题")
    print("-" * 40)
    result3 = agent.run(
        "如果北京今天的温度乘以 3 再加上 100，结果是多少？请先查天气再计算。"
    )
    print(f"\n✅ 结果: {result3}")

    print("\n\n📝 知识点总结：")
    print("1. Agent 根据问题语义自动选择最合适的工具")
    print("2. 复杂问题可能需要多次工具调用（先查后算）")
    print("3. Agent 能够将多个工具的结果综合起来给出答案")


if __name__ == "__main__":
    main()
