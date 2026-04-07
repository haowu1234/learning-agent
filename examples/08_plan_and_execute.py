"""第 8 课：Plan-and-Execute 模式

本课目标：
- 体验 Agent 先规划、再执行、最后汇总的工作流
- 理解“任务策略模式”和“工具调用协议模式”的区别
- 观察 plan_and_execute 如何复用 function_calling / text_parsing 作为执行器
- 体验 Verifier / Reflection / Replan 如何形成闭环

运行方式：
    python -m examples.08_plan_and_execute
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.react import ReActAgent
from src.llm import LLMClient
from src.tools.base import ToolRegistry
from src.tools.calculator import CalculatorTool
from src.tools.search import SearchTool
from src.tools.weather import WeatherTool


def main():
    print("=" * 60)
    print("第 8 课：Plan-and-Execute 模式")
    print("=" * 60)

    llm = LLMClient()

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(WeatherTool())
    registry.register(SearchTool())

    print(f"\n已注册工具: {registry.tool_names}")
    print(f"支持模式: {ReActAgent.available_modes()}")

    agent = ReActAgent(
        llm=llm,
        tool_registry=registry,
        mode="plan_and_execute",
        executor_mode="function_calling",
        max_plan_steps=4,
        verbose=True,
    )

    query = "如果北京今天的温度乘以 3 再加上 100，结果是多少？请先确认天气，再完成计算，并给出简短结论。"
    result = agent.run(query)

    print(f"\n✅ 最终结果: {result}")

    print("\n\n📝 知识点总结：")
    print("1. plan_and_execute 会先生成步骤计划，再逐步执行")
    print("2. executor_mode 决定每个子任务内部使用哪种工具调用协议")
    print("3. Verifier 会在汇总后做任务验收，不通过时会触发下一轮")
    print("4. Reflection 会把失败原因转成新的工作任务，驱动重新规划")


if __name__ == "__main__":
    main()
