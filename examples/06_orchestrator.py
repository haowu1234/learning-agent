"""第 6 课：Orchestrator 编排者模式

本课目标：
- 体验 LLM 自主规划和分派任务
- 观察编排者如何拆解复杂问题
- 理解动态任务分派与静态流水线的区别

运行方式：
    python -m examples.06_orchestrator
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm import LLMClient
from src.tools.base import ToolRegistry
from src.tools.calculator import CalculatorTool
from src.tools.weather import WeatherTool
from src.tools.search import SearchTool
from src.multi.orchestrator import OrchestratorMultiAgent
from src.multi.roles import get_role


def main():
    print("=" * 60)
    print("第 6 课：Orchestrator 编排者模式")
    print("=" * 60)

    llm = LLMClient()

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(WeatherTool())
    registry.register(SearchTool())

    # 创建 Orchestrator
    orchestrator = OrchestratorMultiAgent(
        llm=llm,
        tool_registry=registry,
        verbose=True,
    )

    # 添加可用 Agent
    orchestrator.add_agent(get_role("researcher"))
    orchestrator.add_agent(get_role("analyst"))
    orchestrator.add_agent(get_role("writer"))

    print(f"\n已注册 Agent: {orchestrator.agent_names}")

    # 执行复杂任务
    task = (
        "我想了解 Python 在数据科学领域的应用现状。"
        "请搜索相关信息，分析 Python 的优势和不足，"
        "最后写一篇简短的分析报告。"
    )
    result = orchestrator.run(task)

    print(f"\n{'='*60}")
    print("📄 最终报告:")
    print(f"{'='*60}")
    print(result)

    print(f"\n\n📝 知识点总结：")
    print("1. Orchestrator 用 LLM 自动拆解任务并生成执行计划")
    print("2. 相比 Pipeline，Orchestrator 不需要预定义步骤顺序")
    print("3. 编排者动态决定调用哪个 Agent、传递什么任务")
    print("4. 最后自动汇总所有结果生成最终回答")


if __name__ == "__main__":
    main()
