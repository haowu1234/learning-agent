"""第 7 课：Debate 辩论模式

本课目标：
- 体验多 Agent 围绕同一话题辩论
- 观察 Agent 如何回应和反驳对方观点
- 理解辩论如何帮助得出更全面的结论

运行方式：
    python -m examples.07_debate
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm import LLMClient
from src.tools.base import ToolRegistry
from src.tools.search import SearchTool
from src.multi.debate import DebateMultiAgent
from src.multi.roles import AgentRole, get_role


def main():
    print("=" * 60)
    print("第 7 课：Debate 辩论模式")
    print("=" * 60)

    llm = LLMClient()

    registry = ToolRegistry()
    registry.register(SearchTool())

    # 创建辩论
    debate = DebateMultiAgent(
        llm=llm,
        tool_registry=registry,
        max_rounds=2,
        verbose=True,
    )

    # 添加辩论者
    debate.add_debater(get_role("python_expert"))
    debate.add_debater(get_role("go_expert"))

    # 设置裁判
    judge_role = AgentRole(
        name="judge",
        description="技术裁判",
        system_prompt=(
            "你是一个公正客观的技术裁判。"
            "你需要综合各方观点，给出平衡、深入的最终结论。"
            "不偏向任何一方，但要明确给出推荐意见。"
        ),
        tools=[],
    )
    debate.set_judge(judge_role)

    print(f"\n辩论参与者: {debate._debater_names}")
    print(f"裁判: {debate._judge_role.name}")

    # 开始辩论
    topic = "Python vs Go：哪个更适合开发后端微服务？请从性能、开发效率、生态、运维等角度分析。"
    result = debate.run(topic)

    print(f"\n{'='*60}")
    print("⚖️  最终裁决:")
    print(f"{'='*60}")
    print(result)

    print(f"\n\n📝 知识点总结：")
    print("1. Debate 模式让多个专家各抒己见，减少单一视角的偏见")
    print("2. 第二轮辩论中，Agent 能看到对方观点并进行回应")
    print("3. 裁判综合所有轮次的观点做出最终裁决")
    print("4. 这种模式适合需要多角度分析的决策场景")


if __name__ == "__main__":
    main()
