"""第 4 课：加入记忆能力

本课目标：
- 实现多轮对话，Agent 记住之前的上下文
- 理解对话历史如何影响 Agent 的推理
- 体验上下文窗口管理

运行方式：
    python -m examples.04_with_memory
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
    print("第 4 课：加入记忆能力")
    print("=" * 60)

    llm = LLMClient()

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(WeatherTool())
    registry.register(SearchTool())

    agent = ReActAgent(
        llm=llm,
        tool_registry=registry,
        mode="function_calling",
        verbose=True,
    )

    # ---- 多轮对话演示 ----
    conversations = [
        "北京今天天气怎么样？",
        "那上海呢？",  # Agent 需要记住上一轮在讨论天气
        "这两个城市的温度差是多少？计算一下",  # Agent 需要记住之前的温度数据
    ]

    print("\n🗣️  开始多轮对话：")
    for i, query in enumerate(conversations, 1):
        print(f"\n\n{'='*60}")
        print(f"第 {i} 轮对话")
        print(f"{'='*60}")
        result = agent.run(query)
        print(f"\n✅ 回答: {result}")

    # ---- 查看对话历史 ----
    print(f"\n\n📊 对话历史统计：")
    print(f"  总轮数: {agent.history.turn_count}")
    print(f"  消息数: {len(agent.history)}")

    print("\n📜 完整对话历史：")
    for msg in agent.history.get_messages():
        role = "👤 用户" if msg["role"] == "user" else "🤖 助手"
        content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
        print(f"  {role}: {content}")

    # ---- 重置后对比 ----
    print(f"\n\n🔄 重置对话历史...")
    agent.reset()
    print(f"  轮数: {agent.history.turn_count}")

    print("\n\n📝 知识点总结：")
    print("1. ConversationHistory 自动管理多轮对话上下文")
    print("2. Agent 能通过历史消息理解指代（'那上海呢' → 天气）")
    print("3. 历史信息帮助 Agent 在多轮间串联数据")
    print("4. max_turns 限制防止上下文过长导致性能下降")


if __name__ == "__main__":
    main()
