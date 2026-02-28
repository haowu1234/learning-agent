"""第 5 课：Pipeline 流水线模式

本课目标：
- 体验研究员→分析师→写作者的流水线协作
- 理解 Agent 间如何传递上下文
- 观察每个 Agent 的专业化输出

运行方式：
    python -m examples.05_pipeline
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm import LLMClient
from src.tools.base import ToolRegistry
from src.tools.calculator import CalculatorTool
from src.tools.weather import WeatherTool
from src.tools.search import SearchTool
from src.multi.pipeline import PipelineMultiAgent
from src.multi.message import PipelineStep
from src.multi.roles import get_role


def main():
    print("=" * 60)
    print("第 5 课：Pipeline 流水线模式")
    print("=" * 60)

    llm = LLMClient()

    # 全局工具注册
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(WeatherTool())
    registry.register(SearchTool())

    # 定义流水线步骤
    steps = [
        PipelineStep(
            agent_name="researcher",
            task_template=(
                "请研究以下课题并整理关键信息：\n{task}\n\n"
                "要求：搜索相关资料，整理出 3-5 个关键要点。"
            ),
        ),
        PipelineStep(
            agent_name="analyst",
            task_template=(
                "以下是研究员收集的信息，请进行深入分析：\n\n"
                "{prev_result}\n\n"
                "原始课题：{task}\n\n"
                "要求：提炼核心观点，给出数据支撑的结论。"
            ),
        ),
        PipelineStep(
            agent_name="writer",
            task_template=(
                "请根据以下研究和分析结果，撰写一篇简短的分析报告：\n\n"
                "【研究与分析】\n{prev_result}\n\n"
                "原始课题：{task}\n\n"
                "要求：结构清晰，语言专业，500字以内。"
            ),
        ),
    ]

    # 创建 Pipeline
    pipeline = PipelineMultiAgent(
        llm=llm,
        tool_registry=registry,
        steps=steps,
        verbose=True,
    )

    # 添加角色
    pipeline.add_agent(get_role("researcher"))
    pipeline.add_agent(get_role("analyst"))
    pipeline.add_agent(get_role("writer"))

    print(f"\n已注册 Agent: {pipeline.agent_names}")
    print(f"流水线步骤: {len(steps)} 步")

    # 执行
    task = "分析北京、上海、广州三个城市的天气状况，评估哪个城市最适合本周出行"
    result = pipeline.run(task)

    print(f"\n{'='*60}")
    print("📄 最终报告:")
    print(f"{'='*60}")
    print(result)

    print(f"\n\n📝 知识点总结：")
    print("1. Pipeline 模式让每个 Agent 专注自己的职责")
    print("2. 上一步的输出通过 {prev_result} 传给下一步")
    print("3. 研究员收集信息 → 分析师提炼观点 → 写作者输出报告")
    print("4. 每个 Agent 只能使用自己角色允许的工具")


if __name__ == "__main__":
    main()
