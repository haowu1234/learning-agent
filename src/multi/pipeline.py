"""Pipeline 流水线模式

Agent 按固定顺序依次处理，上一个的输出是下一个的输入。
支持自定义任务模板和失败重试。

示例流程:
    task → [研究员] → 素材 → [分析师] → 观点 → [写作者] → 报告
"""

from __future__ import annotations

from src.llm import LLMClient
from src.multi.base import BaseMultiAgent
from src.multi.message import PipelineStep
from src.tools.base import ToolRegistry


class PipelineMultiAgent(BaseMultiAgent):
    """流水线模式：Agent 按顺序依次执行。"""

    def __init__(
        self,
        llm: LLMClient,
        tool_registry: ToolRegistry,
        steps: list[PipelineStep] | None = None,
        verbose: bool = True,
    ):
        super().__init__(llm=llm, tool_registry=tool_registry, verbose=verbose)
        self._steps: list[PipelineStep] = steps or []

    def add_step(self, step: PipelineStep) -> None:
        """添加流水线步骤。"""
        self._steps.append(step)

    def run(self, task: str) -> str:
        """执行流水线。"""
        if not self._steps:
            return "错误：流水线没有定义任何步骤。"

        self.state.reset()
        self.state.task = task
        self.state.status = "executing"
        self.state.plan = [f"Step {i+1}: {s.agent_name}" for i, s in enumerate(self._steps)]

        self._log_header(f"Pipeline 流水线开始 ({len(self._steps)} 步)")
        self._log(f"  任务: {task}")

        prev_result = ""
        final_result = ""

        for i, step in enumerate(self._steps):
            self.state.current_step = i + 1
            self._log(f"\n--- 步骤 {i+1}/{len(self._steps)}: [{step.agent_name}] ---")

            # 渲染任务模板
            rendered_task = step.task_template.format(
                task=task,
                prev_result=prev_result,
                all_results=self.state.get_all_results(),
            )
            self._log_agent(step.agent_name, "接收任务", rendered_task)

            # 执行（含重试）
            result = self._execute_with_retry(step, rendered_task)

            # 可选的结果转换
            if step.transform is not None:
                result = step.transform(result)

            prev_result = result
            final_result = result
            self._log_agent(step.agent_name, "输出结果", result)
            self._fire_hook("on_step_complete", step=i + 1, state=self.state)

        self.state.status = "done"
        self._log_header("Pipeline 完成")
        self._log(f"  {self.state.summary()}")

        return final_result

    def _execute_with_retry(self, step: PipelineStep, task: str) -> str:
        """带重试的执行。"""
        last_error = ""
        for attempt in range(1, step.retry + 1):
            result = self._dispatch(step.agent_name, task)
            if not result.startswith("错误："):
                return result
            last_error = result
            if attempt < step.retry:
                self._log(f"  ⚠️  重试 ({attempt}/{step.retry})...")
        return last_error
