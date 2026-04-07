"""Orchestrator 编排者模式

一个 Planner Agent 负责拆解任务、动态分派给专业 Agent 执行，
并根据执行结果决定下一步行动，支持失败后的重新规划。

示例流程:
    task → [Planner 规划] → step1 → [Agent A] → step2 → [Agent B] → ... → [汇总] → final
"""

from __future__ import annotations

import json
import re
from typing import Any

from src.llm import LLMClient
from src.multi.base import BaseMultiAgent
from src.tools.base import ToolRegistry

PLANNER_PROMPT = """你是一个任务规划和编排专家。你需要将复杂任务拆解为子任务，并分配给合适的 Agent 执行。

可用的 Agent：
{agents_description}

你需要输出一个 JSON 格式的执行计划，格式如下：
```json
[
  {{"agent": "agent_name", "task": "具体的子任务描述"}},
  {{"agent": "agent_name", "task": "具体的子任务描述"}}
]
```

规则：
- 每个子任务要具体明确，让对应的 Agent 能直接执行
- 合理安排顺序，后面的任务可以依赖前面的结果
- agent 名称必须是可用 Agent 之一
- 只输出 JSON，不要输出其他内容"""

SUMMARIZER_PROMPT = """你是一个汇总专家。请根据以下各个 Agent 的执行结果，综合生成一个完整、结构化的最终回答。

原始任务：{task}

各 Agent 的执行结果：
{all_results}

请综合以上信息，给出完整的最终回答。要求：
- 结构清晰，使用标题和列表
- 涵盖所有 Agent 提供的关键信息
- 语言简洁专业"""


class OrchestratorMultiAgent(BaseMultiAgent):
    """编排者模式：LLM 动态规划并分派子任务。"""

    def __init__(
        self,
        llm: LLMClient,
        tool_registry: ToolRegistry,
        max_replan: int = 2,
        verbose: bool = True,
    ):
        """
        Args:
            max_replan: 最大重新规划次数。
        """
        super().__init__(llm=llm, tool_registry=tool_registry, verbose=verbose)
        self._max_replan = max_replan

    def run(self, task: str) -> str:
        """执行编排模式。"""
        original_task = task
        working_task = task
        total_attempts = self._max_replan + 1
        last_error = "错误：无法生成执行计划。"

        for attempt in range(1, total_attempts + 1):
            self.state.reset()
            self.state.task = original_task
            self.state.status = "planning"
            self.state.metadata["attempt"] = attempt

            self._log_header(f"Orchestrator 编排模式开始（尝试 {attempt}/{total_attempts}）")
            self._log(f"  原始任务: {original_task}")
            if attempt > 1:
                self._log(f"  当前工作任务: {working_task}")
            self._log(f"  可用 Agent: {self.agent_names}")

            # Step 1: 规划
            plan = self._plan(working_task)
            if not plan:
                last_error = "错误：无法生成执行计划。"
                self._log(f"  ⚠️  {last_error}")
                if attempt < total_attempts:
                    working_task = self._build_replan_task(
                        original_task=original_task,
                        working_task=working_task,
                        plan=[],
                        errors=[last_error],
                    )
                    self._log("  🔁 进入下一轮重新规划。")
                    continue

                self.state.status = "failed"
                return last_error

            self.state.plan = [f"{p['agent']}: {p['task']}" for p in plan]
            self._log(f"\n📋 执行计划 ({len(plan)} 步):")
            for i, step in enumerate(plan, 1):
                self._log(f"  {i}. [{step['agent']}] {step['task']}")

            # Step 2: 按计划执行
            self.state.status = "executing"
            execution_errors: list[str] = []

            for i, step in enumerate(plan):
                self.state.current_step = i + 1
                agent_name = step["agent"]
                sub_task = step["task"]

                if self.state.results:
                    sub_task += f"\n\n[参考信息] 前面步骤的结果:\n{self.state.get_all_results()}"

                self._log(f"\n--- 执行步骤 {i + 1}: [{agent_name}] ---")
                self._log_agent(agent_name, "接收任务", step["task"])

                if agent_name not in self._agents:
                    error = f"错误：Agent '{agent_name}' 不存在。"
                    self._log(f"  ⚠️  {error}")
                    execution_errors.append(error)
                    break

                result = self._dispatch(agent_name, sub_task)
                self._log_agent(agent_name, "输出结果", result)
                self._fire_hook("on_step_complete", step=i + 1, state=self.state)

                if result.startswith("错误："):
                    execution_errors.append(
                        f"步骤 {i + 1} 的 Agent '{agent_name}' 执行失败：{result}"
                    )
                    break

            if execution_errors:
                last_error = execution_errors[-1]
                self._log(f"\n⚠️  本轮执行失败：{last_error}")
                if attempt < total_attempts:
                    working_task = self._build_replan_task(
                        original_task=original_task,
                        working_task=working_task,
                        plan=plan,
                        errors=execution_errors,
                    )
                    self._log("🔁 根据失败信息重新规划下一轮。")
                    continue

                self.state.status = "failed"
                return last_error

            # Step 3: 汇总
            self.state.status = "reviewing"
            self._log(f"\n--- 汇总阶段 ---")
            final = self._summarize(original_task)

            self.state.status = "done"
            self._log_header("Orchestrator 完成")
            self._log(f"  {self.state.summary()}")
            return final

        self.state.status = "failed"
        return last_error

    def _plan(self, task: str) -> list[dict[str, str]]:
        """用 LLM 生成执行计划。"""
        agents_desc = "\n".join(
            f"- {name}: {getattr(agent, '_role_system_prompt', '通用Agent')[:80]}..."
            for name, agent in self._agents.items()
        )

        prompt = PLANNER_PROMPT.format(agents_description=agents_desc)
        self._log(f"\n🤔 正在规划...")

        response = self.llm.chat_simple(prompt=task, system=prompt)
        self._log(f"  Planner 输出: {response[:300]}...")

        return self._parse_plan(response)

    def _parse_plan(self, text: str) -> list[dict[str, str]]:
        """从 LLM 输出中解析执行计划 JSON。"""
        json_match = re.search(r"\[.*\]", text, re.DOTALL)
        if not json_match:
            return []

        try:
            plan = json.loads(json_match.group())
            validated = []
            for item in plan:
                if isinstance(item, dict) and "agent" in item and "task" in item:
                    validated.append({"agent": item["agent"], "task": item["task"]})
            return validated
        except (json.JSONDecodeError, TypeError):
            return []

    def _summarize(self, task: str) -> str:
        """汇总所有 Agent 的结果。"""
        prompt = SUMMARIZER_PROMPT.format(
            task=task,
            all_results=self.state.get_all_results(),
        )
        return self.llm.chat_simple(prompt=prompt)

    def _build_replan_task(
        self,
        *,
        original_task: str,
        working_task: str,
        plan: list[dict[str, str]],
        errors: list[str],
    ) -> str:
        """构造带失败上下文的下一轮规划任务。"""
        plan_text = self._format_plan(plan)
        error_text = "\n".join(f"- {error}" for error in errors)
        previous_results = self.state.get_all_results()
        return (
            f"{original_task}\n\n"
            "[上轮执行复盘]\n"
            f"上轮工作任务：{working_task}\n\n"
            f"上轮计划：\n{plan_text}\n\n"
            f"上轮结果：\n{previous_results}\n\n"
            f"上轮失败信息：\n{error_text}\n\n"
            "请基于以上失败信息重新规划，避免重复失败，并尽量保留已经有效的信息。"
        )

    @staticmethod
    def _format_plan(plan: list[dict[str, str]]) -> str:
        """格式化计划文本。"""
        if not plan:
            return "(暂无有效计划)"
        return "\n".join(
            f"{index}. [{item['agent']}] {item['task']}"
            for index, item in enumerate(plan, 1)
        )
