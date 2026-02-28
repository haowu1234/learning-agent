"""Orchestrator 编排者模式

一个 Planner Agent 负责拆解任务、动态分派给专业 Agent 执行，
并根据执行结果决定下一步行动，支持重新规划。

示例流程:
    task → [Planner 规划] → step1 → [Agent A] → step2 → [Agent B] → ... → [汇总] → final
"""

from __future__ import annotations

import json
import re
from typing import Any

from src.llm import LLMClient
from src.multi.base import BaseMultiAgent
from src.multi.message import Message, MessageType
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
        self.state.reset()
        self.state.task = task
        self.state.status = "planning"

        self._log_header(f"Orchestrator 编排模式开始")
        self._log(f"  任务: {task}")
        self._log(f"  可用 Agent: {self.agent_names}")

        # Step 1: 规划
        plan = self._plan(task)
        if not plan:
            return "错误：无法生成执行计划。"

        self.state.plan = [f"{p['agent']}: {p['task']}" for p in plan]
        self._log(f"\n📋 执行计划 ({len(plan)} 步):")
        for i, step in enumerate(plan, 1):
            self._log(f"  {i}. [{step['agent']}] {step['task']}")

        # Step 2: 按计划执行
        self.state.status = "executing"
        for i, step in enumerate(plan):
            self.state.current_step = i + 1
            agent_name = step["agent"]
            sub_task = step["task"]

            # 如果不是第一步，附上之前的结果作为上下文
            if self.state.results:
                sub_task += f"\n\n[参考信息] 前面步骤的结果:\n{self.state.get_all_results()}"

            self._log(f"\n--- 执行步骤 {i+1}: [{agent_name}] ---")
            self._log_agent(agent_name, "接收任务", step["task"])

            if agent_name not in self._agents:
                self._log(f"  ⚠️  Agent '{agent_name}' 不存在，跳过")
                continue

            result = self._dispatch(agent_name, sub_task)
            self._log_agent(agent_name, "输出结果", result)
            self._fire_hook("on_step_complete", step=i + 1, state=self.state)

        # Step 3: 汇总
        self.state.status = "reviewing"
        self._log(f"\n--- 汇总阶段 ---")
        final = self._summarize(task)

        self.state.status = "done"
        self._log_header("Orchestrator 完成")
        self._log(f"  {self.state.summary()}")

        return final

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
        # 尝试提取 JSON 块
        json_match = re.search(r"\[.*\]", text, re.DOTALL)
        if not json_match:
            return []

        try:
            plan = json.loads(json_match.group())
            # 验证格式
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
