"""Debate 辩论模式

多个 Agent 围绕同一话题各抒己见，经过多轮讨论后由裁判 Agent 做出最终裁决。

示例流程:
    Round 1: 所有 Agent 独立回答
    Round 2: 看到对方观点后修正
    Round 3: 裁判综合裁决
"""

from __future__ import annotations

from src.llm import LLMClient
from src.multi.base import BaseMultiAgent
from src.multi.message import Message, MessageType
from src.multi.roles import AgentRole
from src.tools.base import ToolRegistry

JUDGE_PROMPT = """你是一个公正的裁判。多位专家围绕以下话题进行了辩论，请你综合所有观点，给出最终裁决。

原始话题：{topic}

{rounds_summary}

请你：
1. 总结各方的核心观点和论据
2. 分析各方观点的优缺点
3. 给出你的最终结论和建议
4. 说明你的裁决理由

要求：客观公正，论据充分，结论明确。"""


class DebateMultiAgent(BaseMultiAgent):
    """辩论模式：多 Agent 讨论，裁判做最终裁决。"""

    def __init__(
        self,
        llm: LLMClient,
        tool_registry: ToolRegistry,
        max_rounds: int = 2,
        verbose: bool = True,
    ):
        """
        Args:
            max_rounds: 最大辩论轮数（不含裁决轮）。
        """
        super().__init__(llm=llm, tool_registry=tool_registry, verbose=verbose)
        self._max_rounds = max_rounds
        self._judge_role: AgentRole | None = None
        self._debater_names: list[str] = []

    def set_judge(self, role: AgentRole) -> None:
        """设置裁判角色。"""
        self._judge_role = role
        self.add_agent(role)

    def add_debater(self, role: AgentRole, **kwargs) -> None:
        """添加辩论参与者。"""
        self.add_agent(role, **kwargs)
        self._debater_names.append(role.name)

    def run(self, task: str) -> str:
        """执行辩论。"""
        if not self._debater_names:
            return "错误：没有辩论参与者。"
        if self._judge_role is None:
            return "错误：没有设置裁判。"

        self.state.reset()
        self.state.task = task
        self.state.status = "executing"

        self._log_header(f"Debate 辩论模式开始")
        self._log(f"  话题: {task}")
        self._log(f"  参与者: {self._debater_names}")
        self._log(f"  裁判: {self._judge_role.name}")
        self._log(f"  最大轮数: {self._max_rounds}")

        # 记录所有轮次的观点
        all_rounds: list[dict[str, str]] = []

        for round_num in range(1, self._max_rounds + 1):
            self.state.current_step = round_num
            self._log(f"\n{'─'*50}")
            self._log(f"  📢 第 {round_num} 轮辩论")
            self._log(f"{'─'*50}")

            round_opinions: dict[str, str] = {}

            for debater_name in self._debater_names:
                # 构建该轮的任务
                if round_num == 1:
                    sub_task = (
                        f"请就以下话题发表你的观点：\n\n{task}\n\n"
                        f"要求：给出你的核心观点、论据和结论。"
                    )
                else:
                    # 后续轮次附上其他人的观点
                    others_opinions = self._format_opinions(all_rounds, exclude=debater_name)
                    sub_task = (
                        f"话题：{task}\n\n"
                        f"以下是其他参与者在之前轮次的观点：\n{others_opinions}\n\n"
                        f"请你针对其他人的观点进行回应，可以反驳、补充或修正自己的观点。"
                        f"给出你更新后的核心观点和论据。"
                    )

                self._log(f"\n  🎤 [{debater_name}] 发言中...")
                result = self._dispatch(debater_name, sub_task)
                round_opinions[debater_name] = result
                self._log_agent(debater_name, "观点", result)

                self.state.add_message(Message(
                    sender=debater_name,
                    receiver="all",
                    content=result,
                    msg_type=MessageType.RESULT,
                    metadata={"round": round_num},
                ))

            all_rounds.append(round_opinions)
            self._fire_hook("on_step_complete", step=round_num, state=self.state)

        # 裁决阶段
        self._log(f"\n{'─'*50}")
        self._log(f"  ⚖️  裁判裁决")
        self._log(f"{'─'*50}")

        self.state.status = "reviewing"
        final = self._judge(task, all_rounds)

        self.state.status = "done"
        self._log_header("Debate 完成")
        self._log(f"  {self.state.summary()}")

        return final

    def _judge(self, topic: str, all_rounds: list[dict[str, str]]) -> str:
        """裁判做最终裁决。"""
        rounds_summary = self._format_all_rounds(all_rounds)
        prompt = JUDGE_PROMPT.format(topic=topic, rounds_summary=rounds_summary)

        judge_name = self._judge_role.name
        self._log(f"\n  🔨 [{judge_name}] 正在裁决...")
        result = self._dispatch(judge_name, prompt)
        self._log_agent(judge_name, "裁决结果", result)
        return result

    def _format_opinions(
        self, all_rounds: list[dict[str, str]], exclude: str = ""
    ) -> str:
        """格式化历史观点（排除指定参与者）。"""
        lines = []
        for i, round_ops in enumerate(all_rounds, 1):
            for name, opinion in round_ops.items():
                if name != exclude:
                    lines.append(f"[第{i}轮 - {name}]:\n{opinion}\n")
        return "\n".join(lines) if lines else "(暂无其他观点)"

    def _format_all_rounds(self, all_rounds: list[dict[str, str]]) -> str:
        """格式化所有轮次的观点（给裁判看）。"""
        lines = []
        for i, round_ops in enumerate(all_rounds, 1):
            lines.append(f"=== 第 {i} 轮 ===")
            for name, opinion in round_ops.items():
                lines.append(f"\n[{name}]:\n{opinion}\n")
        return "\n".join(lines)
