"""共享状态管理

所有 Agent 可读写的公共黑板，用于跨 Agent 的状态共享和追踪。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from src.multi.message import Message, MessageType


@dataclass
class SharedState:
    """多 Agent 协作的共享状态。"""

    task: str = ""
    plan: list[str] = field(default_factory=list)
    results: dict[str, str] = field(default_factory=dict)
    messages: list[Message] = field(default_factory=list)
    status: Literal["idle", "planning", "executing", "reviewing", "done", "failed"] = "idle"
    current_step: int = 0
    max_steps: int = 20
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(self, message: Message) -> None:
        """记录一条消息。"""
        self.messages.append(message)

    def add_result(self, agent_name: str, result: str) -> None:
        """记录某个 Agent 的执行结果。"""
        self.results[agent_name] = result
        self.add_message(Message(
            sender=agent_name,
            receiver="system",
            content=result,
            msg_type=MessageType.RESULT,
        ))

    def get_result(self, agent_name: str) -> str | None:
        """获取某个 Agent 的执行结果。"""
        return self.results.get(agent_name)

    def get_all_results(self) -> str:
        """获取所有 Agent 结果的格式化文本。"""
        if not self.results:
            return "(暂无结果)"
        lines = []
        for name, result in self.results.items():
            lines.append(f"[{name}] 的结果:\n{result}")
        return "\n\n".join(lines)

    def get_messages_for(self, receiver: str) -> list[Message]:
        """获取发给指定 Agent 的消息。"""
        return [
            m for m in self.messages
            if m.receiver == receiver or m.receiver == "all"
        ]

    def reset(self) -> None:
        """重置状态（保留 max_steps 配置）。"""
        self.task = ""
        self.plan.clear()
        self.results.clear()
        self.messages.clear()
        self.status = "idle"
        self.current_step = 0
        self.metadata.clear()

    def summary(self) -> str:
        """生成状态摘要。"""
        return (
            f"状态: {self.status} | "
            f"步骤: {self.current_step}/{self.max_steps} | "
            f"结果数: {len(self.results)} | "
            f"消息数: {len(self.messages)}"
        )

    def __repr__(self) -> str:
        return f"SharedState({self.summary()})"
