"""Agent 间消息定义与纯数据结构

标准化的消息格式，用于 Agent 间通信和状态追踪。
也包含不依赖 LLM 的纯数据类（如 PipelineStep）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable


class MessageType(Enum):
    """消息类型枚举。"""

    TASK = "task"           # 任务分派
    RESULT = "result"       # 执行结果
    FEEDBACK = "feedback"   # 反馈/评审意见
    SYSTEM = "system"       # 系统消息（状态变更等）


@dataclass
class Message:
    """Agent 间通信的标准消息。"""

    sender: str                             # 发送者角色名
    receiver: str                           # 接收者角色名（"all" 表示广播）
    content: str                            # 消息内容
    msg_type: MessageType = MessageType.TASK
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def __repr__(self) -> str:
        content_preview = self.content[:60] + "..." if len(self.content) > 60 else self.content
        return (
            f"Message({self.sender} → {self.receiver}, "
            f"type={self.msg_type.value}, "
            f"content={content_preview!r})"
        )


@dataclass
class PipelineStep:
    """流水线步骤定义（纯数据类，不依赖 LLM）。

    模板中可用的占位符:
        {task}        - 原始任务
        {prev_result} - 上一步的输出
        {all_results} - 所有步骤的输出汇总
    """

    agent_name: str
    task_template: str = "{task}"
    retry: int = 1
    transform: Callable[[str], str] | None = None
