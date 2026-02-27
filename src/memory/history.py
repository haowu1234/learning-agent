"""对话历史管理模块

管理多轮对话的消息历史，支持：
- 添加用户/助手消息
- 获取消息列表
- 按最大轮数截断历史
"""

from __future__ import annotations

from typing import Any


class ConversationHistory:
    """对话历史管理器。"""

    def __init__(self, max_turns: int = 20):
        """
        Args:
            max_turns: 保留的最大对话轮数（一轮 = 一问一答）。
                      超过时丢弃最早的对话。
        """
        self.max_turns = max_turns
        self._messages: list[dict[str, Any]] = []

    def add_user_message(self, content: str) -> None:
        """添加用户消息。"""
        self._messages.append({"role": "user", "content": content})
        self._truncate()

    def add_assistant_message(self, content: str) -> None:
        """添加助手消息。"""
        self._messages.append({"role": "assistant", "content": content})
        self._truncate()

    def get_messages(self) -> list[dict[str, Any]]:
        """获取消息历史副本。"""
        return list(self._messages)

    def clear(self) -> None:
        """清空对话历史。"""
        self._messages.clear()

    def _truncate(self) -> None:
        """按最大轮数截断历史。

        保留最新的 max_turns 轮对话（每轮 2 条消息）。
        """
        max_messages = self.max_turns * 2
        if len(self._messages) > max_messages:
            self._messages = self._messages[-max_messages:]

    @property
    def turn_count(self) -> int:
        """当前对话轮数。"""
        return len(self._messages) // 2

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        return f"ConversationHistory(turns={self.turn_count}, messages={len(self)})"
