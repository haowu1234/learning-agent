"""Multi-Agent 协作基类

定义所有协作模式的统一接口和公共逻辑，包括：
- Agent 创建与管理
- 子任务分派
- Hook 机制
- 日志输出
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from src.agent.react import ReActAgent
from src.llm import LLMClient
from src.multi.message import Message, MessageType
from src.multi.roles import AgentRole
from src.multi.shared_state import SharedState
from src.tools.base import ToolRegistry


class BaseMultiAgent(ABC):
    """多 Agent 协作基类。所有协作模式必须继承此类。"""

    def __init__(
        self,
        llm: LLMClient,
        tool_registry: ToolRegistry,
        verbose: bool = True,
    ):
        """
        Args:
            llm: 共享的 LLM 客户端。
            tool_registry: 全局工具注册中心。
            verbose: 是否打印中间过程。
        """
        self.llm = llm
        self.global_tools = tool_registry
        self.verbose = verbose
        self.state = SharedState()
        self._agents: dict[str, ReActAgent] = {}
        self._hooks: dict[str, list[Callable[..., None]]] = {}

    # ================================================================
    # Agent 管理
    # ================================================================

    def add_agent(self, role: AgentRole, **agent_kwargs: Any) -> None:
        """根据角色定义创建并注册一个 Agent。

        Args:
            role: 角色定义。
            **agent_kwargs: 传递给 ReActAgent 的额外参数。
        """
        # 为该角色创建专属的工具注册中心（仅包含角色允许的工具）
        role_tools = ToolRegistry()
        for tool_name in role.tools:
            tool = self.global_tools.get(tool_name)
            if tool is not None:
                role_tools.register(tool)

        agent = ReActAgent(
            llm=self.llm,
            tool_registry=role_tools,
            mode=agent_kwargs.pop("mode", "function_calling"),
            max_steps=agent_kwargs.pop("max_steps", 10),
            verbose=False,  # 由 MultiAgent 统一控制日志
            system_prompt=self._build_system_prompt(role),
            available_skills=set(role.skills) if role.skills is not None else None,
        )
        agent._role_name = role.name
        agent._role_system_prompt = role.system_prompt
        self._agents[role.name] = agent

    def get_agent(self, name: str) -> ReActAgent | None:
        """获取指定名称的 Agent。"""
        return self._agents.get(name)

    @property
    def agent_names(self) -> list[str]:
        """所有已注册的 Agent 名称。"""
        return list(self._agents.keys())

    @staticmethod
    def _build_system_prompt(role: AgentRole) -> str:
        """将角色提示词与通用 Agent 指令合并为完整的 system prompt。"""
        from src.agent.prompt import SYSTEM_PROMPT_FUNCTION_CALLING

        return (
            f"{role.system_prompt}\n\n"
            f"{SYSTEM_PROMPT_FUNCTION_CALLING}"
        )

    # ================================================================
    # 核心接口
    # ================================================================

    @abstractmethod
    def run(self, task: str) -> str:
        """执行多 Agent 协作任务。

        Args:
            task: 用户的任务描述。

        Returns:
            最终结果。
        """

    # ================================================================
    # 子任务分派
    # ================================================================

    def _dispatch(self, agent_name: str, task: str) -> str:
        """将子任务分派给指定 Agent 执行。

        Args:
            agent_name: Agent 角色名。
            task: 子任务描述。

        Returns:
            Agent 的执行结果。
        """
        agent = self._agents.get(agent_name)
        if agent is None:
            error = f"错误：Agent '{agent_name}' 不存在。可用：{self.agent_names}"
            self._log(error)
            return error

        self._fire_hook("on_agent_start", agent_name=agent_name, task=task)
        self.state.add_message(Message(
            sender="system",
            receiver=agent_name,
            content=task,
            msg_type=MessageType.TASK,
        ))

        # 角色 prompt 已通过 system_prompt 参数注入，直接执行任务
        try:
            result = agent.run(task)
        except Exception as e:
            result = f"错误：Agent '{agent_name}' 执行失败：{e}"
            self._fire_hook("on_error", agent_name=agent_name, error=e)

        agent.reset()  # 每次执行后重置单 Agent 的对话历史
        self.state.add_result(agent_name, result)
        self._fire_hook("on_agent_finish", agent_name=agent_name, result=result)
        return result

    def _broadcast(self, message: Message) -> None:
        """向所有 Agent 广播消息。"""
        message.receiver = "all"
        self.state.add_message(message)

    # ================================================================
    # Hook 机制
    # ================================================================

    def on(self, event: str, callback: Callable[..., None]) -> None:
        """注册 hook 回调。

        支持的事件:
            - on_agent_start(agent_name, task)
            - on_agent_finish(agent_name, result)
            - on_step_complete(step, state)
            - on_error(agent_name, error)

        Args:
            event: 事件名。
            callback: 回调函数。
        """
        self._hooks.setdefault(event, []).append(callback)

    def _fire_hook(self, event: str, **kwargs: Any) -> None:
        """触发指定事件的所有回调。"""
        for callback in self._hooks.get(event, []):
            try:
                callback(**kwargs)
            except Exception:
                pass  # hook 异常不影响主流程

    # ================================================================
    # 日志
    # ================================================================

    def _log(self, message: str) -> None:
        """打印日志。"""
        if self.verbose:
            print(message)

    def _log_header(self, title: str) -> None:
        """打印区块标题。"""
        self._log(f"\n{'='*60}")
        self._log(f"  {title}")
        self._log(f"{'='*60}")

    def _log_agent(self, agent_name: str, action: str, detail: str = "") -> None:
        """打印 Agent 相关日志。"""
        msg = f"  [{agent_name}] {action}"
        if detail:
            preview = detail[:200] + "..." if len(detail) > 200 else detail
            msg += f": {preview}"
        self._log(msg)
