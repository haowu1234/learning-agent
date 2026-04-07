"""Plan-and-Execute 运行状态模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ToolTrace:
    """单次工具调用轨迹。"""

    tool_name: str
    tool_input: str
    observation: str


@dataclass
class StepRun:
    """单个计划步骤的结构化执行结果。"""

    step_id: str
    title: str
    task: str
    final_answer: str = ""
    status: Literal["pending", "running", "done", "failed", "reused"] = "pending"
    tool_traces: list[ToolTrace] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    reused_from_attempt: int | None = None


@dataclass
class VerificationResult:
    """Verifier 的结构化输出。"""

    passed: bool
    reason: str
    missing: list[str] = field(default_factory=list)
    suggested_fix: str = ""
    failure_type: str = "UNKNOWN"
    parser_failed: bool = False


@dataclass
class AgentRunResult:
    """一次 Agent 运行的结构化返回值。"""

    final_answer: str
    tool_traces: list[ToolTrace] = field(default_factory=list)
    step_runs: list[StepRun] = field(default_factory=list)
    raw_output: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class PlanExecutionState:
    """Plan-and-Execute 模式的跨轮状态。"""

    original_query: str
    working_query: str
    attempt: int = 1
    current_plan: list[dict[str, str]] = field(default_factory=list)
    completed_steps: list[StepRun] = field(default_factory=list)
    last_attempt_steps: list[StepRun] = field(default_factory=list)
    candidate_answer: str = ""
    verification: VerificationResult | None = None

    def find_completed_step(self, title: str, task: str) -> StepRun | None:
        """根据标题和任务查找可复用的已完成步骤。"""
        for step in self.completed_steps:
            if step.title == title and step.task == task and step.status in {"done", "reused"}:
                return step
        return None

    def upsert_completed_step(self, step: StepRun) -> None:
        """记录或覆盖一个已完成步骤。"""
        for index, existing in enumerate(self.completed_steps):
            if existing.title == step.title and existing.task == step.task:
                self.completed_steps[index] = step
                return
        self.completed_steps.append(step)
