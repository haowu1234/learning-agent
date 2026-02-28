"""Multi-Agent 协作模块

提供三种协作模式：
- PipelineMultiAgent: 流水线模式
- OrchestratorMultiAgent: 编排者模式
- DebateMultiAgent: 辩论模式
"""


def __getattr__(name: str):
    """延迟导入，避免在不需要时拉入 LLM 等重依赖。"""
    if name == "PipelineMultiAgent":
        from src.multi.pipeline import PipelineMultiAgent
        return PipelineMultiAgent
    elif name == "OrchestratorMultiAgent":
        from src.multi.orchestrator import OrchestratorMultiAgent
        return OrchestratorMultiAgent
    elif name == "DebateMultiAgent":
        from src.multi.debate import DebateMultiAgent
        return DebateMultiAgent
    raise AttributeError(f"module 'src.multi' has no attribute {name!r}")


__all__ = [
    "PipelineMultiAgent",
    "OrchestratorMultiAgent",
    "DebateMultiAgent",
]
