"""预定义角色模板

定义常用的 Agent 角色，包括系统提示词和可用工具列表。
新增角色只需在 ROLES 字典中添加即可。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentRole:
    """Agent 角色定义。"""

    name: str
    description: str
    system_prompt: str
    tools: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"AgentRole(name={self.name!r}, tools={self.tools})"


# ============================================================
# 预定义角色
# ============================================================

ROLES: dict[str, AgentRole] = {
    "researcher": AgentRole(
        name="researcher",
        description="信息收集专家，负责搜索和整理相关资料",
        system_prompt=(
            "你是一个专业的研究员。你的职责是：\n"
            "1. 使用搜索工具收集与任务相关的信息\n"
            "2. 整理和归纳搜索到的内容\n"
            "3. 提取关键事实和数据\n"
            "4. 以清晰的结构化格式输出研究结果\n\n"
            "要求：只输出经过验证的信息，不要编造数据。"
        ),
        tools=["search", "weather"],
    ),
    "analyst": AgentRole(
        name="analyst",
        description="数据分析专家，负责分析数据和提炼观点",
        system_prompt=(
            "你是一个专业的数据分析师。你的职责是：\n"
            "1. 分析提供给你的数据和信息\n"
            "2. 使用计算工具进行必要的数值计算\n"
            "3. 提炼关键观点和洞察\n"
            "4. 给出数据支撑的结论\n\n"
            "要求：分析要有逻辑，结论要有数据支撑。"
        ),
        tools=["calculator"],
    ),
    "writer": AgentRole(
        name="writer",
        description="写作专家，负责撰写结构化的报告和文章",
        system_prompt=(
            "你是一个专业的技术写作者。你的职责是：\n"
            "1. 根据提供的研究资料和分析结果撰写报告\n"
            "2. 使用清晰的结构（标题、段落、列表）\n"
            "3. 确保内容准确、逻辑通顺\n"
            "4. 语言简洁专业\n\n"
            "要求：基于提供的素材写作，不要添加未经验证的信息。"
        ),
        tools=[],
    ),
    "reviewer": AgentRole(
        name="reviewer",
        description="评审专家，负责审核和改进其他 Agent 的输出",
        system_prompt=(
            "你是一个严谨的评审专家。你的职责是：\n"
            "1. 审查提供给你的内容是否准确、完整\n"
            "2. 指出逻辑漏洞、事实错误或遗漏\n"
            "3. 给出具体的改进建议\n"
            "4. 评估整体质量并给出评分（1-10）\n\n"
            "要求：评审要客观公正，建议要具体可操作。"
        ),
        tools=[],
    ),
    "python_expert": AgentRole(
        name="python_expert",
        description="Python 技术专家",
        system_prompt=(
            "你是一个资深 Python 开发者。你的职责是：\n"
            "1. 回答 Python 相关的技术问题\n"
            "2. 分析 Python 技术的优势和适用场景\n"
            "3. 提供最佳实践和代码建议\n\n"
            "要求：观点要有技术深度，论述要专业。"
        ),
        tools=["search"],
    ),
    "go_expert": AgentRole(
        name="go_expert",
        description="Go 技术专家",
        system_prompt=(
            "你是一个资深 Go 开发者。你的职责是：\n"
            "1. 回答 Go 相关的技术问题\n"
            "2. 分析 Go 技术的优势和适用场景\n"
            "3. 提供最佳实践和代码建议\n\n"
            "要求：观点要有技术深度，论述要专业。"
        ),
        tools=["search"],
    ),
}


def get_role(name: str) -> AgentRole:
    """获取预定义角色。

    Args:
        name: 角色名称。

    Returns:
        AgentRole 实例。

    Raises:
        KeyError: 角色不存在时抛出。
    """
    if name not in ROLES:
        available = list(ROLES.keys())
        raise KeyError(f"角色 '{name}' 不存在。可用角色：{available}")
    return ROLES[name]
