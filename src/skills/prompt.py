from __future__ import annotations

from pathlib import Path

from src.skills.loader import load_skills


def _skill_mutability_label(category: str) -> str:
    return "[custom]" if category == "custom" else "[built-in]"


def get_skills_prompt_section(
    available_skills: set[str] | None = None,
    *,
    skills_path: str | Path | None = None,
) -> str:
    """构建 skills system prompt 片段。"""
    skills = load_skills(
        skills_path=skills_path,
        enabled_only=True,
        available_skills=available_skills,
    )
    if not skills:
        return ""

    skill_items = "\n".join(
        "    <skill>\n"
        f"        <name>{skill.name}</name>\n"
        f"        <description>{skill.description} {_skill_mutability_label(skill.category)}</description>\n"
        f"        <location>{skill.file_path}</location>\n"
        "    </skill>"
        for skill in skills
    )

    return f"""<skill_system>
你可以使用 skills 来处理特定类型的问题。每个 skill 都由一个 `SKILL.md` 主文件和同目录下的补充资源组成。

渐进加载规则：
1. 当用户需求与某个 skill 的描述匹配时，优先调用 `read_local_file` 读取该 skill 的 `SKILL.md`
2. 读完主文件后，再按其中指引按需读取同目录下的 `references/`、`templates/`、`scripts/` 或其他补充文件
3. 不要一次性把整个 skill 目录全部读入上下文，只在确实需要时继续展开
4. 严格遵循 skill 中给出的 workflow、输出格式和约束
5. 如果当前任务不匹配任何 skill，就按常规方式处理

<available_skills>
{skill_items}
</available_skills>
</skill_system>"""
