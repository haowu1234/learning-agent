from __future__ import annotations

import os
from pathlib import Path

from src.skills.parser import parse_skill_file
from src.skills.types import Skill


def get_skills_root_path() -> Path:
    """返回项目根目录下的 `skills/`。"""
    return Path(__file__).resolve().parents[2] / "skills"


def load_skills(
    skills_path: str | Path | None = None,
    *,
    enabled_only: bool = False,
    available_skills: set[str] | None = None,
) -> list[Skill]:
    """递归扫描 `skills/public` 与 `skills/custom` 并加载 skills。"""
    resolved_root = Path(skills_path).resolve() if skills_path else get_skills_root_path()
    if not resolved_root.exists():
        return []

    skills_by_name: dict[str, Skill] = {}
    for category in ("public", "custom"):
        category_path = resolved_root / category
        if not category_path.exists() or not category_path.is_dir():
            continue

        for current_root, dir_names, file_names in os.walk(category_path, followlinks=True):
            dir_names[:] = sorted(name for name in dir_names if not name.startswith("."))
            if "SKILL.md" not in file_names:
                continue

            skill_file = Path(current_root) / "SKILL.md"
            relative_path = skill_file.parent.relative_to(category_path)
            skill = parse_skill_file(skill_file, category=category, relative_path=relative_path)
            if skill is None:
                continue
            skills_by_name[skill.name] = skill

    skills = list(skills_by_name.values())
    if enabled_only:
        skills = [skill for skill in skills if skill.enabled]
    if available_skills is not None:
        skills = [skill for skill in skills if skill.name in available_skills]
    skills.sort(key=lambda item: item.name)
    return skills
