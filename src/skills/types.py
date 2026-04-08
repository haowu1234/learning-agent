from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Skill:
    """运行时 Skill 元数据。"""

    name: str
    description: str
    skill_dir: Path
    skill_file: Path
    relative_path: Path
    category: str
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def skill_path(self) -> str:
        """返回 skills/{category} 目录下的相对路径。"""
        path = self.relative_path.as_posix()
        return "" if path == "." else path

    @property
    def file_path(self) -> str:
        """返回 SKILL.md 的绝对路径字符串。"""
        return str(self.skill_file.resolve())

    def __repr__(self) -> str:
        return (
            "Skill("
            f"name={self.name!r}, "
            f"category={self.category!r}, "
            f"relative_path={self.relative_path.as_posix()!r}"
            ")"
        )
