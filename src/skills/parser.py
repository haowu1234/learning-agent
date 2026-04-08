from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from src.skills.types import Skill

logger = logging.getLogger(__name__)

_FRONT_MATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*(?:\n|$)", re.DOTALL)


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _finalize_multiline_value(lines: list[str], style: str) -> str:
    text = "\n".join(lines).rstrip()
    if style == "|":
        return text
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text)


def parse_frontmatter_text(frontmatter_text: str) -> dict[str, Any]:
    """解析一个极简 YAML frontmatter。

    当前主要支持：
    - `key: value`
    - `description: >` / `description: |`
    - 顶层简单键值
    """
    metadata: dict[str, Any] = {}
    lines = frontmatter_text.split("\n")

    current_key: str | None = None
    current_value: list[str] = []
    multiline_style: str | None = None
    indent_level: int | None = None

    for line in lines:
        if current_key is not None:
            if not line.strip():
                current_value.append("")
                continue

            current_indent = len(line) - len(line.lstrip())
            if indent_level is None and current_indent > 0:
                indent_level = current_indent
                current_value.append(line[indent_level:])
                continue
            if indent_level is not None and current_indent >= indent_level:
                current_value.append(line[indent_level:])
                continue

            metadata[current_key] = _finalize_multiline_value(
                current_value,
                multiline_style or ">",
            )
            current_key = None
            current_value = []
            multiline_style = None
            indent_level = None

        if not line.strip():
            continue
        if line.startswith((" ", "\t")):
            # 嵌套字段（如 metadata 下的子键）当前不做完整解析，直接跳过。
            continue
        if ":" not in line:
            continue

        key, raw_value = line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()

        if value in {">", "|", ">-", "|-"}:
            current_key = key
            multiline_style = value[0]
            current_value = []
            indent_level = None
            continue

        metadata[key] = _strip_quotes(value)

    if current_key is not None:
        metadata[current_key] = _finalize_multiline_value(
            current_value,
            multiline_style or ">",
        )

    return metadata


def parse_skill_file(
    skill_file: Path,
    category: str,
    relative_path: Path | None = None,
) -> Skill | None:
    """解析 `SKILL.md` 并抽取最关键的 skill 元数据。"""
    if not skill_file.exists() or skill_file.name != "SKILL.md":
        return None

    try:
        content = skill_file.read_text(encoding="utf-8")
    except OSError as exc:
        logger.error("读取 skill 文件失败 %s: %s", skill_file, exc)
        return None

    match = _FRONT_MATTER_PATTERN.match(content)
    if not match:
        return None

    metadata = parse_frontmatter_text(match.group(1))
    name = str(metadata.get("name") or "").strip()
    description = str(metadata.get("description") or "").strip()
    if not name or not description:
        return None

    return Skill(
        name=name,
        description=description,
        skill_dir=skill_file.parent,
        skill_file=skill_file,
        relative_path=relative_path or Path(skill_file.parent.name),
        category=category,
        enabled=True,
        metadata=metadata,
    )
