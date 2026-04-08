from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from src.tools.base import Tool


class ReadLocalFileTool(Tool):
    """读取项目工作区内的本地文本文件。"""

    def __init__(
        self,
        *,
        project_root: str | Path | None = None,
        allowed_roots: Iterable[str | Path] | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[2]
        roots = list(allowed_roots) if allowed_roots is not None else [self.project_root]
        self.allowed_roots = [Path(root).resolve() for root in roots]

    @property
    def name(self) -> str:
        return "read_local_file"

    @property
    def description(self) -> str:
        return (
            "读取当前项目工作区内的本地文本文件。"
            "适合按路径读取 `SKILL.md`、参考文档、模板文件或其他本地资料。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "要读取的文件路径。支持项目内相对路径，或工作区内绝对路径。",
                },
                "start_line": {
                    "type": "integer",
                    "description": "从第几行开始读取，默认 1。",
                    "default": 1,
                    "minimum": 1,
                },
                "max_lines": {
                    "type": "integer",
                    "description": "最多读取多少行，默认 120，最大 400。",
                    "default": 120,
                    "minimum": 1,
                    "maximum": 400,
                },
            },
            "required": ["path"],
        }

    def run(
        self,
        path: str,
        start_line: int = 1,
        max_lines: int = 120,
        **_: Any,
    ) -> str:
        if start_line < 1:
            return "错误：start_line 必须大于等于 1。"
        if max_lines < 1:
            return "错误：max_lines 必须大于等于 1。"
        if max_lines > 400:
            return "错误：max_lines 不能超过 400。"

        resolved = self._resolve_path(path)
        if resolved is None:
            return "错误：路径不能为空。"
        if not self._is_allowed_path(resolved):
            return f"错误：不允许访问工作区外的路径：{resolved}"
        if not resolved.exists() or not resolved.is_file():
            return f"错误：文件不存在或不是普通文件：{resolved}"
        if resolved.stat().st_size > 1024 * 1024:
            return f"错误：文件过大，暂不支持直接读取超过 1MB 的文件：{resolved}"

        try:
            content = resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"错误：当前工具只支持读取 UTF-8 文本文件：{resolved}"
        except OSError as exc:
            return f"错误：读取文件失败：{exc}"

        lines = content.splitlines()
        if not lines:
            return f"文件：{resolved}\n内容：文件为空。"
        if start_line > len(lines):
            return f"错误：start_line={start_line} 超出文件总行数 {len(lines)}。"

        start_index = start_line - 1
        end_index = min(len(lines), start_index + max_lines)
        numbered = "\n".join(
            f"{line_no:>4}: {line}"
            for line_no, line in enumerate(lines[start_index:end_index], start=start_line)
        )
        suffix = "\n(内容已截断，可调整 start_line / max_lines 继续读取。)" if end_index < len(lines) else ""
        return (
            f"文件：{resolved}\n"
            f"行范围：{start_line}-{end_index} / 共 {len(lines)} 行\n\n"
            f"{numbered}{suffix}"
        )

    def _resolve_path(self, path: str) -> Path | None:
        candidate = path.strip()
        if not candidate:
            return None
        raw_path = Path(candidate)
        if raw_path.is_absolute():
            return raw_path.resolve()
        return (self.project_root / raw_path).resolve()

    def _is_allowed_path(self, path: Path) -> bool:
        for root in self.allowed_roots:
            try:
                path.relative_to(root)
                return True
            except ValueError:
                continue
        return False
