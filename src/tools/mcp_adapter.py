"""把 MCP server 暴露的远程工具适配为项目内的 Tool。"""

from __future__ import annotations

from typing import Any

from src.tools.base import Tool
from src.tools.mcp_client import MCPStdioClient


class MCPToolAdapter(Tool):
    """通用 MCP Tool 适配器。"""

    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any],
        mcp_client: MCPStdioClient,
        remote_name: str,
        argument_map: dict[str, str] | None = None,
    ) -> None:
        if not name.strip():
            raise ValueError("internal tool name 不能为空。")
        if not remote_name.strip():
            raise ValueError("remote tool name 不能为空。")

        self._name = name.strip()
        self._description = description.strip()
        self._parameters = parameters
        self.mcp_client = mcp_client
        self.remote_name = remote_name.strip()
        self.argument_map = dict(argument_map or {})

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    def backend_label(self) -> str:
        return f"{self.mcp_client.describe()}#{self.remote_name}"

    def run(self, **kwargs: Any) -> str:
        mapped_arguments = {
            self.argument_map.get(key, key): value
            for key, value in kwargs.items()
        }
        try:
            return self.mcp_client.call_tool(
                tool_name=self.remote_name,
                arguments=mapped_arguments,
            )
        except Exception as exc:
            return f"错误：MCP 工具 '{self.name}' 调用失败：{exc}"
