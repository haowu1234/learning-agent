"""基于 Python MCP SDK 的 stdio 客户端封装。"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
from contextlib import AsyncExitStack
from threading import Thread
from typing import Any


class MCPStdioClient:
    """通过 Python MCP SDK 调用本地 stdio MCP Server。"""

    def __init__(
        self,
        *,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        if not command.strip():
            raise ValueError("MCP server command 不能为空。")
        self.command = command.strip()
        self.args = list(args or [])
        self.env = dict(env or {})

    @classmethod
    def from_env(cls, prefix: str = "MCP_SEARCH_") -> MCPStdioClient | None:
        """从环境变量读取 MCP stdio server 配置。"""
        command = os.getenv(f"{prefix}SERVER_COMMAND", "").strip()
        if not command:
            return None

        raw_args = os.getenv(f"{prefix}SERVER_ARGS", "").strip()
        raw_env = os.getenv(f"{prefix}SERVER_ENV", "").strip()
        args = shlex.split(raw_args) if raw_args else []
        env = cls._parse_env_pairs(raw_env)
        return cls(command=command, args=args, env=env)

    def describe(self) -> str:
        command_line = " ".join([self.command, *self.args]).strip()
        return f"mcp({command_line})"

    def call_tool(
        self,
        *,
        arguments: dict[str, Any],
        tool_name: str | None = None,
    ) -> str:
        """调用 MCP server 暴露的工具并返回可读文本。"""
        return self._run_async(self._call_tool(arguments=arguments, tool_name=tool_name))

    async def _call_tool(
        self,
        *,
        arguments: dict[str, Any],
        tool_name: str | None,
    ) -> str:
        ClientSession, StdioServerParameters, stdio_client = self._import_sdk()

        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self._build_child_env(),
        )

        async with AsyncExitStack() as stack:
            read_stream, write_stream = await stack.enter_async_context(stdio_client(server_params))
            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()

            tools_response = await session.list_tools()
            resolved_tool_name = self._resolve_tool_name(tools_response, requested_name=tool_name)
            result = await session.call_tool(resolved_tool_name, arguments)

        return self._format_call_result(result)

    def _build_child_env(self) -> dict[str, str] | None:
        if not self.env:
            return None
        merged = os.environ.copy()
        merged.update(self.env)
        return merged

    @staticmethod
    def _parse_env_pairs(raw_env: str) -> dict[str, str]:
        if not raw_env:
            return {}

        parsed: dict[str, str] = {}
        for token in shlex.split(raw_env):
            if "=" not in token:
                raise ValueError(
                    "MCP_SEARCH_SERVER_ENV 格式错误，请使用 'KEY=value OTHER=value'。"
                )
            key, value = token.split("=", 1)
            key = key.strip()
            if not key:
                raise ValueError("MCP_SEARCH_SERVER_ENV 中存在空的环境变量名。")
            parsed[key] = value
        return parsed

    @staticmethod
    def _import_sdk() -> tuple[Any, Any, Any]:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:  # pragma: no cover - 依赖缺失时走错误提示
            raise RuntimeError(
                "未安装 Python MCP SDK，请先执行 `pip install mcp` 或安装 requirements.txt 中新增的依赖。"
            ) from exc
        return ClientSession, StdioServerParameters, stdio_client

    @classmethod
    def _resolve_tool_name(cls, tools_response: Any, requested_name: str | None) -> str:
        tool_names = cls._extract_tool_names(tools_response)
        if not tool_names:
            raise RuntimeError("MCP server 未暴露任何工具。")

        if requested_name:
            if requested_name not in tool_names:
                raise RuntimeError(
                    f"找不到 MCP 工具 '{requested_name}'。可用工具：{', '.join(tool_names)}"
                )
            return requested_name

        exact_matches = [name for name in tool_names if name.lower() == "search"]
        if len(exact_matches) == 1:
            return exact_matches[0]

        for keyword in ("search", "web", "duckduckgo", "ddg"):
            matches = [name for name in tool_names if keyword in name.lower()]
            if len(matches) == 1:
                return matches[0]

        if len(tool_names) == 1:
            return tool_names[0]

        raise RuntimeError(
            "未指定 MCP_SEARCH_TOOL_NAME，且无法自动判断搜索工具。"
            f"可用工具：{', '.join(tool_names)}"
        )

    @staticmethod
    def _extract_tool_names(tools_response: Any) -> list[str]:
        tools = getattr(tools_response, "tools", None) or []
        names: list[str] = []
        for tool in tools:
            name = getattr(tool, "name", "")
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
        return names

    @classmethod
    def _format_call_result(cls, result: Any) -> str:
        is_error = bool(
            getattr(result, "isError", False) or getattr(result, "is_error", False)
        )
        content = getattr(result, "content", None) or []
        parts: list[str] = []

        for item in content:
            text = getattr(item, "text", None)
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
                continue

            serialized = cls._serialize(item)
            if serialized is None:
                continue
            if isinstance(serialized, str):
                parts.append(serialized)
            else:
                parts.append(json.dumps(serialized, ensure_ascii=False, indent=2))

        if not parts:
            serialized = cls._serialize(result)
            if serialized is None:
                serialized = "MCP 工具执行成功，但未返回可显示内容。"
            if isinstance(serialized, str):
                parts.append(serialized)
            else:
                parts.append(json.dumps(serialized, ensure_ascii=False, indent=2))

        rendered = "\n".join(part for part in parts if part).strip()
        if is_error:
            raise RuntimeError(rendered or "MCP 工具返回错误结果。")
        return rendered or "MCP 工具执行成功，但未返回可显示内容。"

    @staticmethod
    def _serialize(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool, list, dict)):
            return value

        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            return model_dump()

        dict_method = getattr(value, "dict", None)
        if callable(dict_method):
            return dict_method()

        if hasattr(value, "__dict__"):
            return {
                key: val
                for key, val in vars(value).items()
                if not key.startswith("_")
            }

        return str(value)

    @staticmethod
    def _run_async(coro: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        result: dict[str, Any] = {}
        error: dict[str, BaseException] = {}

        def runner() -> None:
            try:
                result["value"] = asyncio.run(coro)
            except BaseException as exc:  # pragma: no cover - 线程内异常回传
                error["value"] = exc

        thread = Thread(target=runner, daemon=True)
        thread.start()
        thread.join()

        if "value" in error:
            raise error["value"]
        return result.get("value")
