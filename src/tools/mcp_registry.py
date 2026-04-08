"""MCP 工具配置加载与注册。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.tools.base import Tool
from src.tools.mcp_adapter import MCPToolAdapter
from src.tools.mcp_client import MCPStdioClient
from src.tools.search import SearchTool

DEFAULT_MCP_CONFIG_ENV = "MCP_SERVERS_CONFIG"
DEFAULT_MCP_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "mcp_servers.json"


def get_mcp_config_path(config_path: str | Path | None = None) -> Path:
    """解析 MCP 配置文件路径。"""
    if config_path is not None:
        return Path(config_path).expanduser().resolve()

    env_path = os.getenv(DEFAULT_MCP_CONFIG_ENV, "").strip()
    if env_path:
        return Path(env_path).expanduser().resolve()

    return DEFAULT_MCP_CONFIG_PATH


def load_mcp_tools(config_path: str | Path | None = None) -> list[Tool]:
    """从 JSON 配置中加载启用的 MCP 工具。"""
    path = get_mcp_config_path(config_path)
    if not path.exists():
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))
    servers = payload.get("servers", [])
    if not isinstance(servers, list):
        raise ValueError("MCP 配置格式错误：servers 必须是数组。")

    tools: list[Tool] = []
    for server in servers:
        if not isinstance(server, dict):
            raise ValueError("MCP 配置格式错误：server 项必须是对象。")
        if not server.get("enabled", False):
            continue
        tools.extend(_build_server_tools(server))
    return tools


def _build_server_tools(server: dict[str, Any]) -> list[Tool]:
    transport = str(server.get("transport", "stdio")).strip().lower()
    if transport != "stdio":
        raise ValueError(f"暂不支持的 MCP transport: {transport}")

    command = str(server.get("command", "")).strip()
    if not command:
        raise ValueError("MCP server 配置缺少 command。")

    args = server.get("args", [])
    if not isinstance(args, list) or any(not isinstance(item, str) for item in args):
        raise ValueError("MCP server 配置错误：args 必须是字符串数组。")

    env = server.get("env", {})
    if not isinstance(env, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in env.items()):
        raise ValueError("MCP server 配置错误：env 必须是字符串键值对对象。")

    client = MCPStdioClient(command=command, args=args, env=env)
    configured_tools = server.get("tools", [])
    if not isinstance(configured_tools, list):
        raise ValueError("MCP server 配置错误：tools 必须是数组。")

    return [_build_tool(tool_config, client) for tool_config in configured_tools if tool_config.get("enabled", True)]


def _build_tool(tool_config: dict[str, Any], client: MCPStdioClient) -> Tool:
    if not isinstance(tool_config, dict):
        raise ValueError("MCP tool 配置错误：每个 tool 必须是对象。")

    internal_name = str(tool_config.get("internal_name", "")).strip()
    remote_name = str(tool_config.get("remote_name", "")).strip()
    if not internal_name:
        raise ValueError("MCP tool 配置缺少 internal_name。")
    if not remote_name:
        raise ValueError("MCP tool 配置缺少 remote_name。")

    argument_map = tool_config.get("argument_map", {})
    if not isinstance(argument_map, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in argument_map.items()):
        raise ValueError("MCP tool 配置错误：argument_map 必须是字符串映射。")

    if internal_name == "search":
        query_argument = argument_map.get("query", "query")
        return SearchTool(
            backend="mcp",
            mcp_client=client,
            mcp_tool_name=remote_name,
            query_argument=query_argument,
        )

    description = str(tool_config.get("description", "")).strip()
    parameters = tool_config.get("parameters")
    if not description:
        raise ValueError(f"MCP tool '{internal_name}' 配置缺少 description。")
    if not isinstance(parameters, dict):
        raise ValueError(f"MCP tool '{internal_name}' 配置缺少 parameters。")

    return MCPToolAdapter(
        name=internal_name,
        description=description,
        parameters=parameters,
        mcp_client=client,
        remote_name=remote_name,
        argument_map=argument_map,
    )
