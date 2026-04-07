"""LLM 调用封装模块。

支持 OpenAI API 及所有兼容接口（DeepSeek、通义千问、vLLM 等）。
通过 .env 文件或环境变量配置 API Key、Base URL 和模型名称。
默认优先连接本地 vLLM OpenAI 兼容端点，方便离线实验。
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletion

load_dotenv()

DEFAULT_OPENAI_BASE_URL = "http://localhost:8002/v1"
DEFAULT_OPENAI_MODEL = "openai/gpt-oss-20b"
LOCAL_API_KEY_PLACEHOLDER = "local-vllm"
LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0"}


class LLMClient:
    """LLM 客户端，封装 OpenAI ChatCompletion 调用。"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL
        resolved_model = model or os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not resolved_api_key and self._is_local_base_url(resolved_base_url):
            resolved_api_key = LOCAL_API_KEY_PLACEHOLDER

        self.api_key = resolved_api_key or ""
        self.base_url = resolved_base_url
        self.model = resolved_model

        if not self.api_key:
            raise ValueError(
                "未设置 API Key。请在 .env 文件中设置 OPENAI_API_KEY，"
                "或通过参数传入 api_key。若使用默认本地 vLLM 端点，则会自动使用占位 key。"
            )

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> ChatCompletion:
        """发送对话请求到 LLM。

        Args:
            messages: 对话消息列表，格式为 OpenAI messages 格式。
            tools: 可选的工具定义列表（OpenAI Function Calling 格式）。
            temperature: 生成温度，越高越随机。
            max_tokens: 最大生成 token 数。

        Returns:
            OpenAI ChatCompletion 响应对象。
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        return self.client.chat.completions.create(**kwargs)

    def chat_simple(self, prompt: str, system: str = "") -> str:
        """简单的单轮对话，返回文本内容。"""
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.chat(messages)
        return response.choices[0].message.content or ""

    @staticmethod
    def _is_local_base_url(base_url: str) -> bool:
        """判断 base_url 是否指向本地 OpenAI 兼容端点。"""
        host = urlparse(base_url).hostname
        return host in LOCAL_HOSTS
