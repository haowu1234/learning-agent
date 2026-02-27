"""LLM 调用封装模块

支持 OpenAI API 及所有兼容接口（DeepSeek、通义千问等）。
通过 .env 文件或环境变量配置 API Key、Base URL 和模型名称。
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletion

load_dotenv()


class LLMClient:
    """LLM 客户端，封装 OpenAI ChatCompletion 调用。"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        if not self.api_key:
            raise ValueError(
                "未设置 API Key。请在 .env 文件中设置 OPENAI_API_KEY，"
                "或通过参数传入 api_key。"
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
        """简单的单轮对话，返回文本内容。

        Args:
            prompt: 用户消息。
            system: 可选的系统消息。

        Returns:
            LLM 回复的文本内容。
        """
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.chat(messages)
        return response.choices[0].message.content or ""
