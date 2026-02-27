"""ReAct Agent 核心实现

支持两种模式：
1. function_calling：使用 OpenAI 原生 Function Calling（推荐）
2. text_parsing：纯文本解析模式，通过 Prompt + 正则提取工具调用
"""

from __future__ import annotations

import json
import re
from typing import Any

from src.agent.prompt import (
    SYSTEM_PROMPT_FUNCTION_CALLING,
    SYSTEM_PROMPT_TEXT_PARSING,
)
from src.llm import LLMClient
from src.memory.history import ConversationHistory
from src.tools.base import ToolRegistry


class ReActAgent:
    """ReAct Agent：推理（Reasoning）+ 行动（Acting）循环。"""

    def __init__(
        self,
        llm: LLMClient,
        tool_registry: ToolRegistry,
        mode: str = "function_calling",
        max_steps: int = 10,
        verbose: bool = True,
    ):
        """
        Args:
            llm: LLM 客户端实例。
            tool_registry: 工具注册中心。
            mode: 工具调用模式，'function_calling' 或 'text_parsing'。
            max_steps: 最大推理步数，防止无限循环。
            verbose: 是否打印中间推理过程。
        """
        self.llm = llm
        self.tools = tool_registry
        self.mode = mode
        self.max_steps = max_steps
        self.verbose = verbose
        self.history = ConversationHistory()

    def run(self, query: str) -> str:
        """运行 Agent 处理用户查询。

        Args:
            query: 用户的问题。

        Returns:
            Agent 的最终回答。
        """
        if self.mode == "function_calling":
            return self._run_function_calling(query)
        elif self.mode == "text_parsing":
            return self._run_text_parsing(query)
        else:
            raise ValueError(f"不支持的模式: {self.mode}，请使用 'function_calling' 或 'text_parsing'")

    # ================================================================
    # 模式 1：OpenAI Function Calling
    # ================================================================

    def _run_function_calling(self, query: str) -> str:
        """使用 OpenAI Function Calling 模式运行 Agent。"""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT_FUNCTION_CALLING},
        ]
        # 加入对话历史
        messages.extend(self.history.get_messages())
        messages.append({"role": "user", "content": query})

        openai_tools = self.tools.to_openai_tools()

        self._log(f"\n{'='*60}")
        self._log(f"用户问题: {query}")
        self._log(f"{'='*60}")

        for step in range(1, self.max_steps + 1):
            self._log(f"\n--- 第 {step} 步 ---")

            response = self.llm.chat(messages, tools=openai_tools if openai_tools else None)
            choice = response.choices[0]
            message = choice.message

            # 将 assistant 消息加入上下文
            assistant_msg: dict[str, Any] = {"role": "assistant"}
            if message.content:
                assistant_msg["content"] = message.content
            if message.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ]
            messages.append(assistant_msg)

            # 如果 LLM 没有调用工具，说明它准备给出最终答案
            if not message.tool_calls:
                final_answer = message.content or ""
                self._log(f"\n最终回答: {final_answer}")
                # 保存到对话历史
                self.history.add_user_message(query)
                self.history.add_assistant_message(final_answer)
                return final_answer

            # 处理工具调用
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                func_args = tool_call.function.arguments

                self._log(f"  Thought: {message.content or '(思考中...)'}")
                self._log(f"  Action: {func_name}")
                self._log(f"  Action Input: {func_args}")

                # 执行工具
                observation = self.tools.execute(func_name, func_args)
                self._log(f"  Observation: {observation}")

                # 将工具结果加入上下文
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": observation,
                })

        # 达到最大步数
        fallback = "抱歉，我在多次尝试后仍未能得出明确结论。请尝试简化问题或提供更多信息。"
        self.history.add_user_message(query)
        self.history.add_assistant_message(fallback)
        return fallback

    # ================================================================
    # 模式 2：纯文本解析
    # ================================================================

    def _run_text_parsing(self, query: str) -> str:
        """使用纯文本解析模式运行 Agent。

        LLM 被引导输出特定格式的文本，Agent 用正则表达式提取工具调用信息。
        这种方式不依赖 Function Calling 特性，兼容所有 LLM。
        """
        system_prompt = SYSTEM_PROMPT_TEXT_PARSING.format(
            tools_description=self.tools.get_tools_description()
        )

        # 构建初始上下文
        context = f"Question: {query}\n"

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]
        messages.extend(self.history.get_messages())
        messages.append({"role": "user", "content": context})

        self._log(f"\n{'='*60}")
        self._log(f"用户问题: {query}")
        self._log(f"{'='*60}")

        for step in range(1, self.max_steps + 1):
            self._log(f"\n--- 第 {step} 步 ---")

            response = self.llm.chat(messages)
            text = response.choices[0].message.content or ""
            self._log(f"  LLM 输出:\n{self._indent(text)}")

            # 检查是否有 Final Answer
            final_match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
            if final_match:
                final_answer = final_match.group(1).strip()
                self._log(f"\n最终回答: {final_answer}")
                self.history.add_user_message(query)
                self.history.add_assistant_message(final_answer)
                return final_answer

            # 提取 Action 和 Action Input
            action_match = re.search(r"Action:\s*(\w+)", text)
            input_match = re.search(r"Action Input:\s*({.*?})", text, re.DOTALL)

            if not action_match:
                # LLM 没有按格式输出，将其作为最终答案
                self.history.add_user_message(query)
                self.history.add_assistant_message(text)
                return text

            action_name = action_match.group(1)
            action_input = input_match.group(1) if input_match else "{}"

            self._log(f"  Action: {action_name}")
            self._log(f"  Action Input: {action_input}")

            # 执行工具
            observation = self.tools.execute(action_name, action_input)
            self._log(f"  Observation: {observation}")

            # 将结果拼入对话，让 LLM 继续推理
            messages.append({"role": "assistant", "content": text})
            messages.append({
                "role": "user",
                "content": f"Observation: {observation}\n\n请继续思考，或给出 Final Answer。",
            })

        fallback = "抱歉，我在多次尝试后仍未能得出明确结论。请尝试简化问题或提供更多信息。"
        self.history.add_user_message(query)
        self.history.add_assistant_message(fallback)
        return fallback

    # ================================================================
    # 辅助方法
    # ================================================================

    def reset(self) -> None:
        """重置对话历史。"""
        self.history.clear()

    def _log(self, message: str) -> None:
        """打印调试信息。"""
        if self.verbose:
            print(message)

    @staticmethod
    def _indent(text: str, prefix: str = "    ") -> str:
        """缩进文本。"""
        return "\n".join(f"{prefix}{line}" for line in text.split("\n"))
