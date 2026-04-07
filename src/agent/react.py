"""Agent 核心实现。

支持三种模式：
1. function_calling：使用 OpenAI 原生 Function Calling（推荐）
2. text_parsing：纯文本解析模式，通过 Prompt + 正则提取工具调用
3. plan_and_execute：先规划任务步骤，再逐步执行、验证、反思并重规划
"""

from __future__ import annotations

import json
import re
from typing import Any

from src.agent.prompt import (
    SYSTEM_PROMPT_FUNCTION_CALLING,
    SYSTEM_PROMPT_PLAN_AND_EXECUTE_PLANNER,
    SYSTEM_PROMPT_PLAN_AND_EXECUTE_REFLECTOR,
    SYSTEM_PROMPT_PLAN_AND_EXECUTE_SUMMARIZER,
    SYSTEM_PROMPT_PLAN_AND_EXECUTE_VERIFIER,
    SYSTEM_PROMPT_TEXT_PARSING,
)
from src.llm import LLMClient
from src.memory.history import ConversationHistory
from src.tools.base import ToolRegistry

SUPPORTED_MODES = ("function_calling", "text_parsing", "plan_and_execute")
SUPPORTED_EXECUTOR_MODES = ("function_calling", "text_parsing")
DEFAULT_FALLBACK_MESSAGE = "抱歉，我在多次尝试后仍未能得出明确结论。请尝试简化问题或提供更多信息。"


class ReActAgent:
    """Agent：支持直接 ReAct 与 Plan-and-Execute 两类策略。"""

    def __init__(
        self,
        llm: LLMClient,
        tool_registry: ToolRegistry,
        mode: str = "function_calling",
        max_steps: int = 10,
        verbose: bool = True,
        system_prompt: str | None = None,
        executor_mode: str | None = None,
        max_plan_steps: int = 5,
        max_replans: int = 1,
        enable_verifier: bool = True,
    ):
        """
        Args:
            llm: LLM 客户端实例。
            tool_registry: 工具注册中心。
            mode: Agent 运行模式，支持 'function_calling'、'text_parsing'、
                'plan_and_execute'。
            max_steps: 执行阶段最大推理步数，防止无限循环。
            verbose: 是否打印中间推理过程。
            system_prompt: 自定义系统提示词，为 None 时使用默认模板。
            executor_mode: 在 plan_and_execute 模式下，执行器使用的模式。
                默认沿用 function_calling。
            max_plan_steps: plan_and_execute 模式下最多规划多少步。
            max_replans: 验证未通过时，最多允许重新规划多少次。
            enable_verifier: 是否在 plan_and_execute 模式下启用结果验收。
        """
        self.llm = llm
        self.tools = tool_registry
        self.mode = mode
        self.max_steps = max_steps
        self.verbose = verbose
        self.system_prompt = system_prompt
        self.executor_mode = executor_mode or "function_calling"
        self.max_plan_steps = max_plan_steps
        self.max_replans = max_replans
        self.enable_verifier = enable_verifier
        self.history = ConversationHistory()

    def run(self, query: str) -> str:
        """运行 Agent 处理用户查询。"""
        self._validate_mode_config()

        if self.mode == "function_calling":
            return self._run_function_calling(query)
        if self.mode == "text_parsing":
            return self._run_text_parsing(query)
        if self.mode == "plan_and_execute":
            return self._run_plan_and_execute(query)

        raise ValueError(
            f"不支持的模式: {self.mode}，请使用 {', '.join(repr(mode) for mode in SUPPORTED_MODES)}"
        )

    @classmethod
    def available_modes(cls) -> tuple[str, ...]:
        """返回支持的运行模式列表。"""
        return SUPPORTED_MODES

    # ================================================================
    # 模式 1：OpenAI Function Calling
    # ================================================================

    def _run_function_calling(self, query: str) -> str:
        """使用 OpenAI Function Calling 模式运行 Agent。"""
        sys_prompt = self.system_prompt or SYSTEM_PROMPT_FUNCTION_CALLING
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": sys_prompt},
        ]
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

            if not message.tool_calls:
                final_answer = message.content or ""
                self._log(f"\n最终回答: {final_answer}")
                self._save_to_history(query, final_answer)
                return final_answer

            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                func_args = tool_call.function.arguments

                self._log(f"  Thought: {message.content or '(思考中...)'}")
                self._log(f"  Action: {func_name}")
                self._log(f"  Action Input: {func_args}")

                observation = self.tools.execute(func_name, func_args)
                self._log(f"  Observation: {observation}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": observation,
                })

        self._save_to_history(query, DEFAULT_FALLBACK_MESSAGE)
        return DEFAULT_FALLBACK_MESSAGE

    # ================================================================
    # 模式 2：纯文本解析
    # ================================================================

    def _run_text_parsing(self, query: str) -> str:
        """使用纯文本解析模式运行 Agent。"""
        system_prompt = self._build_text_parsing_system_prompt()
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

            final_match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
            if final_match:
                final_answer = final_match.group(1).strip()
                self._log(f"\n最终回答: {final_answer}")
                self._save_to_history(query, final_answer)
                return final_answer

            action_match = re.search(r"Action:\s*(\w+)", text)
            input_match = re.search(r"Action Input:\s*({.*?})", text, re.DOTALL)

            if not action_match:
                self._save_to_history(query, text)
                return text

            action_name = action_match.group(1)
            action_input = input_match.group(1) if input_match else "{}"

            self._log(f"  Action: {action_name}")
            self._log(f"  Action Input: {action_input}")

            observation = self.tools.execute(action_name, action_input)
            self._log(f"  Observation: {observation}")

            messages.append({"role": "assistant", "content": text})
            messages.append({
                "role": "user",
                "content": f"Observation: {observation}\n\n请继续思考，或给出 Final Answer。",
            })

        self._save_to_history(query, DEFAULT_FALLBACK_MESSAGE)
        return DEFAULT_FALLBACK_MESSAGE

    # ================================================================
    # 模式 3：Plan-and-Execute
    # ================================================================

    def _run_plan_and_execute(self, query: str) -> str:
        """先规划，再逐步执行、验证、反思并在必要时重规划。"""
        self._log(f"\n{'='*60}")
        self._log(f"用户问题: {query}")
        self._log("模式: Plan-and-Execute")
        self._log(f"执行器模式: {self.executor_mode}")
        self._log(f"Verifier: {'开启' if self.enable_verifier else '关闭'}")
        self._log(f"最大重规划次数: {self.max_replans}")
        self._log(f"{'='*60}")

        working_query = query
        total_attempts = self.max_replans + 1
        last_answer = DEFAULT_FALLBACK_MESSAGE

        for attempt in range(1, total_attempts + 1):
            self._log(f"\n=== Plan-and-Execute 尝试 {attempt}/{total_attempts} ===")
            if attempt > 1:
                self._log("🔁 根据上轮复盘重新规划...")
                self._log(self._indent(working_query))

            plan = self._plan(working_query)
            results: list[dict[str, str]] = []

            if not plan:
                self._log("\n⚠️  规划失败，回退到直接执行。")
                final_answer = self._run_sub_agent(working_query, self.executor_mode)
            else:
                self._log_plan(plan)
                results = self._execute_plan_steps(working_query, plan)
                final_answer = self._summarize_plan_execution(query, working_query, plan, results)

            last_answer = final_answer
            self._log(f"\n候选回答: {final_answer}")

            verification = self._verify_plan_execution(
                original_query=query,
                working_query=working_query,
                plan=plan,
                results=results,
                final_answer=final_answer,
            )
            if verification["passed"]:
                self._log(f"\n最终回答: {final_answer}")
                self._save_to_history(query, final_answer)
                return final_answer

            if attempt >= total_attempts:
                self._log("\n⚠️  已达到最大重规划次数，返回最后一次候选回答。")
                self._save_to_history(query, final_answer)
                return final_answer

            working_query = self._reflect_and_replan(
                original_query=query,
                working_query=working_query,
                plan=plan,
                results=results,
                final_answer=final_answer,
                verification=verification,
            )

        self._save_to_history(query, last_answer)
        return last_answer

    def _plan(self, query: str) -> list[dict[str, str]]:
        """生成线性执行计划。"""
        planner_prompt = self._build_planner_prompt()
        response = self.llm.chat_simple(prompt=query, system=planner_prompt)
        self._log("\n🤔 Planner 输出:")
        self._log(self._indent(response))
        return self._parse_plan(response)

    def _parse_plan(self, text: str) -> list[dict[str, str]]:
        """从 Planner 输出中提取步骤列表。"""
        json_block = self._extract_json_block(text)
        if not json_block:
            return []

        try:
            payload = json.loads(json_block)
        except json.JSONDecodeError:
            return []

        raw_steps: Any
        if isinstance(payload, dict):
            raw_steps = payload.get("steps", [])
        elif isinstance(payload, list):
            raw_steps = payload
        else:
            return []

        if not isinstance(raw_steps, list):
            return []

        steps: list[dict[str, str]] = []
        for index, item in enumerate(raw_steps, 1):
            title = f"步骤 {index}"
            task = ""

            if isinstance(item, str):
                task = item.strip()
            elif isinstance(item, dict):
                title = str(item.get("title") or item.get("name") or title).strip()
                task = str(
                    item.get("task")
                    or item.get("instruction")
                    or item.get("description")
                    or ""
                ).strip()

            if not task:
                continue

            steps.append({"title": title, "task": task})
            if len(steps) >= self.max_plan_steps:
                break

        return steps

    def _execute_plan_steps(
        self,
        working_query: str,
        plan: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """顺序执行所有计划步骤。"""
        results: list[dict[str, str]] = []
        for index, step in enumerate(plan, 1):
            self._log(f"\n--- 执行步骤 {index}/{len(plan)}: {step['title']} ---")
            step_query = self._build_step_query(working_query, plan, results, index, step)
            step_result = self._run_sub_agent(step_query, self.executor_mode)
            results.append(
                {
                    "title": step["title"],
                    "task": step["task"],
                    "result": step_result,
                }
            )
            self._log(f"  Step Result: {step_result}")
        return results

    def _build_step_query(
        self,
        query: str,
        plan: list[dict[str, str]],
        results: list[dict[str, str]],
        step_index: int,
        step: dict[str, str],
    ) -> str:
        """构建单个执行步骤的输入。"""
        plan_text = self._format_plan(plan)
        completed_text = self._format_results(results, empty_text="(暂无已完成步骤)")

        return (
            "你正在执行一个 Plan-and-Execute 任务。\n\n"
            f"当前工作任务:\n{query}\n\n"
            f"总体计划:\n{plan_text}\n\n"
            f"当前步骤: 第 {step_index} 步 / {len(plan)}\n"
            f"步骤标题: {step['title']}\n"
            f"当前子任务: {step['task']}\n\n"
            f"已完成步骤结果:\n{completed_text}\n\n"
            "请只完成当前步骤，不要提前总结整个任务。如果需要，可调用工具。"
        )

    def _run_sub_agent(self, query: str, mode: str) -> str:
        """使用指定模式执行单个子任务。"""
        sub_agent = ReActAgent(
            llm=self.llm,
            tool_registry=self.tools,
            mode=mode,
            max_steps=self.max_steps,
            verbose=False,
            system_prompt=self._build_executor_system_prompt(mode),
            enable_verifier=False,
        )
        return sub_agent.run(query)

    def _build_planner_prompt(self) -> str:
        """构建 Planner 的系统提示词。"""
        tools_description = self.tools.get_tools_description() or "(当前没有可用工具)"
        prompt = SYSTEM_PROMPT_PLAN_AND_EXECUTE_PLANNER.format(
            max_plan_steps=self.max_plan_steps,
            tools_description=tools_description,
        )
        if self.system_prompt:
            prompt += f"\n\n补充上下文：\n{self.system_prompt}"
        return prompt

    def _build_executor_system_prompt(self, mode: str) -> str:
        """构建执行阶段的系统提示词。"""
        execution_guidance = (
            "你当前处于 Plan-and-Execute 的执行阶段。"
            "请专注完成当前子任务，必要时使用工具，但不要抢先总结整个原始任务。"
        )

        if mode == "function_calling":
            base_prompt = self.system_prompt or SYSTEM_PROMPT_FUNCTION_CALLING
            return f"{base_prompt}\n\n{execution_guidance}"

        base_prompt = self.system_prompt or "你是一个智能助手，可以使用工具完成当前子任务。"
        return f"{base_prompt}\n\n{execution_guidance}"

    def _summarize_plan_execution(
        self,
        original_query: str,
        working_query: str,
        plan: list[dict[str, str]],
        results: list[dict[str, str]],
    ) -> str:
        """汇总所有步骤结果，生成最终回答。"""
        summary_prompt = SYSTEM_PROMPT_PLAN_AND_EXECUTE_SUMMARIZER
        if self.system_prompt:
            summary_prompt += f"\n\n补充上下文：\n{self.system_prompt}"

        summary_input = (
            f"原始任务:\n{original_query}\n\n"
            f"当前工作任务:\n{working_query}\n\n"
            f"执行计划:\n{self._format_plan(plan)}\n\n"
            f"步骤结果:\n{self._format_results(results)}"
        )
        summary = self.llm.chat_simple(prompt=summary_input, system=summary_prompt).strip()
        return summary or (results[-1]["result"] if results else DEFAULT_FALLBACK_MESSAGE)

    def _verify_plan_execution(
        self,
        *,
        original_query: str,
        working_query: str,
        plan: list[dict[str, str]],
        results: list[dict[str, str]],
        final_answer: str,
    ) -> dict[str, Any]:
        """对候选回答做最终验收。"""
        if not self.enable_verifier:
            return {
                "passed": True,
                "reason": "Verifier 已关闭。",
                "missing": [],
                "suggested_fix": "",
                "parser_failed": False,
            }

        verifier_prompt = SYSTEM_PROMPT_PLAN_AND_EXECUTE_VERIFIER
        if self.system_prompt:
            verifier_prompt += f"\n\n补充上下文：\n{self.system_prompt}"

        verifier_input = (
            f"原始任务:\n{original_query}\n\n"
            f"当前工作任务:\n{working_query}\n\n"
            f"执行计划:\n{self._format_plan(plan, empty_text='(未生成有效计划)')}\n\n"
            f"步骤结果:\n{self._format_results(results, empty_text='(没有步骤结果，可能直接执行了任务)')}\n\n"
            f"候选回答:\n{final_answer}"
        )
        response = self.llm.chat_simple(prompt=verifier_input, system=verifier_prompt)
        self._log("\n🔍 Verifier 输出:")
        self._log(self._indent(response))

        verification = self._parse_verification_result(response)
        if verification["passed"]:
            self._log(f"✅ 验证通过: {verification['reason']}")
        else:
            self._log(f"❌ 验证未通过: {verification['reason']}")
            if verification["missing"]:
                self._log(f"  缺失点: {', '.join(verification['missing'])}")
            if verification["suggested_fix"]:
                self._log(f"  修正方向: {verification['suggested_fix']}")
        return verification

    def _parse_verification_result(self, text: str) -> dict[str, Any]:
        """解析 Verifier 的 JSON 输出。"""
        json_block = self._extract_json_block(text)
        if not json_block:
            return {
                "passed": True,
                "reason": "Verifier 输出不可解析，默认接受当前结果。",
                "missing": [],
                "suggested_fix": "",
                "parser_failed": True,
            }

        try:
            payload = json.loads(json_block)
        except json.JSONDecodeError:
            return {
                "passed": True,
                "reason": "Verifier JSON 非法，默认接受当前结果。",
                "missing": [],
                "suggested_fix": "",
                "parser_failed": True,
            }

        if not isinstance(payload, dict):
            return {
                "passed": True,
                "reason": "Verifier 输出格式不正确，默认接受当前结果。",
                "missing": [],
                "suggested_fix": "",
                "parser_failed": True,
            }

        raw_passed = payload.get("passed")
        if isinstance(raw_passed, bool):
            passed = raw_passed
        elif isinstance(raw_passed, str):
            passed = raw_passed.strip().lower() in {"true", "pass", "passed", "yes", "ok"}
        else:
            passed = True

        reason = str(payload.get("reason") or "未提供原因。")
        missing = self._normalize_text_list(payload.get("missing"))
        suggested_fix = str(
            payload.get("suggested_fix")
            or payload.get("fix")
            or payload.get("action")
            or ""
        ).strip()

        return {
            "passed": passed,
            "reason": reason,
            "missing": missing,
            "suggested_fix": suggested_fix,
            "parser_failed": False,
        }

    def _reflect_and_replan(
        self,
        *,
        original_query: str,
        working_query: str,
        plan: list[dict[str, str]],
        results: list[dict[str, str]],
        final_answer: str,
        verification: dict[str, Any],
    ) -> str:
        """根据验收失败结果生成下一轮工作任务。"""
        reflection_prompt = SYSTEM_PROMPT_PLAN_AND_EXECUTE_REFLECTOR
        if self.system_prompt:
            reflection_prompt += f"\n\n补充上下文：\n{self.system_prompt}"

        reflection_input = (
            f"原始任务:\n{original_query}\n\n"
            f"当前工作任务:\n{working_query}\n\n"
            f"上一轮执行计划:\n{self._format_plan(plan, empty_text='(未生成有效计划)')}\n\n"
            f"上一轮步骤结果:\n{self._format_results(results, empty_text='(没有步骤结果，可能直接执行了任务)')}\n\n"
            f"上一轮候选回答:\n{final_answer}\n\n"
            "Verifier 结论:\n"
            f"- 是否通过: {verification['passed']}\n"
            f"- 原因: {verification['reason']}\n"
            f"- 缺失点: {json.dumps(verification['missing'], ensure_ascii=False)}\n"
            f"- 修正方向: {verification['suggested_fix']}"
        )
        response = self.llm.chat_simple(prompt=reflection_input, system=reflection_prompt)
        self._log("\n🪞 Reflection 输出:")
        self._log(self._indent(response))

        reflection = self._parse_reflection_result(response)
        revised_task = reflection.get("revised_task", "").strip()
        if not revised_task or revised_task == working_query:
            revised_task = self._build_fallback_replan_query(
                original_query=original_query,
                working_query=working_query,
                verification=verification,
            )

        issues = reflection.get("issues", [])
        if issues:
            self._log(f"📝 复盘问题: {', '.join(issues)}")
        self._log("🧭 下一轮工作任务:")
        self._log(self._indent(revised_task))
        return revised_task

    def _parse_reflection_result(self, text: str) -> dict[str, Any]:
        """解析 Reflection 的 JSON 输出。"""
        json_block = self._extract_json_block(text)
        if not json_block:
            return {}

        try:
            payload = json.loads(json_block)
        except json.JSONDecodeError:
            return {}

        if not isinstance(payload, dict):
            return {}

        return {
            "issues": self._normalize_text_list(payload.get("issues")),
            "revised_task": str(payload.get("revised_task") or "").strip(),
            "notes": str(payload.get("notes") or "").strip(),
        }

    def _build_fallback_replan_query(
        self,
        *,
        original_query: str,
        working_query: str,
        verification: dict[str, Any],
    ) -> str:
        """当 Reflection 输出无效时，构造兜底的重规划任务。"""
        missing_text = "；".join(verification["missing"]) or "请补齐缺失信息并确保最终结论完整。"
        suggested_fix = verification["suggested_fix"] or "重新规划并补救缺失点。"
        return (
            f"{original_query}\n\n"
            "[上轮执行复盘]\n"
            f"上轮工作任务：{working_query}\n"
            f"验收未通过原因：{verification['reason']}\n"
            f"缺失点：{missing_text}\n"
            f"修正建议：{suggested_fix}\n\n"
            "请基于以上复盘重新规划任务，并优先补齐缺失部分。"
        )

    # ================================================================
    # 辅助方法
    # ================================================================

    def reset(self) -> None:
        """重置对话历史。"""
        self.history.clear()

    def _validate_mode_config(self) -> None:
        """校验模式配置是否合法。"""
        if self.mode not in SUPPORTED_MODES:
            raise ValueError(
                f"不支持的模式: {self.mode}，请使用 {', '.join(repr(mode) for mode in SUPPORTED_MODES)}"
            )

        if self.executor_mode not in SUPPORTED_EXECUTOR_MODES:
            raise ValueError(
                "不支持的 executor_mode: "
                f"{self.executor_mode}，请使用 {', '.join(repr(mode) for mode in SUPPORTED_EXECUTOR_MODES)}"
            )

        if self.max_plan_steps <= 0:
            raise ValueError("max_plan_steps 必须大于 0")

        if self.max_replans < 0:
            raise ValueError("max_replans 不能小于 0")

    def _build_text_parsing_system_prompt(self) -> str:
        """构建 text_parsing 模式使用的系统提示词。"""
        if self.system_prompt:
            return (
                f"{self.system_prompt}\n\n"
                f"可用工具：\n{self.tools.get_tools_description()}\n\n"
                "你必须严格按照以下格式回答：\n"
                "Thought: <你的思考过程>\n"
                "Action: <要调用的工具名称>\n"
                "Action Input: <工具的参数，JSON 格式>\n\n"
                "当你准备好给出最终答案时：\n"
                "Thought: <最终思考>\n"
                "Final Answer: <你的最终回答>"
            )

        return SYSTEM_PROMPT_TEXT_PARSING.format(
            tools_description=self.tools.get_tools_description()
        )

    def _save_to_history(self, query: str, answer: str) -> None:
        """保存一轮完整对话到历史。"""
        self.history.add_user_message(query)
        self.history.add_assistant_message(answer)

    def _extract_json_block(self, text: str) -> str | None:
        """从任意文本中提取 JSON 对象或数组。"""
        fenced_match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
        if fenced_match:
            return fenced_match.group(1)

        object_start = text.find("{")
        object_end = text.rfind("}")
        if object_start != -1 and object_end != -1 and object_end > object_start:
            return text[object_start:object_end + 1]

        array_start = text.find("[")
        array_end = text.rfind("]")
        if array_start != -1 and array_end != -1 and array_end > array_start:
            return text[array_start:array_end + 1]

        return None

    @staticmethod
    def _normalize_text_list(value: Any) -> list[str]:
        """将任意值归一化为字符串列表。"""
        if value is None:
            return []
        if isinstance(value, str):
            item = value.strip()
            return [item] if item else []
        if isinstance(value, list):
            items: list[str] = []
            for item in value:
                text = str(item).strip()
                if text:
                    items.append(text)
            return items
        text = str(value).strip()
        return [text] if text else []

    def _format_plan(self, plan: list[dict[str, str]], empty_text: str = "(暂无计划)") -> str:
        """格式化计划文本。"""
        if not plan:
            return empty_text
        return "\n".join(
            f"{index}. {item['title']}: {item['task']}"
            for index, item in enumerate(plan, 1)
        )

    def _format_results(self, results: list[dict[str, str]], empty_text: str = "(暂无步骤结果)") -> str:
        """格式化步骤结果文本。"""
        if not results:
            return empty_text
        return "\n\n".join(
            (
                f"步骤 {index}: {item['title']}\n"
                f"子任务: {item['task']}\n"
                f"执行结果: {item['result']}"
            )
            for index, item in enumerate(results, 1)
        )

    def _log_plan(self, plan: list[dict[str, str]]) -> None:
        """打印计划摘要。"""
        self._log(f"\n📋 执行计划 ({len(plan)} 步):")
        for index, step in enumerate(plan, 1):
            self._log(f"  {index}. {step['title']} -> {step['task']}")

    def _log(self, message: str) -> None:
        """打印调试信息。"""
        if self.verbose:
            print(message)

    @staticmethod
    def _indent(text: str, prefix: str = "    ") -> str:
        """缩进文本。"""
        return "\n".join(f"{prefix}{line}" for line in text.split("\n"))
