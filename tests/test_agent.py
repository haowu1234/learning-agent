"""Agent 模块单元测试。

注意：这些测试不依赖真实 LLM API，仅测试辅助组件与 Agent 编排逻辑。
集成测试需要配置 API Key 后运行 examples/ 中的示例。
"""

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if "dotenv" not in sys.modules:
    dotenv_module = types.ModuleType("dotenv")
    dotenv_module.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv_module

if "openai" not in sys.modules:
    openai_module = types.ModuleType("openai")

    class _DummyOpenAI:
        def __init__(self, *args, **kwargs):
            pass

    openai_module.OpenAI = _DummyOpenAI
    sys.modules["openai"] = openai_module
    sys.modules["openai.types"] = types.ModuleType("openai.types")

    chat_module = types.ModuleType("openai.types.chat")

    class _DummyChatCompletion:
        pass

    chat_module.ChatCompletion = _DummyChatCompletion
    sys.modules["openai.types.chat"] = chat_module

from src.agent import ReActAgent, SUPPORTED_MODES
from src.memory.history import ConversationHistory
from src.tools.base import ToolRegistry
from src.tools.calculator import CalculatorTool
from src.tools.read_local_file import ReadLocalFileTool
from src.tools.search import SearchTool
from src.tools.weather import WeatherTool


class _FakeMessage:
    def __init__(self, content: str = "", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class _FakeChoice:
    def __init__(self, message: _FakeMessage):
        self.message = message


class _FakeResponse:
    def __init__(self, content: str = "", tool_calls=None):
        self.choices = [_FakeChoice(_FakeMessage(content, tool_calls))]


class _FakeToolFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, tool_id: str, name: str, arguments: str):
        self.id = tool_id
        self.function = _FakeToolFunction(name, arguments)


class FakeLLM:
    """用于测试 Agent 的轻量级 LLM 替身。"""

    def __init__(self, *, simple_responses=None, chat_responses=None):
        self.simple_responses = list(simple_responses or [])
        self.chat_responses = list(chat_responses or [])
        self.simple_calls: list[tuple[str, str]] = []
        self.chat_calls: list[dict] = []

    def chat(self, messages, tools=None, temperature=0.7, max_tokens=2048):
        self.chat_calls.append(
            {
                "messages": messages,
                "tools": tools,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        response = self.chat_responses.pop(0) if self.chat_responses else ""
        if isinstance(response, _FakeResponse):
            return response
        return _FakeResponse(content=response)

    def chat_simple(self, prompt: str, system: str = "") -> str:
        self.simple_calls.append((prompt, system))
        return self.simple_responses.pop(0) if self.simple_responses else ""


class TestConversationHistory(unittest.TestCase):
    def setUp(self):
        self.history = ConversationHistory(max_turns=3)

    def test_add_messages(self):
        self.history.add_user_message("你好")
        self.history.add_assistant_message("你好！有什么可以帮你的？")
        self.assertEqual(len(self.history), 2)
        self.assertEqual(self.history.turn_count, 1)

    def test_get_messages(self):
        self.history.add_user_message("问题1")
        self.history.add_assistant_message("回答1")
        messages = self.history.get_messages()
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[1]["role"], "assistant")

    def test_truncation(self):
        for i in range(5):
            self.history.add_user_message(f"问题{i}")
            self.history.add_assistant_message(f"回答{i}")

        self.assertEqual(self.history.turn_count, 3)
        messages = self.history.get_messages()
        self.assertEqual(messages[0]["content"], "问题2")

    def test_clear(self):
        self.history.add_user_message("你好")
        self.history.add_assistant_message("你好！")
        self.history.clear()
        self.assertEqual(len(self.history), 0)

    def test_immutable_get(self):
        self.history.add_user_message("测试")
        messages = self.history.get_messages()
        messages.append({"role": "user", "content": "篡改"})
        self.assertEqual(len(self.history), 1)


class TestToolRegistryIntegration(unittest.TestCase):
    """测试工具注册中心的集成功能。"""

    def test_full_workflow(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        registry.register(WeatherTool())
        registry.register(SearchTool())

        self.assertEqual(len(registry), 3)
        self.assertEqual(
            set(registry.tool_names),
            {"calculator", "weather", "search"},
        )

        tools = registry.to_openai_tools()
        self.assertEqual(len(tools), 3)
        for tool in tools:
            self.assertEqual(tool["type"], "function")
            self.assertIn("name", tool["function"])
            self.assertIn("description", tool["function"])
            self.assertIn("parameters", tool["function"])

        desc = registry.get_tools_description()
        self.assertIn("calculator", desc)
        self.assertIn("weather", desc)
        self.assertIn("search", desc)


class TestReActAgentModes(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        self.registry.register(CalculatorTool())

    def test_available_modes_contains_plan_and_execute(self):
        self.assertEqual(ReActAgent.available_modes(), SUPPORTED_MODES)
        self.assertIn("plan_and_execute", SUPPORTED_MODES)

    def test_invalid_mode_raises_error(self):
        agent = ReActAgent(
            llm=FakeLLM(),
            tool_registry=self.registry,
            mode="unsupported_mode",
            verbose=False,
        )

        with self.assertRaises(ValueError) as context:
            agent.run("测试")

        self.assertIn("plan_and_execute", str(context.exception))

    def test_run_with_trace_collects_tool_calls_in_function_calling(self):
        fake_llm = FakeLLM(
            chat_responses=[
                _FakeResponse(tool_calls=[
                    _FakeToolCall("call_1", "calculator", '{"expression": "2+3"}')
                ]),
                _FakeResponse(content="最终答案是 5。"),
            ]
        )
        agent = ReActAgent(
            llm=fake_llm,
            tool_registry=self.registry,
            mode="function_calling",
            verbose=False,
        )

        result = agent.run_with_trace("请计算 2+3")

        self.assertEqual(result.final_answer, "最终答案是 5。")
        self.assertEqual(len(result.tool_traces), 1)
        self.assertEqual(result.tool_traces[0].tool_name, "calculator")
        self.assertIn('"expression": "2+3"', result.tool_traces[0].tool_input)
        self.assertIn("5", result.tool_traces[0].observation)

    def test_plan_and_execute_runs_planner_executor_summarizer_and_verifier(self):
        fake_llm = FakeLLM(
            simple_responses=[
                '{"steps": [{"title": "收集信息", "task": "先确认已知条件"}, {"title": "完成计算", "task": "基于上一步结果给出计算结论"}]}',
                "综合结论：先确认了条件，再完成了计算。",
                '{"passed": true, "reason": "回答已经覆盖任务要求", "missing": [], "suggested_fix": ""}',
            ],
            chat_responses=[
                "Thought: 先整理信息\nFinal Answer: 已确认已知条件。",
                "Thought: 继续执行\nFinal Answer: 计算结果是 142。",
            ],
        )
        agent = ReActAgent(
            llm=fake_llm,
            tool_registry=self.registry,
            mode="plan_and_execute",
            executor_mode="text_parsing",
            max_plan_steps=4,
            verbose=False,
        )

        result = agent.run("如果温度乘以 3 再加上 100，结果是多少？")

        self.assertEqual(result, "综合结论：先确认了条件，再完成了计算。")
        self.assertEqual(len(fake_llm.simple_calls), 3)
        self.assertEqual(len(fake_llm.chat_calls), 2)
        self.assertIn("任务规划器", fake_llm.simple_calls[0][1])
        self.assertIn("结果汇总器", fake_llm.simple_calls[1][1])
        self.assertIn("任务验收器", fake_llm.simple_calls[2][1])
        self.assertIn("当前步骤: 第 1 步 / 2", fake_llm.chat_calls[0]["messages"][-1]["content"])
        self.assertEqual(agent.history.get_messages()[-1]["content"], result)

    def test_plan_and_execute_reflects_and_replans_after_failed_verification(self):
        fake_llm = FakeLLM(
            simple_responses=[
                '{"steps": [{"title": "先给结论", "task": "先尝试给出一个简短结论"}]}',
                "初版结论：目前信息不足。",
                '{"passed": false, "reason": "缺少最终计算结果", "missing": ["最终数值"], "suggested_fix": "重新执行并补充最终计算"}',
                '{"issues": ["没有产出最终数值"], "revised_task": "请重新完成原始任务，重点补充最终计算结果，并确保给出明确数值结论。", "notes": "不要只给模糊结论"}',
                '{"steps": [{"title": "完成计算", "task": "根据已知条件给出最终数值结果"}]}',
                "修正后结论：最终结果是 142。",
                '{"passed": true, "reason": "已经补齐数值并完成任务", "missing": [], "suggested_fix": ""}',
            ],
            chat_responses=[
                "Thought: 先试着总结\nFinal Answer: 目前信息不足。",
                "Thought: 重新计算\nFinal Answer: 最终数值结果是 142。",
            ],
        )
        agent = ReActAgent(
            llm=fake_llm,
            tool_registry=self.registry,
            mode="plan_and_execute",
            executor_mode="text_parsing",
            max_plan_steps=3,
            max_replans=1,
            verbose=False,
        )

        result = agent.run("请给出最终计算结果。")

        self.assertEqual(result, "修正后结论：最终结果是 142。")
        self.assertEqual(len(fake_llm.simple_calls), 7)
        self.assertEqual(len(fake_llm.chat_calls), 2)
        self.assertIn("任务验收器", fake_llm.simple_calls[2][1])
        self.assertIn("任务复盘与重规划助手", fake_llm.simple_calls[3][1])
        self.assertIn("重点补充最终计算结果", fake_llm.simple_calls[4][0])
        self.assertEqual(agent.history.get_messages()[-1]["content"], result)

    def test_plan_and_execute_falls_back_to_direct_execution_when_plan_invalid(self):
        fake_llm = FakeLLM(
            simple_responses=[
                "我觉得先想一想，但这里没有给出 JSON 计划。",
                '{"passed": true, "reason": "直接执行结果已经满足任务", "missing": [], "suggested_fix": ""}',
            ],
            chat_responses=["Thought: 直接执行\nFinal Answer: 直接执行结果。"],
        )
        agent = ReActAgent(
            llm=fake_llm,
            tool_registry=self.registry,
            mode="plan_and_execute",
            executor_mode="text_parsing",
            verbose=False,
        )

        result = agent.run("直接帮我完成这个简单问题")

        self.assertEqual(result, "直接执行结果。")
        self.assertEqual(len(fake_llm.simple_calls), 2)
        self.assertEqual(len(fake_llm.chat_calls), 1)

    def test_function_calling_injects_skills_prompt_when_read_tool_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            skills_root = project_root / "skills"
            skill_dir = skills_root / "public" / "demo-skill"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\n"
                "name: demo-skill\n"
                "description: 演示技能\n"
                "---\n\n"
                "# Demo\n",
                encoding="utf-8",
            )

            registry = ToolRegistry()
            registry.register(ReadLocalFileTool(project_root=project_root))
            fake_llm = FakeLLM(chat_responses=["最终回答"])
            agent = ReActAgent(
                llm=fake_llm,
                tool_registry=registry,
                mode="function_calling",
                verbose=False,
                skills_path=skills_root,
            )

            agent.run("请处理这个技能相关任务")

        system_prompt = fake_llm.chat_calls[0]["messages"][0]["content"]
        self.assertIn("<skill_system>", system_prompt)
        self.assertIn("demo-skill", system_prompt)
        self.assertIn("read_local_file", system_prompt)

    def test_run_with_trace_can_execute_read_local_file_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            target = project_root / "materials" / "brief.txt"
            target.parent.mkdir(parents=True)
            target.write_text("第一行\n第二行\n第三行\n", encoding="utf-8")

            registry = ToolRegistry()
            registry.register(ReadLocalFileTool(project_root=project_root))
            fake_llm = FakeLLM(
                chat_responses=[
                    _FakeResponse(tool_calls=[
                        _FakeToolCall(
                            "call_read_1",
                            "read_local_file",
                            '{"path": "materials/brief.txt", "start_line": 2, "max_lines": 1}',
                        )
                    ]),
                    _FakeResponse(content="已读取材料并完成总结。"),
                ]
            )
            agent = ReActAgent(
                llm=fake_llm,
                tool_registry=registry,
                mode="function_calling",
                verbose=False,
            )

            result = agent.run_with_trace("先读取本地材料，再总结")

        self.assertEqual(result.final_answer, "已读取材料并完成总结。")
        self.assertEqual(len(result.tool_traces), 1)
        self.assertEqual(result.tool_traces[0].tool_name, "read_local_file")
        self.assertIn("第二行", result.tool_traces[0].observation)


if __name__ == "__main__":
    unittest.main()
