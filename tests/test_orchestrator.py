"""Orchestrator 模块单元测试。

验证失败后的重规划行为，不依赖真实 LLM API。
"""

import os
import sys
import types
import unittest

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

from src.multi.orchestrator import OrchestratorMultiAgent
from src.tools.base import ToolRegistry


class FakeLLM:
    def __init__(self, *, simple_responses=None):
        self.simple_responses = list(simple_responses or [])
        self.simple_calls: list[tuple[str, str]] = []

    def chat_simple(self, prompt: str, system: str = "") -> str:
        self.simple_calls.append((prompt, system))
        return self.simple_responses.pop(0) if self.simple_responses else ""


class StubOrchestrator(OrchestratorMultiAgent):
    def __init__(self, *, llm, dispatch_responses, max_replan=1):
        super().__init__(llm=llm, tool_registry=ToolRegistry(), max_replan=max_replan, verbose=False)
        self._agents = {"researcher": object()}
        self.dispatch_responses = list(dispatch_responses)

    def _dispatch(self, agent_name: str, task: str) -> str:
        result = self.dispatch_responses.pop(0)
        self.state.add_result(agent_name, result)
        return result


class TestOrchestratorReplan(unittest.TestCase):
    def test_replans_when_first_plan_is_invalid(self):
        fake_llm = FakeLLM(
            simple_responses=[
                "这轮没有返回 JSON 计划。",
                '[{"agent": "researcher", "task": "收集有效信息"}]',
                "最终总结：任务已经完成。",
            ]
        )
        orchestrator = StubOrchestrator(
            llm=fake_llm,
            dispatch_responses=["已完成信息收集。"],
            max_replan=1,
        )

        result = orchestrator.run("请先研究资料，再输出结果。")

        self.assertEqual(result, "最终总结：任务已经完成。")
        self.assertEqual(len(fake_llm.simple_calls), 3)
        self.assertIn("上轮失败信息", fake_llm.simple_calls[1][0])
        self.assertEqual(orchestrator.state.status, "done")

    def test_replans_when_execution_step_fails(self):
        fake_llm = FakeLLM(
            simple_responses=[
                '[{"agent": "researcher", "task": "第一次尝试收集信息"}]',
                '[{"agent": "researcher", "task": "重新收集并补齐信息"}]',
                "最终总结：第二轮执行成功。",
            ]
        )
        orchestrator = StubOrchestrator(
            llm=fake_llm,
            dispatch_responses=["错误：搜索失败。", "已重新完成信息收集。"],
            max_replan=1,
        )

        result = orchestrator.run("完成一次研究任务。")

        self.assertEqual(result, "最终总结：第二轮执行成功。")
        self.assertEqual(len(fake_llm.simple_calls), 3)
        self.assertIn("执行失败", fake_llm.simple_calls[1][0])
        self.assertEqual(orchestrator.state.status, "done")


if __name__ == "__main__":
    unittest.main()
