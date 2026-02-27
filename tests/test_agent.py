"""Agent 模块单元测试

注意：这些测试不依赖真实 LLM API，仅测试辅助组件。
集成测试需要配置 API Key 后运行 examples/ 中的示例。
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from src.memory.history import ConversationHistory
from src.tools.base import ToolRegistry
from src.tools.calculator import CalculatorTool
from src.tools.weather import WeatherTool
from src.tools.search import SearchTool


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
        # max_turns=3，超过后应丢弃最早的对话
        for i in range(5):
            self.history.add_user_message(f"问题{i}")
            self.history.add_assistant_message(f"回答{i}")

        self.assertEqual(self.history.turn_count, 3)
        messages = self.history.get_messages()
        # 应该保留最后 3 轮
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
        # 原始历史不应被修改
        self.assertEqual(len(self.history), 1)


class TestToolRegistryIntegration(unittest.TestCase):
    """测试工具注册中心的集成功能。"""

    def test_full_workflow(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        registry.register(WeatherTool())
        registry.register(SearchTool())

        # 验证所有工具注册
        self.assertEqual(len(registry), 3)
        self.assertEqual(
            set(registry.tool_names),
            {"calculator", "weather", "search"},
        )

        # 验证 OpenAI 格式输出
        tools = registry.to_openai_tools()
        self.assertEqual(len(tools), 3)
        for tool in tools:
            self.assertEqual(tool["type"], "function")
            self.assertIn("name", tool["function"])
            self.assertIn("description", tool["function"])
            self.assertIn("parameters", tool["function"])

        # 验证工具描述生成
        desc = registry.get_tools_description()
        self.assertIn("calculator", desc)
        self.assertIn("weather", desc)
        self.assertIn("search", desc)


if __name__ == "__main__":
    unittest.main()
