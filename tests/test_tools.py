"""工具模块单元测试"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
import unittest
from pathlib import Path

from src.tools.base import Tool, ToolRegistry
from src.tools.calculator import CalculatorTool
from src.tools.read_local_file import ReadLocalFileTool
from src.tools.weather import WeatherTool
from src.tools.search import SearchTool


class FakeMCPClient:
    def __init__(self, response: str = "1. [示例结果](https://example.com)\n   示例摘要"):
        self.response = response
        self.calls: list[tuple[str | None, dict[str, str]]] = []

    def call_tool(self, *, arguments: dict[str, str], tool_name: str | None = None) -> str:
        self.calls.append((tool_name, arguments))
        return self.response

    def describe(self) -> str:
        return "mcp(uvx duckduckgo-mcp-server)"


class TestCalculatorTool(unittest.TestCase):
    def setUp(self):
        self.calc = CalculatorTool()

    def test_basic_arithmetic(self):
        self.assertEqual(self.calc.run(expression="2 + 3"), "5")
        self.assertEqual(self.calc.run(expression="10 - 4"), "6")
        self.assertEqual(self.calc.run(expression="6 * 7"), "42")
        self.assertEqual(self.calc.run(expression="15 / 3"), "5")

    def test_complex_expression(self):
        self.assertEqual(self.calc.run(expression="(2 + 3) * 4"), "20")
        self.assertEqual(self.calc.run(expression="2 ** 10"), "1024")

    def test_math_functions(self):
        result = self.calc.run(expression="sqrt(16)")
        self.assertEqual(result, "4")

    def test_division_by_zero(self):
        result = self.calc.run(expression="1 / 0")
        self.assertIn("错误", result)

    def test_invalid_expression(self):
        result = self.calc.run(expression="import os")
        self.assertIn("错误", result)

    def test_openai_tool_format(self):
        tool_def = self.calc.to_openai_tool()
        self.assertEqual(tool_def["type"], "function")
        self.assertEqual(tool_def["function"]["name"], "calculator")
        self.assertIn("parameters", tool_def["function"])


class TestWeatherTool(unittest.TestCase):
    def setUp(self):
        self.weather = WeatherTool()

    def test_known_city(self):
        result = self.weather.run(city="北京")
        self.assertIn("北京", result)
        self.assertIn("°C", result)

    def test_unknown_city(self):
        result = self.weather.run(city="纽约")
        self.assertIn("纽约", result)
        self.assertIn("°C", result)


class TestSearchTool(unittest.TestCase):
    def setUp(self):
        self.original_env = os.environ.copy()
        self.search = SearchTool()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_known_query(self):
        result = self.search.run(query="python 编程")
        self.assertIn("Python", result)

    def test_unknown_query(self):
        result = self.search.run(query="随机搜索词")
        self.assertIn("搜索", result)

    def test_from_env_uses_mcp_backend_when_configured(self):
        os.environ["SEARCH_PROVIDER"] = "mcp"
        os.environ["MCP_SEARCH_SERVER_COMMAND"] = "uvx"
        os.environ["MCP_SEARCH_SERVER_ARGS"] = "duckduckgo-mcp-server"
        fake_client = FakeMCPClient()

        tool = SearchTool.from_env(mcp_client=fake_client)

        self.assertEqual(tool.backend, "mcp")
        self.assertEqual(tool.backend_label(), "mcp(uvx duckduckgo-mcp-server)")

    def test_mcp_backend_calls_client(self):
        fake_client = FakeMCPClient(response="1. [Python 官方文档](https://docs.python.org)\n   Python 是一种解释型语言。")
        tool = SearchTool(
            backend="mcp",
            mcp_client=fake_client,
            mcp_tool_name="duckduckgo_search",
        )

        result = tool.run(query="python")

        self.assertIn("搜索 'python' 的结果", result)
        self.assertIn("Python 官方文档", result)
        self.assertEqual(
            fake_client.calls,
            [("duckduckgo_search", {"query": "python"})],
        )

    def test_mcp_backend_without_client_returns_helpful_error(self):
        tool = SearchTool(backend="mcp")

        result = tool.run(query="python")

        self.assertIn("MCP_SEARCH_SERVER_COMMAND", result)


class TestReadLocalFileTool(unittest.TestCase):
    def test_reads_requested_line_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            target = project_root / "notes.txt"
            target.write_text("第一行\n第二行\n第三行\n第四行\n", encoding="utf-8")
            tool = ReadLocalFileTool(project_root=project_root)

            result = tool.run(path="notes.txt", start_line=2, max_lines=2)

        self.assertIn("行范围：2-3 / 共 4 行", result)
        self.assertIn("   2: 第二行", result)
        self.assertIn("   3: 第三行", result)
        self.assertNotIn("第一行", result)

    def test_rejects_paths_outside_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp) / "workspace"
            project_root.mkdir()
            outside_file = Path(tmp) / "secret.txt"
            outside_file.write_text("top secret", encoding="utf-8")
            tool = ReadLocalFileTool(project_root=project_root)

            result = tool.run(path=str(outside_file))

        self.assertIn("错误：不允许访问工作区外的路径", result)


class TestToolRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        self.registry.register(CalculatorTool())
        self.registry.register(WeatherTool())

    def test_register_and_get(self):
        tool = self.registry.get("calculator")
        self.assertIsNotNone(tool)
        self.assertEqual(tool.name, "calculator")

    def test_get_nonexistent(self):
        self.assertIsNone(self.registry.get("nonexistent"))

    def test_duplicate_register(self):
        with self.assertRaises(ValueError):
            self.registry.register(CalculatorTool())

    def test_execute(self):
        result = self.registry.execute("calculator", '{"expression": "1 + 1"}')
        self.assertEqual(result, "2")

    def test_execute_nonexistent(self):
        result = self.registry.execute("nonexistent", "{}")
        self.assertIn("错误", result)

    def test_to_openai_tools(self):
        tools = self.registry.to_openai_tools()
        self.assertEqual(len(tools), 2)

    def test_contains(self):
        self.assertIn("calculator", self.registry)
        self.assertNotIn("nonexistent", self.registry)

    def test_len(self):
        self.assertEqual(len(self.registry), 2)


if __name__ == "__main__":
    unittest.main()
