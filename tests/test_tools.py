"""工具模块单元测试"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from src.tools.base import Tool, ToolRegistry
from src.tools.calculator import CalculatorTool
from src.tools.weather import WeatherTool
from src.tools.search import SearchTool


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
        self.search = SearchTool()

    def test_known_query(self):
        result = self.search.run(query="python 编程")
        self.assertIn("Python", result)

    def test_unknown_query(self):
        result = self.search.run(query="随机搜索词")
        self.assertIn("搜索", result)


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
