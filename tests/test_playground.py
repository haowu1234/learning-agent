"""Playground 单元测试。"""

import importlib
import os
import sys
import types
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def _install_fake_dependencies() -> None:
    dotenv_module = types.ModuleType("dotenv")
    dotenv_module.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv_module

    openai_module = types.ModuleType("openai")

    class DummyOpenAI:
        def __init__(self, api_key, base_url):
            self.api_key = api_key
            self.base_url = base_url
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **kwargs: kwargs)
            )

    openai_module.OpenAI = DummyOpenAI
    sys.modules["openai"] = openai_module
    sys.modules["openai.types"] = types.ModuleType("openai.types")

    chat_module = types.ModuleType("openai.types.chat")

    class DummyChatCompletion:
        pass

    chat_module.ChatCompletion = DummyChatCompletion
    sys.modules["openai.types.chat"] = chat_module


class TestPlaygroundHelpers(unittest.TestCase):
    def setUp(self):
        _install_fake_dependencies()
        sys.modules.pop("src.llm", None)
        sys.modules.pop("playground", None)
        self.playground = importlib.import_module("playground")

    def tearDown(self):
        sys.modules.pop("playground", None)
        sys.modules.pop("src.llm", None)

    def test_parse_bool(self):
        self.assertTrue(self.playground.parse_bool("true"))
        self.assertTrue(self.playground.parse_bool("ON"))
        self.assertFalse(self.playground.parse_bool("false"))
        self.assertFalse(self.playground.parse_bool("0"))

    def test_parse_bool_invalid(self):
        with self.assertRaises(ValueError):
            self.playground.parse_bool("maybe")

    def test_apply_runtime_updates(self):
        config = self.playground.PlaygroundConfig()
        agent = self.playground.build_agent(config)

        new_agent, new_config, message = self.playground.apply_runtime_updates(
            agent,
            config,
            ":set mode=function_calling quiet=true max_steps=7",
        )

        self.assertEqual(new_config.mode, "function_calling")
        self.assertTrue(new_config.quiet)
        self.assertEqual(new_config.max_steps, 7)
        self.assertIn("mode=function_calling", message)
        self.assertEqual(new_agent.mode, "function_calling")
        self.assertFalse(new_agent.verbose)

    def test_apply_runtime_updates_preserves_history_when_enabled(self):
        config = self.playground.PlaygroundConfig(keep_history=True)
        agent = self.playground.build_agent(config)
        agent.history.add_user_message("hello")
        agent.history.add_assistant_message("world")

        new_agent, new_config, _ = self.playground.apply_runtime_updates(
            agent,
            config,
            ":set max_plan_steps=6",
        )

        self.assertTrue(new_config.keep_history)
        self.assertEqual(len(new_agent.history), 2)
        self.assertEqual(new_agent.max_plan_steps, 6)

    def test_args_to_config_maps_replan_flags(self):
        parser = self.playground.build_parser()
        args = parser.parse_args(["--max-replans", "3", "--disable-verifier"])

        config = self.playground.args_to_config(args)

        self.assertEqual(config.max_replans, 3)
        self.assertFalse(config.enable_verifier)

    def test_apply_runtime_updates_rejects_invalid_key(self):
        config = self.playground.PlaygroundConfig()
        agent = self.playground.build_agent(config)

        with self.assertRaises(ValueError):
            self.playground.apply_runtime_updates(agent, config, ":set unknown=1")

    def test_build_registry_includes_read_local_file_tool(self):
        registry = self.playground.build_registry()
        self.assertIn("read_local_file", registry.tool_names)


if __name__ == "__main__":
    unittest.main()
