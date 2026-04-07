"""LLMClient 单元测试。"""

import importlib
import os
import sys
import types
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


class DummyOpenAI:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=lambda **kwargs: kwargs)
        )


def _install_fake_dependencies() -> None:
    dotenv_module = types.ModuleType("dotenv")
    dotenv_module.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv_module

    openai_module = types.ModuleType("openai")
    openai_module.OpenAI = DummyOpenAI
    sys.modules["openai"] = openai_module
    sys.modules["openai.types"] = types.ModuleType("openai.types")

    chat_module = types.ModuleType("openai.types.chat")

    class DummyChatCompletion:
        pass

    chat_module.ChatCompletion = DummyChatCompletion
    sys.modules["openai.types.chat"] = chat_module


class TestLLMClientDefaults(unittest.TestCase):
    def setUp(self):
        self.original_env = os.environ.copy()
        _install_fake_dependencies()
        sys.modules.pop("src.llm", None)
        self.llm_module = importlib.import_module("src.llm")

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.original_env)
        sys.modules.pop("src.llm", None)

    def test_defaults_to_local_vllm_endpoint(self):
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("OPENAI_BASE_URL", None)
        os.environ.pop("OPENAI_MODEL", None)

        client = self.llm_module.LLMClient()

        self.assertEqual(client.base_url, "http://localhost:8002/v1")
        self.assertEqual(client.model, "openai/gpt-oss-20b")
        self.assertEqual(client.api_key, "local-vllm")
        self.assertEqual(client.client.base_url, client.base_url)

    def test_remote_endpoint_requires_api_key(self):
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ["OPENAI_BASE_URL"] = "https://api.openai.com/v1"
        os.environ["OPENAI_MODEL"] = "gpt-4o-mini"

        with self.assertRaises(ValueError):
            self.llm_module.LLMClient()

    def test_explicit_values_override_defaults(self):
        client = self.llm_module.LLMClient(
            api_key="custom-key",
            base_url="https://example.com/v1",
            model="custom-model",
        )

        self.assertEqual(client.api_key, "custom-key")
        self.assertEqual(client.base_url, "https://example.com/v1")
        self.assertEqual(client.model, "custom-model")


if __name__ == "__main__":
    unittest.main()
