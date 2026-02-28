"""Multi-Agent 基础模块单元测试

测试 Message、SharedState、AgentRole 等基础组件。
不依赖 LLM API。
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from src.multi.message import Message, MessageType
from src.multi.shared_state import SharedState
from src.multi.roles import AgentRole, get_role, ROLES


class TestMessage(unittest.TestCase):
    def test_create_message(self):
        msg = Message(sender="a", receiver="b", content="hello")
        self.assertEqual(msg.sender, "a")
        self.assertEqual(msg.receiver, "b")
        self.assertEqual(msg.content, "hello")
        self.assertEqual(msg.msg_type, MessageType.TASK)

    def test_message_type(self):
        msg = Message(
            sender="a", receiver="b", content="done",
            msg_type=MessageType.RESULT,
        )
        self.assertEqual(msg.msg_type, MessageType.RESULT)

    def test_message_metadata(self):
        msg = Message(
            sender="a", receiver="b", content="data",
            metadata={"score": 0.9},
        )
        self.assertEqual(msg.metadata["score"], 0.9)

    def test_repr(self):
        msg = Message(sender="agent_a", receiver="agent_b", content="short")
        r = repr(msg)
        self.assertIn("agent_a", r)
        self.assertIn("agent_b", r)


class TestSharedState(unittest.TestCase):
    def setUp(self):
        self.state = SharedState()

    def test_initial_state(self):
        self.assertEqual(self.state.status, "idle")
        self.assertEqual(self.state.current_step, 0)
        self.assertEqual(len(self.state.results), 0)

    def test_add_result(self):
        self.state.add_result("researcher", "some findings")
        self.assertEqual(self.state.results["researcher"], "some findings")
        self.assertEqual(len(self.state.messages), 1)
        self.assertEqual(self.state.messages[0].msg_type, MessageType.RESULT)

    def test_get_result(self):
        self.state.add_result("analyst", "analysis done")
        self.assertEqual(self.state.get_result("analyst"), "analysis done")
        self.assertIsNone(self.state.get_result("nonexistent"))

    def test_get_all_results(self):
        self.state.add_result("a", "result_a")
        self.state.add_result("b", "result_b")
        text = self.state.get_all_results()
        self.assertIn("result_a", text)
        self.assertIn("result_b", text)

    def test_get_all_results_empty(self):
        self.assertEqual(self.state.get_all_results(), "(暂无结果)")

    def test_get_messages_for(self):
        self.state.add_message(Message(
            sender="a", receiver="b", content="for b",
        ))
        self.state.add_message(Message(
            sender="a", receiver="all", content="broadcast",
        ))
        self.state.add_message(Message(
            sender="a", receiver="c", content="for c",
        ))
        msgs = self.state.get_messages_for("b")
        self.assertEqual(len(msgs), 2)  # "for b" + "broadcast"

    def test_reset(self):
        self.state.task = "test"
        self.state.add_result("a", "result")
        self.state.status = "executing"
        self.state.reset()
        self.assertEqual(self.state.status, "idle")
        self.assertEqual(self.state.task, "")
        self.assertEqual(len(self.state.results), 0)

    def test_summary(self):
        s = self.state.summary()
        self.assertIn("idle", s)
        self.assertIn("0", s)


class TestAgentRole(unittest.TestCase):
    def test_create_role(self):
        role = AgentRole(
            name="test",
            description="test role",
            system_prompt="you are a test",
            tools=["search"],
        )
        self.assertEqual(role.name, "test")
        self.assertEqual(role.tools, ["search"])

    def test_get_predefined_role(self):
        researcher = get_role("researcher")
        self.assertEqual(researcher.name, "researcher")
        self.assertIn("search", researcher.tools)

    def test_get_nonexistent_role(self):
        with self.assertRaises(KeyError):
            get_role("nonexistent_role")

    def test_all_predefined_roles(self):
        for name, role in ROLES.items():
            self.assertEqual(role.name, name)
            self.assertTrue(len(role.description) > 0)
            self.assertTrue(len(role.system_prompt) > 0)


class TestPipelineStep(unittest.TestCase):
    def test_import(self):
        from src.multi.message import PipelineStep
        step = PipelineStep(agent_name="researcher", task_template="{task}")
        self.assertEqual(step.agent_name, "researcher")
        self.assertEqual(step.retry, 1)
        self.assertIsNone(step.transform)


if __name__ == "__main__":
    unittest.main()
