"""Skills 模块单元测试。"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.skills.loader import load_skills
from src.skills.parser import parse_skill_file
from src.skills.prompt import get_skills_prompt_section


def _write_skill(skill_dir: Path, name: str, description: str) -> None:
    skill_dir.mkdir(parents=True, exist_ok=True)
    content = (
        f"---\nname: {name}\ndescription: {description}\n---\n\n"
        f"# {name}\n"
    )
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")


class TestSkillParser(unittest.TestCase):
    def test_parse_skill_file_supports_multiline_description(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_file = Path(tmp) / "SKILL.md"
            skill_file.write_text(
                "---\n"
                "name: multiline-skill\n"
                "description: >\n"
                "  第一行描述\n"
                "  第二行描述\n"
                "---\n\n"
                "# body\n",
                encoding="utf-8",
            )

            skill = parse_skill_file(skill_file, category="public")

        self.assertIsNotNone(skill)
        self.assertEqual(skill.name, "multiline-skill")
        self.assertEqual(skill.description, "第一行描述 第二行描述")


class TestSkillLoader(unittest.TestCase):
    def test_load_skills_discovers_nested_dirs_and_prefers_custom(self):
        with tempfile.TemporaryDirectory() as tmp:
            skills_root = Path(tmp) / "skills"
            _write_skill(skills_root / "public" / "reports" / "summary", "shared-skill", "public version")
            _write_skill(skills_root / "custom" / "shared-skill", "shared-skill", "custom version")
            _write_skill(skills_root / "public" / "notes" / "reader", "reader-skill", "reader")

            skills = load_skills(skills_path=skills_root)
            by_name = {skill.name: skill for skill in skills}

        self.assertEqual(set(by_name), {"reader-skill", "shared-skill"})
        self.assertEqual(by_name["shared-skill"].category, "custom")
        self.assertEqual(by_name["shared-skill"].description, "custom version")
        self.assertEqual(by_name["reader-skill"].skill_path, "notes/reader")

    def test_load_skills_can_filter_available_skills(self):
        with tempfile.TemporaryDirectory() as tmp:
            skills_root = Path(tmp) / "skills"
            _write_skill(skills_root / "public" / "a", "skill-a", "A")
            _write_skill(skills_root / "public" / "b", "skill-b", "B")

            skills = load_skills(skills_path=skills_root, available_skills={"skill-b"})

        self.assertEqual([skill.name for skill in skills], ["skill-b"])


class TestSkillPrompt(unittest.TestCase):
    def test_get_skills_prompt_section_contains_filtered_skill_locations(self):
        with tempfile.TemporaryDirectory() as tmp:
            skills_root = Path(tmp) / "skills"
            _write_skill(skills_root / "public" / "report", "report-from-materials", "基于材料输出报告")
            _write_skill(skills_root / "public" / "other", "other-skill", "其他 skill")

            prompt = get_skills_prompt_section(
                {"report-from-materials"},
                skills_path=skills_root,
            )

        self.assertIn("<skill_system>", prompt)
        self.assertIn("report-from-materials", prompt)
        self.assertNotIn("other-skill", prompt)
        self.assertIn("read_local_file", prompt)
        self.assertIn("SKILL.md", prompt)


if __name__ == "__main__":
    unittest.main()
