---
name: report-from-materials
description: 当用户要求基于本地文档、会议纪要、需求说明、代码片段或其他已有文件做总结、提炼结论、撰写结构化报告时使用。尤其适合“先读文件再回答”“根据这份材料输出结论”“把多份资料整理成报告”等请求。
---

# Report From Materials

用于把本地文件里的信息整理成结构化输出。

## Architecture

```text
report-from-materials/
├── SKILL.md
├── references/analysis-checklist.md
└── templates/report-outline.md
```

## Workflow

1. 如果用户已经给出文件路径，先调用 `read_local_file` 读取对应文件。
2. 读完主材料后，再读取 `references/analysis-checklist.md`，按检查清单提取事实、结论、风险、待确认项。
3. 当用户明确需要“报告 / 汇总 / 摘要 / 结论”这类结构化输出时，再读取 `templates/report-outline.md` 作为组织模板。
4. 如果材料里涉及简单数值比较或换算，可使用 `calculator`。
5. 如果需要补充公开背景信息，再使用 `search`，并把“来自本地材料”与“来自补充检索”的信息分开说明。

## Output Rules

- 先说结论，再展开证据。
- 明确区分“材料中明确写明”“根据材料推断”“仍需确认”三类信息。
- 不要编造材料里没有的事实。
- 若用户只要简短摘要，就压缩为 3-5 个要点。
- 若用户要求正式汇报，优先使用模板中的标题层级。
