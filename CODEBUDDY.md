# CODEBUDDY.md This file provides guidance to CodeBuddy when working with code in this repository.

## Project Overview

从零实现 AI Agent 系统，涵盖单 Agent（ReAct）和 Multi-Agent 协作，不依赖 LangChain 等框架。使用 OpenAI 兼容 API（支持 vLLM 本地部署）。

## Commands

```bash
# 安装依赖
pip install -r requirements.txt

# 配置 API Key（复制并编辑 .env）
cp .env.example .env

# 运行单元测试（不需要 API Key）
python3 -m unittest discover tests -v

# 运行示例（需要配置 API Key）
python3 -m examples.01_simple_tool       # 单次工具调用
python3 -m examples.02_react_loop        # ReAct 循环 + 双模式
python3 -m examples.03_multi_tools       # 多工具协作
python3 -m examples.04_with_memory       # 多轮对话记忆
python3 -m examples.05_pipeline          # Multi-Agent 流水线
python3 -m examples.06_orchestrator      # Multi-Agent 编排者
python3 -m examples.07_debate            # Multi-Agent 辩论
```

## Architecture

```
src/
├── llm.py                  # LLM 客户端，封装 OpenAI ChatCompletion，支持 base_url 切换
├── tools/
│   ├── base.py             # Tool 抽象基类 + ToolRegistry 注册中心
│   ├── calculator.py       # 数学计算（ast 安全解析）
│   ├── weather.py          # 天气查询（Mock）
│   └── search.py           # 网页搜索（Mock）
├── agent/
│   ├── prompt.py           # 系统 Prompt 模板
│   └── react.py            # ReActAgent：function_calling / text_parsing 双模式
├── memory/
│   └── history.py          # 对话历史管理，按轮数截断
└── multi/                  # Multi-Agent 协作框架
    ├── message.py          # Message 消息定义 + PipelineStep 数据类
    ├── shared_state.py     # SharedState 共享状态黑板
    ├── roles.py            # AgentRole 预定义角色模板
    ├── base.py             # BaseMultiAgent 基类（Agent 管理、分派、Hook）
    ├── pipeline.py         # PipelineMultiAgent 流水线模式
    ├── orchestrator.py     # OrchestratorMultiAgent 编排者模式（LLM 自主规划）
    └── debate.py           # DebateMultiAgent 辩论模式（多轮讨论 + 裁判裁决）
```

**单 Agent 流程**：`ReActAgent.run()` → LLM 推理 → 解析 tool_calls → `ToolRegistry.execute()` → 结果回填 → 循环直到最终答案。

**Multi-Agent 架构**：`BaseMultiAgent` 基类管理多个 `ReActAgent` 实例，每个 Agent 通过 `AgentRole` 定义角色和可用工具。三种模式通过继承基类实现：
- **Pipeline**：固定顺序执行，`{prev_result}` 模板传递上下文
- **Orchestrator**：LLM Planner 动态生成 JSON 计划，逐步分派，最后汇总
- **Debate**：多轮辩论 + 裁判裁决，后续轮次可见对方观点

**扩展点**：
- 新增工具：继承 `Tool`，实现 `name`/`description`/`parameters`/`run()`
- 新增角色：在 `roles.py` 的 `ROLES` 字典中添加
- 新增协作模式：继承 `BaseMultiAgent`，实现 `run()` 方法
- Hook 机制：通过 `multi_agent.on("on_agent_start", callback)` 注册回调
