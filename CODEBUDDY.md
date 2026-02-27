# CODEBUDDY.md This file provides guidance to CodeBuddy when working with code in this repository.

## Project Overview

从零实现 ReAct (Reasoning + Acting) Agent，不依赖 LangChain 等框架，深入理解 Agent 核心原理。使用 OpenAI 兼容 API。

## Commands

```bash
# 安装依赖
pip install -r requirements.txt

# 配置 API Key（复制并编辑 .env）
cp .env.example .env

# 运行单元测试（不需要 API Key）
python3 -m unittest discover tests -v

# 运行示例（需要配置 API Key）
python3 -m examples.01_simple_tool
python3 -m examples.02_react_loop
python3 -m examples.03_multi_tools
python3 -m examples.04_with_memory
```

## Architecture

```
src/
├── llm.py              # LLM 客户端，封装 OpenAI ChatCompletion，支持 base_url 切换模型提供商
├── tools/
│   ├── base.py         # Tool 抽象基类 + ToolRegistry 注册中心
│   ├── calculator.py   # 数学计算（ast 安全解析）
│   ├── weather.py      # 天气查询（Mock）
│   └── search.py       # 网页搜索（Mock）
├── agent/
│   ├── prompt.py       # 两种模式的系统 Prompt 模板
│   └── react.py        # ReActAgent 核心：function_calling / text_parsing 双模式
└── memory/
    └── history.py      # 对话历史管理，支持按轮数截断
```

**核心流程**：`ReActAgent.run()` → LLM 推理 → 解析 tool_calls → `ToolRegistry.execute()` → 结果回填上下文 → 循环直到最终答案。

**两种工具调用模式**：
- `function_calling`：使用 OpenAI 原生 tools 参数，结构化可靠（生产推荐）
- `text_parsing`：Prompt 引导 + 正则提取，兼容所有 LLM（学习用）

**新增工具**：继承 `Tool` 基类，实现 `name`/`description`/`parameters`/`run()`，然后 `registry.register()` 注册即可。
