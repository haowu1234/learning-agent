# learning-agent

<p align="center">
  <strong>一个从零实现 AI Agent 的学习型项目</strong>
</p>

<p align="center">
  不依赖 LangChain 等重量级框架，聚焦 Agent 的核心抽象、执行链路与多智能体协作机制。<br />
  支持 OpenAI 兼容接口，默认面向本地 <code>vLLM</code> 端点开箱即用。
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" />
  <img alt="OpenAI Compatible" src="https://img.shields.io/badge/API-OpenAI%20Compatible-412991" />
  <img alt="vLLM Ready" src="https://img.shields.io/badge/Local-vLLM%20Ready-0F766E" />
  <img alt="Architecture" src="https://img.shields.io/badge/Architecture-Single%20Agent%20%2B%20Multi--Agent-111827" />
</p>

---

## 项目简介

`learning-agent` 是一个面向**学习、实验与原理理解**的 AI Agent 项目。
它从最基础的工具调用开始，逐步扩展到：

- 单 Agent 的 `ReAct` 推理循环
- `function_calling` 与 `text_parsing` 两种工具调用协议
- `plan_and_execute` 任务拆解与执行策略
- Multi-Agent 的 `Pipeline`、`Orchestrator`、`Debate` 协作模式
- 交互式 `Playground`，用于快速体验和自定义任务测试

这个项目的目标不是“封装一个黑盒框架”，而是把 Agent 系统的关键部件拆开，让你可以真正看懂、修改、验证并扩展。

---

## 为什么选择这个项目

- **从零实现，结构清晰**：核心流程可直接阅读，不依赖复杂外部编排框架。
- **适合学习 Agent 原理**：可以清楚看到 Prompt、工具调用、状态管理、计划生成、子任务执行等关键环节。
- **同时覆盖单 Agent 与 Multi-Agent**：既能研究最小闭环，也能上升到协作系统。
- **默认支持本地 vLLM**：便于在本地模型或私有推理服务上做实验。
- **工程上可持续扩展**：工具、角色、协作模式、CLI 入口都可以独立增强。

---

## 核心特性

### 单 Agent

- **`function_calling`**：基于 OpenAI 原生 `tools` 协议的工具调用方式，适合生产风格实验。
- **`text_parsing`**：通过 Prompt + 文本格式约束 + 正则解析来模拟工具调用，更利于理解 Agent 底层机制。
- **`plan_and_execute`**：先让 Planner 产出步骤计划，再逐步执行子任务，最后统一汇总，适合复杂、长链路任务。
- **对话历史管理**：支持多轮上下文保留与重置。
- **可定制 system prompt**：便于注入角色、风格与任务边界。

### Multi-Agent

- **`Pipeline`**：固定顺序接力执行，适合研究 → 分析 → 写作等线性流程。
- **`Orchestrator`**：由 Planner 动态规划和分派任务，适合复杂问题拆解。
- **`Debate`**：多角色独立思考、互相讨论，再由裁判汇总，适合方案对比与观点竞争。
- **角色隔离工具集**：每个 Agent 可拥有不同工具权限。
- **Hook 机制**：支持 `on_agent_start`、`on_agent_finish`、`on_step_complete`、`on_error` 等关键事件。

### 开发体验

- **交互式 `playground.py`**：无需改代码即可体验不同模式与任务。
- **默认连接本地 vLLM**：`http://localhost:8002/v1`
- **单元测试覆盖核心模块**：包括 `tools`、`agent`、`multi`、`llm`、`playground`。

---

## 能力矩阵

### 单 Agent 模式

| 模式 | 适用场景 | 特点 |
| --- | --- | --- |
| `function_calling` | 工具调用稳定性优先 | 结构化、可靠、接近生产方案 |
| `text_parsing` | 教学、调试、理解 Agent 输出格式 | 透明、易观察、兼容不支持原生工具调用的模型 |
| `plan_and_execute` | 长任务、复杂任务、需显式拆解的问题 | 先规划、再执行、最后汇总；可搭配不同执行器 |

### Multi-Agent 模式

| 模式 | 适用场景 | 特点 |
| --- | --- | --- |
| `Pipeline` | 固定顺序协作 | 简单稳定，容易调试 |
| `Orchestrator` | 动态拆解复杂任务 | 灵活，具备规划能力 |
| `Debate` | 方案对比、观点竞争 | 有助于获得更全面的结论 |

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

项目依赖保持精简：

- `openai`
- `httpx`
- `python-dotenv`

### 2. 配置模型端点

复制环境变量模板：

```bash
cp .env.example .env
```

默认配置已经指向本地 `vLLM` OpenAI 兼容端点：

```env
OPENAI_BASE_URL=http://localhost:8002/v1
OPENAI_MODEL=openai/gpt-oss-20b
```

如果你使用的是本地 `localhost` 端点，代码会自动补一个占位 API Key；
如果切换到 OpenAI、DeepSeek、DashScope 等远端服务，则需要提供真实 `OPENAI_API_KEY`。

### 3. 运行测试

```bash
python3 -m unittest discover tests -v
```

### 4. 开始试玩

```bash
python3 playground.py
```

---

## Playground：最适合体验项目的入口

`playground.py` 是本项目的交互式 CLI 入口，适合快速验证不同 Agent 模式和自定义任务。

### 常用启动方式

```bash
python3 playground.py
python3 playground.py --task "先查询北京和上海天气，再比较哪个更暖和。"
python3 playground.py --mode plan_and_execute --executor-mode function_calling
python3 playground.py --system-prompt "你是一个严谨的研究助手。"
```

### 交互命令

- `:help`：查看帮助
- `:examples`：查看内置示例任务
- `:config`：查看当前运行配置
- `:reset`：清空历史
- `:multiline`：输入多行复杂任务，最后用 `:end` 提交
- `:set mode=function_calling quiet=true`：运行时修改配置
- `:quit`：退出 Playground

### 推荐体验方式

1. 先用 `plan_and_execute` 处理一个明显需要拆步骤的问题
2. 再切到 `function_calling`，感受“边推理边调工具”的差异
3. 最后尝试 `text_parsing`，观察 Prompt 驱动工具调用的输出格式

---

## 示例一览

项目提供了从入门到进阶的完整示例：

| 文件 | 说明 |
| --- | --- |
| `examples/01_simple_tool.py` | 理解一次最小工具调用 |
| `examples/02_react_loop.py` | 体验 ReAct 循环，以及 `function_calling` / `text_parsing` 对比 |
| `examples/03_multi_tools.py` | 观察 Agent 如何在多个工具之间自主选择 |
| `examples/04_with_memory.py` | 体验带对话历史的多轮交互 |
| `examples/05_pipeline.py` | 体验 Multi-Agent 流水线模式 |
| `examples/06_orchestrator.py` | 体验编排者动态规划与分派 |
| `examples/07_debate.py` | 体验多角色辩论与裁决 |
| `examples/08_plan_and_execute.py` | 体验 Planner → Executor → Summarizer 的任务闭环 |

运行方式示例：

```bash
python3 -m examples.08_plan_and_execute
```

---

## 架构概览

```text
src/
├── llm.py                  # LLM 客户端，封装 OpenAI 兼容接口，默认连接本地 vLLM
├── tools/
│   ├── base.py             # Tool 抽象基类 + ToolRegistry 注册中心
│   ├── calculator.py       # 数学计算（安全解析）
│   ├── weather.py          # 天气查询（Mock）
│   └── search.py           # 搜索工具（Mock）
├── agent/
│   ├── prompt.py           # Agent 的 Prompt 模板
│   └── react.py            # 单 Agent 核心实现：ReAct + Plan-and-Execute
├── memory/
│   └── history.py          # 多轮对话历史管理
└── multi/
    ├── message.py          # Message、PipelineStep 等数据结构
    ├── shared_state.py     # 共享状态黑板
    ├── roles.py            # 角色模板定义
    ├── base.py             # Multi-Agent 基类
    ├── pipeline.py         # 流水线模式
    ├── orchestrator.py     # 编排者模式
    └── debate.py           # 辩论模式
```

### 单 Agent 执行链路

- 用户输入任务
- Agent 根据模式选择执行策略
- 必要时调用工具并回填观察结果
- 在 `plan_and_execute` 下先生成计划，再逐步执行子任务
- 最终产出答案并写入对话历史

### Multi-Agent 执行链路

- `BaseMultiAgent` 管理角色、工具和共享状态
- 各模式通过继承基类实现自己的协作逻辑
- 子任务执行仍然复用 `ReActAgent`
- 结果通过共享状态与消息结构沉淀，便于调试和扩展

---

## 设计理念

### 1. 保持抽象简单

这个项目尽量避免引入多层封装和黑盒 DSL，而是围绕几个清晰的核心概念组织代码：

- `LLMClient`
- `Tool` / `ToolRegistry`
- `ReActAgent`
- `ConversationHistory`
- `BaseMultiAgent` / `SharedState`

### 2. 教学价值优先于“魔法体验”

项目保留了 `text_parsing` 这样的实现方式，不是因为它更先进，而是因为它更能帮助你理解 Agent 的底层工作方式。

### 3. 演进式架构

你可以从单 Agent 开始，逐步演进到：

- 多工具协作
- 记忆增强
- 显式计划执行
- 多智能体协作
- 后续进一步加入反思、验证器、长期记忆等能力

---

## 如何扩展

### 新增工具

在 `src/tools/` 下新增一个继承 `Tool` 的类，实现：

- `name`
- `description`
- `parameters`
- `run()`

然后注册到 `ToolRegistry` 即可。

### 新增角色

在 `src/multi/roles.py` 中添加新的 `AgentRole` 定义，指定：

- 角色名
- 描述
- system prompt
- 可用工具列表

### 新增协作模式

继承 `BaseMultiAgent` 并实现 `run()`，即可把新的协作策略接入现有系统。

---

## 质量保障

测试目录覆盖了项目的关键组件：

- `tests/test_tools.py`
- `tests/test_agent.py`
- `tests/test_multi.py`
- `tests/test_llm.py`
- `tests/test_playground.py`

这些测试重点验证：

- 工具注册与执行
- Agent 模式切换
- Plan-and-Execute 的规划与回退逻辑
- Multi-Agent 基础协作结构
- 本地 vLLM 默认配置
- Playground 的运行时配置更新能力

---

## 适合谁使用

这个项目尤其适合：

- 想系统学习 Agent 原理的开发者
- 想自己实现 Agent，而不是只调用框架的工程师
- 想实验本地模型、OpenAI 兼容接口与自定义工具链的人
- 想研究单 Agent 与 Multi-Agent 差异的学习者

---

## 路线图

接下来适合继续演进的方向包括：

- `Reflection / Critic` 反思与纠错机制
- `Verifier` 结果验证层
- 更丰富的长期记忆策略
- 更强的可观测性与执行追踪
- 更多真实工具接入（Web、数据库、代码执行等）

---

## 贡献

欢迎通过 Issue 和 Pull Request 参与改进。

如果你准备贡献代码，建议先：

1. 运行测试，确保当前环境正常
2. 阅读 `examples/` 和 `src/` 的核心结构
3. 尽量保持实现风格简洁、可读、可教学
4. 为新增能力补充对应测试与示例

---

## 结语

如果你想真正理解一个 Agent 系统是如何从 Prompt、工具、记忆、计划、状态管理一步步搭起来的，`learning-agent` 会是一个合适的起点。

它既可以作为学习项目，也足够作为你继续实验更先进 Agent 模式的基础骨架。