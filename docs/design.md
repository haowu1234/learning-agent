# ReAct Agent 学习项目设计方案

## 目标

从零实现一个 ReAct Agent，不依赖 LangChain 等框架，深入理解 Agent 的核心原理。

## 技术选型

- **语言**：Python 3.11+
- **LLM 接口**：OpenAI API（兼容其他 OpenAI 兼容接口，如 DeepSeek、通义千问等）
- **依赖**：尽量精简，仅 `openai` + `httpx`

## 项目结构

```
learning-agent/
├── README.md
├── requirements.txt
├── .env.example            # API Key 配置模板
├── src/
│   ├── __init__.py
│   ├── llm.py              # LLM 调用封装
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── base.py         # Tool 基类，定义工具注册机制
│   │   ├── search.py       # 示例工具：网页搜索
│   │   ├── calculator.py   # 示例工具：数学计算
│   │   └── weather.py      # 示例工具：天气查询
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── react.py        # ReAct Agent 核心循环
│   │   └── prompt.py       # Prompt 模板
│   └── memory/
│       ├── __init__.py
│       └── history.py      # 对话历史管理
├── examples/
│   ├── 01_simple_tool.py   # 第1课：理解工具调用
│   ├── 02_react_loop.py    # 第2课：实现 ReAct 循环
│   ├── 03_multi_tools.py   # 第3课：多工具协作
│   └── 04_with_memory.py   # 第4课：加入记忆能力
└── tests/
    ├── test_tools.py
    └── test_agent.py
```

## 分阶段实现计划

### 阶段 1：基础设施

| 文件 | 内容 |
|------|------|
| `llm.py` | 封装 OpenAI ChatCompletion 调用，支持配置 base_url 以兼容其他模型 |
| `tools/base.py` | 定义 `Tool` 基类：`name`、`description`、`parameters`（JSON Schema）、`run()` 方法；实现 `ToolRegistry` 用于注册和查找工具 |

### 阶段 2：ReAct 核心循环

| 文件 | 内容 |
|------|------|
| `agent/prompt.py` | ReAct 系统提示词模板，指导 LLM 按 `Thought → Action → Observation` 格式输出 |
| `agent/react.py` | Agent 主循环：1) 将用户问题 + 历史发给 LLM → 2) 解析 LLM 输出，提取 Action → 3) 执行工具 → 4) 将 Observation 拼回上下文 → 5) 判断是否终止 |

核心伪代码：

```python
class ReActAgent:
    def run(self, query: str) -> str:
        messages = [system_prompt, user_query]

        for step in range(max_steps):
            # 1. LLM 推理
            response = llm.chat(messages)

            # 2. 检查是否有 tool_call
            if response.has_tool_calls():
                # 3. 执行工具
                tool_name, tool_args = response.parse_tool_call()
                observation = self.tools[tool_name].run(**tool_args)

                # 4. 将结果加入上下文
                messages.append(assistant_message)
                messages.append(tool_result(observation))
            else:
                # 5. 无工具调用，返回最终答案
                return response.content

        return "达到最大步数，未能得出结论"
```

### 阶段 3：示例工具

| 工具 | 说明 |
|------|------|
| `calculator.py` | 安全地执行数学表达式（用 `ast.literal_eval` 或受限 eval） |
| `weather.py` | 模拟天气查询（返回 mock 数据，无需真实 API） |
| `search.py` | 使用 httpx 调用免费搜索 API 或返回 mock 数据 |

### 阶段 4：记忆与对话

| 文件 | 内容 |
|------|------|
| `memory/history.py` | 管理多轮对话历史，支持 token 截断策略 |

## 渐进式学习路径

```
第1课 (01_simple_tool.py)
  → 理解 OpenAI Function Calling 机制
  → 手动调用一个工具并返回结果

第2课 (02_react_loop.py)
  → 实现完整的 ReAct 循环
  → 观察 Thought → Action → Observation 的过程

第3课 (03_multi_tools.py)
  → 注册多个工具，Agent 自主选择
  → 观察 Agent 如何分解复杂问题

第4课 (04_with_memory.py)
  → 支持多轮对话
  → 理解上下文窗口管理
```

## 两种实现方式对比

项目会同时展示两种工具调用方式：

1. **OpenAI Function Calling**（原生 `tools` 参数）—— 生产级方案
2. **纯文本解析**（Prompt 引导 LLM 输出特定格式，正则提取）—— 理解底层原理
