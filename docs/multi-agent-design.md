# Multi-Agent 协作系统设计方案

## 设计目标

在现有 ReAct Agent 基础上，构建一个优雅、易扩展的 Multi-Agent 协作框架，支持多种协作模式。

## 设计原则

1. **复用现有 ReActAgent** —— 每个 Agent 仍然是 ReActAgent，Multi-Agent 是更高层的编排
2. **协作模式可插拔** —— Pipeline、Orchestrator、Debate 通过统一接口实现，随时切换
3. **消息驱动** —— Agent 间通过标准化消息通信，解耦具体实现
4. **共享状态** —— 提供可观测的共享上下文，便于调试和追踪

## 核心抽象

### 1. AgentRole（角色定义）

每个 Agent 不再只是"通用助手"，而是有明确角色：

```python
@dataclass
class AgentRole:
    name: str               # 角色名，如 "researcher"
    description: str        # 角色描述，如 "负责搜索和收集信息"
    system_prompt: str      # 该角色专属的系统提示词
    tools: list[str]        # 该角色可用的工具名列表
```

### 2. Message（标准消息）

Agent 间通信的统一格式：

```python
@dataclass
class Message:
    sender: str             # 发送者角色名
    receiver: str           # 接收者角色名（"all" 表示广播）
    content: str            # 消息内容
    msg_type: MessageType   # TASK / RESULT / FEEDBACK / SYSTEM
    metadata: dict          # 附加数据（如中间结果、置信度等）
```

### 3. SharedState（共享状态）

所有 Agent 可读写的公共黑板：

```python
class SharedState:
    task: str                           # 原始任务描述
    plan: list[str]                     # 执行计划
    results: dict[str, str]             # 各 Agent 的执行结果
    messages: list[Message]             # 完整消息历史
    status: Literal["planning", "executing", "reviewing", "done", "failed"]
    current_step: int
    max_steps: int
```

### 4. BaseMultiAgent（协作基类）

所有协作模式的统一接口：

```python
class BaseMultiAgent(ABC):
    def __init__(self, agents: dict[str, ReActAgent], shared_state: SharedState): ...

    @abstractmethod
    def run(self, task: str) -> str:
        """执行多智能体协作任务"""

    def _dispatch(self, agent_name: str, task: str) -> str:
        """将子任务分派给指定 Agent"""

    def _broadcast(self, message: Message) -> None:
        """向所有 Agent 广播消息"""
```

## 三种协作模式

### 模式 1：Pipeline（流水线）

```
task → [Agent A] → result_a → [Agent B] → result_b → [Agent C] → final
```

实现要点：
- agents 按顺序执行
- 支持定义 transform 函数，将上一步输出转换为下一步输入
- 任一环节失败可配置：跳过 / 重试 / 终止

```python
class PipelineMultiAgent(BaseMultiAgent):
    def __init__(self, agents, shared_state, pipeline: list[PipelineStep]): ...

@dataclass
class PipelineStep:
    agent_name: str
    task_template: str      # 支持 {prev_result} 占位符
    transform: Callable     # 可选的结果转换函数
    retry: int = 1
```

### 模式 2：Orchestrator（编排者）

```
task → [Planner] → plan → [Orchestrator 循环分派] → [汇总] → final
              ↑                                   │
              └──── 如果结果不满意，重新规划 ────────┘
```

实现要点：
- Orchestrator 本身也是一个 LLM Agent，负责拆解任务和分派
- 每一步动态决定调用哪个 Agent
- 支持 re-plan：如果某步结果不理想，可以调整计划

```python
class OrchestratorMultiAgent(BaseMultiAgent):
    def __init__(self, agents, shared_state, planner_llm: LLMClient): ...

    def _plan(self, task: str) -> list[PlanStep]:
        """用 LLM 生成执行计划"""

    def _should_replan(self, step_result: str) -> bool:
        """判断是否需要重新规划"""
```

### 模式 3：Debate（辩论）

```
Round 1: 所有 Agent 独立回答
Round 2: 看到其他人的回答后，修正自己的观点
Round 3: 裁判 Agent 汇总，输出最终结论
```

实现要点：
- 每轮所有 Agent 并行（概念上）生成回答
- 下一轮将所有人的回答作为上下文
- 最终由 judge Agent 综合裁决

```python
class DebateMultiAgent(BaseMultiAgent):
    def __init__(self, agents, shared_state, judge: ReActAgent, max_rounds: int = 3): ...

    def _collect_opinions(self, topic: str) -> dict[str, str]:
        """收集所有 Agent 的观点"""

    def _judge(self, opinions: dict[str, str]) -> str:
        """裁判 Agent 做最终裁决"""
```

## 项目结构（新增部分）

```
src/
├── agent/
│   ├── react.py                # 已有，无需修改
│   └── prompt.py               # 已有，新增角色 prompt 模板
├── multi/
│   ├── __init__.py
│   ├── message.py              # Message、MessageType 定义
│   ├── shared_state.py         # SharedState 共享状态
│   ├── roles.py                # 预定义角色模板（研究员、分析师、写作者等）
│   ├── base.py                 # BaseMultiAgent 抽象基类
│   ├── pipeline.py             # PipelineMultiAgent
│   ├── orchestrator.py         # OrchestratorMultiAgent
│   └── debate.py               # DebateMultiAgent
examples/
├── 05_pipeline.py              # 示例：研究→分析→写作 流水线
├── 06_orchestrator.py          # 示例：智能编排完成复杂研究任务
└── 07_debate.py                # 示例：多 Agent 辩论得出最佳方案
tests/
├── test_multi_message.py       # 消息模块测试
├── test_multi_pipeline.py      # Pipeline 模式测试（mock LLM）
└── test_multi_shared_state.py  # 共享状态测试
```

## 扩展性设计

### 新增协作模式

只需继承 `BaseMultiAgent`，实现 `run()` 方法：

```python
class MyCustomMultiAgent(BaseMultiAgent):
    def run(self, task: str) -> str:
        # 自定义协作逻辑
        ...
```

### 新增角色

在 `roles.py` 中添加角色定义即可：

```python
ROLES = {
    "researcher": AgentRole(
        name="researcher",
        description="信息收集专家",
        system_prompt="你是一个研究员...",
        tools=["search"],
    ),
    "coder": AgentRole(
        name="coder",
        description="编程专家",
        system_prompt="你是一个程序员...",
        tools=["calculator"],
    ),
    # 新增角色只需在这里添加...
}
```

### Hook 机制

在关键节点提供 hook，便于监控和自定义：

```python
class BaseMultiAgent:
    def on_agent_start(self, agent_name: str, task: str) -> None: ...
    def on_agent_finish(self, agent_name: str, result: str) -> None: ...
    def on_step_complete(self, step: int, state: SharedState) -> None: ...
    def on_error(self, agent_name: str, error: Exception) -> None: ...
```

## 实现顺序

| 阶段 | 内容 | 依赖 |
|------|------|------|
| 1 | `message.py` + `shared_state.py` + `roles.py` | 无 |
| 2 | `base.py` 基类 | 阶段 1 |
| 3 | `pipeline.py` + `05_pipeline.py` | 阶段 2 |
| 4 | `orchestrator.py` + `06_orchestrator.py` | 阶段 2 |
| 5 | `debate.py` + `07_debate.py` | 阶段 2 |
| 6 | 测试 + 文档 | 全部 |

## 示例场景预览

### Pipeline 示例：深度研究报告

```
用户："帮我研究 AI Agent 的最新发展趋势，写一篇分析报告"

[研究员 Agent] → 搜索信息、整理素材
    ↓ 传递：搜索结果摘要
[分析师 Agent] → 数据分析、提炼观点
    ↓ 传递：关键观点列表
[写作者 Agent] → 撰写结构化报告
    ↓ 输出：完整报告
```

### Orchestrator 示例：复杂问题求解

```
用户："对比北京、上海、深圳三个城市的宜居性，给出排名和理由"

[编排者] 规划：
  Step 1: 研究员查询三城天气
  Step 2: 研究员搜索三城生活成本
  Step 3: 分析师综合分析
  Step 4: 写作者输出报告

[编排者] 逐步分派执行，动态调整...
```

### Debate 示例：技术选型辩论

```
用户："Python vs Go，哪个更适合开发微服务？"

Round 1:
  [Python专家]: Python 灵活、生态丰富...
  [Go专家]: Go 高性能、并发强...
  [架构师]: 从架构角度分析...

Round 2: 互相反驳

Round 3: [裁判] 综合裁决，给出建议
```
