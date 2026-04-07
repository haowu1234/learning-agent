# learning-agent

一个从零实现 AI Agent 的学习项目，包含单 Agent（ReAct / Plan-and-Execute）和 Multi-Agent 协作模式。

## 快速试玩

- 运行交互式 Playground：`python3 playground.py`
- 直接执行一个自定义任务：`python3 playground.py --task "先查询北京和上海天气，再比较哪个更暖和。"`
- 指定 Plan-and-Execute 执行器：`python3 playground.py --mode plan_and_execute --executor-mode function_calling`
- 指定自定义系统提示词：`python3 playground.py --system-prompt "你是一个严谨的研究助手。"`

### Playground 交互命令

- `:help`：查看帮助
- `:examples`：查看内置示例任务
- `:config`：查看当前运行配置
- `:reset`：清空历史
- `:multiline`：输入多行复杂任务，最后用 `:end` 提交
- `:set mode=function_calling quiet=true`：运行时切换模式或参数
- `:quit`：退出

默认会连接本地 OpenAI 兼容端点：`http://localhost:8002/v1`
