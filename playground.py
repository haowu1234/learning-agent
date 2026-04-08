"""交互式 Playground。

用法示例：
    python3 playground.py
    python3 playground.py --mode plan_and_execute --executor-mode function_calling
    python3 playground.py --task "先搜索 Python 的主要特性，再整理成 5 条学习建议。"
    python3 playground.py --system-prompt "你是一个严谨的研究助手。"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from src.agent.react import ReActAgent
from src.llm import LLMClient
from src.skills.loader import load_skills
from src.tools.base import ToolRegistry
from src.tools.calculator import CalculatorTool
from src.tools.mcp_registry import load_mcp_tools
from src.tools.read_local_file import ReadLocalFileTool
from src.tools.search import SearchTool
from src.tools.weather import WeatherTool

EXAMPLE_TASKS = [
    "先查询北京和上海的天气，再比较哪个城市更暖和，并给出一句出行建议。",
    "先搜索 Python 的主要特性，再整理成 5 条适合初学者的学习建议。",
    "如果北京今天温度乘以 3 再加上 100，结果是多少？请先确认天气，再完成计算。",
    "先搜索 AI Agent 的核心特征，再写一段适合分享给同事的简短介绍。",
]

HELP_TEXT = """可用命令：
:help                 查看帮助
:examples             查看内置示例任务
:config               查看当前配置
:skills               查看当前已加载的 skills
:reset                清空历史
:multiline            输入多行任务，最后用 :end 提交
:set k=v [k=v...]     运行时修改配置，如 :set mode=plan_and_execute max_plan_steps=5
:quit                 退出 Playground

支持动态修改的配置：
- mode=function_calling|text_parsing|plan_and_execute
- executor_mode=function_calling|text_parsing
- max_steps=<int>
- max_plan_steps=<int>
- max_replans=<int>
- enable_verifier=true|false
- keep_history=true|false
- quiet=true|false
"""


@dataclass
class PlaygroundConfig:
    mode: str = "plan_and_execute"
    executor_mode: str = "function_calling"
    max_steps: int = 10
    max_plan_steps: int = 4
    max_replans: int = 1
    enable_verifier: bool = True
    keep_history: bool = False
    quiet: bool = False
    system_prompt: str | None = None


def build_registry() -> ToolRegistry:
    """构建默认工具集。"""
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(WeatherTool())
    registry.register(ReadLocalFileTool())

    for tool in load_mcp_tools():
        registry.register(tool)

    if "search" not in registry:
        registry.register(SearchTool.from_env())
    return registry


def build_agent(config: PlaygroundConfig) -> ReActAgent:
    """根据配置构建 Agent。"""
    return ReActAgent(
        llm=LLMClient(),
        tool_registry=build_registry(),
        mode=config.mode,
        executor_mode=config.executor_mode,
        max_steps=config.max_steps,
        max_plan_steps=config.max_plan_steps,
        max_replans=config.max_replans,
        enable_verifier=config.enable_verifier,
        verbose=not config.quiet,
        system_prompt=config.system_prompt,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="learning-agent 交互式 Playground")
    parser.add_argument(
        "--mode",
        default="plan_and_execute",
        choices=ReActAgent.available_modes(),
        help="Agent 运行模式",
    )
    parser.add_argument(
        "--executor-mode",
        default="function_calling",
        choices=("function_calling", "text_parsing"),
        help="plan_and_execute 模式下每个子任务的执行模式",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="单次执行的最大推理步数",
    )
    parser.add_argument(
        "--max-plan-steps",
        type=int,
        default=4,
        help="plan_and_execute 模式下最多规划多少步",
    )
    parser.add_argument(
        "--max-replans",
        type=int,
        default=1,
        help="Verifier 未通过时最多允许重新规划多少次",
    )
    parser.add_argument(
        "--disable-verifier",
        action="store_true",
        help="关闭 Plan-and-Execute 的结果验收与自动重规划",
    )
    parser.add_argument(
        "--task",
        help="直接执行一次任务并退出，不进入交互模式",
    )
    parser.add_argument(
        "--keep-history",
        action="store_true",
        help="默认每次任务后自动 reset；开启后保留会话历史",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="关闭中间日志，只输出最终结果",
    )
    parser.add_argument(
        "--system-prompt",
        help="为 Playground 指定一段自定义 system prompt",
    )
    parser.add_argument(
        "--system-prompt-file",
        help="从文件读取自定义 system prompt",
    )
    return parser


def args_to_config(args: argparse.Namespace) -> PlaygroundConfig:
    system_prompt = args.system_prompt
    if args.system_prompt_file:
        system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8")

    return PlaygroundConfig(
        mode=args.mode,
        executor_mode=args.executor_mode,
        max_steps=args.max_steps,
        max_plan_steps=args.max_plan_steps,
        max_replans=args.max_replans,
        enable_verifier=not args.disable_verifier,
        keep_history=args.keep_history,
        quiet=args.quiet,
        system_prompt=system_prompt,
    )


def print_examples() -> None:
    print("\n内置示例任务：")
    for index, task in enumerate(EXAMPLE_TASKS, 1):
        print(f"{index}. {task}")


def get_loaded_skills(agent: ReActAgent) -> list:
    """返回当前 Agent 实际可见的 skills。"""
    if "read_local_file" not in agent.tools:
        return []
    return load_skills(
        skills_path=agent.skills_path,
        enabled_only=True,
        available_skills=agent.available_skills,
    )


def skills_summary(agent: ReActAgent) -> str:
    """返回适合启动页展示的 skills 摘要。"""
    skills = get_loaded_skills(agent)
    if not skills:
        return "loaded_skills=0"
    names = ", ".join(skill.name for skill in skills)
    return f"loaded_skills={len(skills)} [{names}]"


def print_skills(agent: ReActAgent) -> None:
    """打印当前 Agent 已加载的 skills 详情。"""
    skills = get_loaded_skills(agent)
    if not skills:
        print("当前未加载任何 skills。")
        return

    print(f"当前已加载 {len(skills)} 个 skill：")
    for index, skill in enumerate(skills, 1):
        print(f"{index}. {skill.name} ({skill.category})")
        print(f"   path: {skill.file_path}")
        print(f"   desc: {skill.description}")


def config_summary(config: PlaygroundConfig, agent: ReActAgent | None = None) -> str:
    base_url = getattr(getattr(agent, "llm", None), "base_url", "(unknown)")
    model = getattr(getattr(agent, "llm", None), "model", "(unknown)")
    has_system_prompt = bool(config.system_prompt)

    search_backend = "(unknown)"
    if agent is not None:
        search_tool = agent.tools.get("search")
        if search_tool is not None and hasattr(search_tool, "backend_label"):
            search_backend = search_tool.backend_label()

    return (
        f"mode={config.mode}, executor_mode={config.executor_mode}, "
        f"max_steps={config.max_steps}, max_plan_steps={config.max_plan_steps}, "
        f"max_replans={config.max_replans}, enable_verifier={config.enable_verifier}, "
        f"keep_history={config.keep_history}, quiet={config.quiet}, "
        f"system_prompt={'on' if has_system_prompt else 'off'}, "
        f"base_url={base_url}, model={model}, search_backend={search_backend}"
    )


def print_welcome(config: PlaygroundConfig, agent: ReActAgent) -> None:
    print("=" * 72)
    print("learning-agent Playground")
    print("=" * 72)
    print(config_summary(config, agent))
    print(skills_summary(agent))
    print()
    print("输入你的任务后回车即可执行。")
    print("输入 :help 查看命令，输入 :examples 查看推荐测试任务，输入 :skills 查看已加载 skills。")
    print("默认每次执行后都会重置历史；如需保留上下文，请加 --keep-history")
    print("=" * 72)


def run_single_task(agent: ReActAgent, task: str, keep_history: bool) -> None:
    result = agent.run(task)
    print("\n最终结果：")
    print(result)
    if not keep_history:
        agent.reset()


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"无法解析布尔值: {value}")


def read_multiline_task() -> str:
    print("请输入多行任务，最后输入 :end 提交：")
    lines: list[str] = []
    while True:
        line = input("... ")
        if line.strip() == ":end":
            return "\n".join(lines).strip()
        lines.append(line)


def apply_runtime_updates(
    agent: ReActAgent,
    config: PlaygroundConfig,
    command: str,
) -> tuple[ReActAgent, PlaygroundConfig, str]:
    if not command.startswith(":set "):
        raise ValueError("命令必须以 ':set ' 开头")

    updates = command[len(":set "):].strip().split()
    if not updates:
        raise ValueError("请提供至少一个 key=value 配置")

    new_config = PlaygroundConfig(**config.__dict__)

    for item in updates:
        if "=" not in item:
            raise ValueError(f"配置格式错误: {item}，请使用 key=value")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()

        if key == "mode":
            if raw_value not in ReActAgent.available_modes():
                raise ValueError(f"不支持的 mode: {raw_value}")
            new_config.mode = raw_value
        elif key == "executor_mode":
            if raw_value not in {"function_calling", "text_parsing"}:
                raise ValueError(f"不支持的 executor_mode: {raw_value}")
            new_config.executor_mode = raw_value
        elif key == "max_steps":
            new_config.max_steps = int(raw_value)
        elif key == "max_plan_steps":
            new_config.max_plan_steps = int(raw_value)
        elif key == "max_replans":
            new_config.max_replans = int(raw_value)
        elif key == "enable_verifier":
            new_config.enable_verifier = parse_bool(raw_value)
        elif key == "keep_history":
            new_config.keep_history = parse_bool(raw_value)
        elif key == "quiet":
            new_config.quiet = parse_bool(raw_value)
        else:
            raise ValueError(f"不支持的配置项: {key}")

    new_agent = build_agent(new_config)
    if new_config.keep_history:
        new_agent.history = agent.history

    return new_agent, new_config, f"配置已更新：{config_summary(new_config, new_agent)}"


def interactive_loop(agent: ReActAgent, config: PlaygroundConfig) -> None:
    print_welcome(config, agent)

    while True:
        prompt = f"\n[{config.mode}] 请输入任务 > "
        try:
            task = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出 Playground。")
            return

        if not task:
            continue
        if task in {":quit", "quit", "exit"}:
            print("已退出 Playground。")
            return
        if task == ":help":
            print(HELP_TEXT)
            continue
        if task == ":examples":
            print_examples()
            continue
        if task == ":reset":
            agent.reset()
            print("历史已清空。")
            continue
        if task == ":config":
            print(f"当前配置: {config_summary(config, agent)}")
            continue
        if task == ":skills":
            print_skills(agent)
            continue
        if task == ":multiline":
            task = read_multiline_task()
            if not task:
                print("未输入任何内容，已取消。")
                continue
        elif task.startswith(":set "):
            try:
                agent, config, message = apply_runtime_updates(agent, config, task)
                print(message)
            except ValueError as exc:
                print(f"配置更新失败：{exc}")
            continue

        run_single_task(agent, task, config.keep_history)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = args_to_config(args)
    agent = build_agent(config)

    if args.task:
        run_single_task(agent, args.task, config.keep_history)
        return

    interactive_loop(agent, config)


if __name__ == "__main__":
    main()
