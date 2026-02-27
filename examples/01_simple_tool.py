"""第 1 课：理解工具调用

本课目标：
- 了解 OpenAI Function Calling 的工作机制
- 手动调用一个工具并将结果返回给 LLM
- 观察 LLM 如何决定调用工具以及如何使用工具结果

运行方式：
    python -m examples.01_simple_tool
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm import LLMClient
from src.tools.calculator import CalculatorTool


def main():
    # ---- 步骤 1：初始化 LLM 和工具 ----
    print("=" * 60)
    print("第 1 课：理解工具调用")
    print("=" * 60)

    llm = LLMClient()
    calculator = CalculatorTool()

    # 将工具转换为 OpenAI 格式
    tools = [calculator.to_openai_tool()]
    print(f"\n工具定义（OpenAI 格式）：")
    print(json.dumps(tools, indent=2, ensure_ascii=False))

    # ---- 步骤 2：发送包含工具定义的请求 ----
    query = "请帮我计算 (15 + 27) * 3 - 18 / 6 等于多少？"
    print(f"\n用户问题：{query}")
    print("-" * 40)

    messages = [
        {"role": "system", "content": "你是一个智能助手，可以使用工具来帮助计算。"},
        {"role": "user", "content": query},
    ]

    # 第一次调用：LLM 决定是否使用工具
    print("\n[第 1 次 LLM 调用] 发送问题 + 工具定义...")
    response = llm.chat(messages, tools=tools)
    message = response.choices[0].message

    print(f"  LLM 回复内容：{message.content or '(无文本，准备调用工具)'}")
    print(f"  是否调用工具：{'是' if message.tool_calls else '否'}")

    if not message.tool_calls:
        print(f"\nLLM 直接回答了，无需调用工具。")
        return

    # ---- 步骤 3：执行工具调用 ----
    tool_call = message.tool_calls[0]
    func_name = tool_call.function.name
    func_args = json.loads(tool_call.function.arguments)

    print(f"\n  工具名称：{func_name}")
    print(f"  工具参数：{func_args}")

    # 手动执行工具
    result = calculator.run(**func_args)
    print(f"  工具结果：{result}")

    # ---- 步骤 4：将工具结果返回给 LLM ----
    # 必须按 OpenAI 格式拼装 assistant + tool 消息
    messages.append({
        "role": "assistant",
        "content": message.content,
        "tool_calls": [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": tool_call.function.arguments,
                },
            }
        ],
    })
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": result,
    })

    print(f"\n[第 2 次 LLM 调用] 发送工具结果，获取最终回答...")
    response2 = llm.chat(messages, tools=tools)
    final_answer = response2.choices[0].message.content

    print(f"\n最终回答：{final_answer}")
    print("=" * 60)

    # ---- 知识点总结 ----
    print("\n📝 知识点总结：")
    print("1. 工具定义通过 tools 参数传给 LLM")
    print("2. LLM 返回 tool_calls 表示它想调用工具")
    print("3. 我们执行工具后，将结果以 tool 角色消息返回")
    print("4. LLM 根据工具结果生成最终回答")
    print("5. 整个过程需要 2 次 LLM 调用")


if __name__ == "__main__":
    main()
