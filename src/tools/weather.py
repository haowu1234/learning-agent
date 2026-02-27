"""天气查询工具（Mock 版本）

返回模拟的天气数据，用于学习和测试 Agent 工具调用流程。
无需真实 API Key。
"""

from __future__ import annotations

import random
from typing import Any

from src.tools.base import Tool

# 模拟天气数据
_MOCK_WEATHER: dict[str, dict[str, Any]] = {
    "北京": {"temp": 12, "weather": "晴", "humidity": 35, "wind": "北风3级"},
    "上海": {"temp": 18, "weather": "多云", "humidity": 65, "wind": "东风2级"},
    "广州": {"temp": 25, "weather": "阴", "humidity": 80, "wind": "南风2级"},
    "深圳": {"temp": 26, "weather": "小雨", "humidity": 85, "wind": "东南风3级"},
    "杭州": {"temp": 16, "weather": "多云", "humidity": 60, "wind": "东风2级"},
    "成都": {"temp": 14, "weather": "阴", "humidity": 70, "wind": "微风"},
    "武汉": {"temp": 15, "weather": "晴", "humidity": 50, "wind": "北风2级"},
    "南京": {"temp": 14, "weather": "多云", "humidity": 55, "wind": "东北风2级"},
    "西安": {"temp": 10, "weather": "晴", "humidity": 40, "wind": "西北风3级"},
    "重庆": {"temp": 16, "weather": "阴", "humidity": 75, "wind": "微风"},
}


class WeatherTool(Tool):
    """天气查询工具（Mock 数据）。"""

    @property
    def name(self) -> str:
        return "weather"

    @property
    def description(self) -> str:
        return (
            "查询指定城市的当前天气信息，包括温度、天气状况、湿度和风力。"
            "支持中国主要城市的天气查询。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "要查询天气的城市名称，例如 '北京'、'上海'",
                }
            },
            "required": ["city"],
        }

    def run(self, city: str, **_: Any) -> str:
        data = _MOCK_WEATHER.get(city)

        if data is None:
            # 对于未预设的城市，随机生成数据
            weathers = ["晴", "多云", "阴", "小雨", "大雨", "雪"]
            winds = ["微风", "北风2级", "东风3级", "南风2级", "西北风3级"]
            data = {
                "temp": random.randint(-5, 38),
                "weather": random.choice(weathers),
                "humidity": random.randint(20, 95),
                "wind": random.choice(winds),
            }

        return (
            f"{city}天气：{data['weather']}，"
            f"温度 {data['temp']}°C，"
            f"湿度 {data['humidity']}%，"
            f"风力 {data['wind']}"
        )
