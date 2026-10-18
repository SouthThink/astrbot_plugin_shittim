"""时间格式化工具。

此前的注入块均不带时间信息，原生会话历史（OpenAI messages 格式）本身也没有
时间字段，导致「几十分钟前」与「刚刚」在上下文中无法区分。

本模块提供时间标签所需的最小工具。上下文中的消息行使用**绝对时间戳**
（`format_clock`）：相对时间会随当前时刻变化，使请求前缀逐轮改变并破坏缓存命中。
`format_age` 仅用于工具返回等一次性文本。
"""

from __future__ import annotations

import time


def format_age(seconds: float) -> str:
    """将「距今秒数」格式化为可读的相对时间。"""
    seconds = max(float(seconds), 0.0)
    if seconds < 60:
        return "刚刚"
    if seconds < 3600:
        return f"{int(seconds // 60)}分钟前"
    if seconds < 86400:
        return f"{int(seconds // 3600)}小时前"
    return f"{int(seconds // 86400)}天前"


def format_clock(ts: float) -> str:
    """将时间戳格式化为绝对时间，如「2026-10-16 07:00」。

    上下文消息行与时间锚点均使用绝对时间；只有工具返回等一次性文本
    使用 format_age 的相对时间。
    """
    if not ts:
        return ""
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(float(ts)))
