"""时间格式化工具。

此前的注入块均不带时间信息，原生会话历史（OpenAI messages 格式）本身也没有
时间字段，导致「几十分钟前」与「刚刚」在上下文中无法区分。

本模块提供相对时间标签所需的最小工具。刻意采用相对时间而非绝对时间：判断
「刚刚还是很久以前」不需要任何锚点，模型可直接使用。
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

    仅在需要时间锚点时使用（例如刚被叫醒、刚起床），
    日常消息行仍使用 format_age 的相对时间。
    """
    if not ts:
        return ""
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(float(ts)))
