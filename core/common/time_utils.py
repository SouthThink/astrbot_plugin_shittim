"""时间格式化工具。

她之前完全没有时间参照——所有注入块都不带时间，原生会话历史（OpenAI messages
格式）本身也没有时间字段。结果就是「几十分钟前」和「刚刚」在她眼里一样。

这里提供相对时间标签所需的最小工具。刻意用相对时间而不是绝对时间：判断
「刚刚还是很久以前」不需要任何锚点，模型拿到即用。
"""

from __future__ import annotations

import time


def format_age(seconds: float) -> str:
    """把「距今多少秒」说成人话。"""
    seconds = max(float(seconds), 0.0)
    if seconds < 60:
        return "刚刚"
    if seconds < 3600:
        return f"{int(seconds // 60)}分钟前"
    if seconds < 86400:
        return f"{int(seconds // 3600)}小时前"
    return f"{int(seconds // 86400)}天前"


def format_clock(ts: float) -> str:
    """把时间戳说成绝对时间，如「2026-10-16 07:00」。

    只在「她自己需要一个时间锚点」时用（比如刚被叫醒、刚起床），
    日常消息行仍用 format_age 的相对时间。
    """
    if not ts:
        return ""
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(float(ts)))
