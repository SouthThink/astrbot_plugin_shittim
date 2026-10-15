"""AstrBot 事件对象的取用工具。

集中处理两类版本差异，避免各处重复写 try/except：
- 事件上某个取值方法在当前 AstrBot 版本可能不存在
- 上游指令识别所依赖的内部 filter 类型路径可能随版本变化
"""

from __future__ import annotations

try:
    from astrbot.api import logger
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

try:  # 上游指令识别依赖 AstrBot 内部类型，路径随版本变化，取不到就整体降级
    from astrbot.core.star.filter.command import CommandFilter
    from astrbot.core.star.filter.command_group import CommandGroupFilter

    _COMMAND_FILTERS: tuple = (CommandFilter, CommandGroupFilter)
except Exception:  # pragma: no cover - 兼容不同 AstrBot 版本
    _COMMAND_FILTERS = ()

# 内部类型取不到时的兜底指令前缀（AstrBot 默认指令前缀）
_FALLBACK_COMMAND_PREFIXES = ("/",)

# 事件属性上的临时标记键
EXTRA_MESSAGE_ID = "shittim_message_id"


def call_str(event, method_name: str, default: str = "") -> str:
    """安全调用事件上的取值方法，缺方法或抛异常时回退默认值。"""
    method = getattr(event, method_name, None)
    if not callable(method):
        return default
    try:
        return str(method() or default)
    except Exception:
        return default


def get_message_text(event) -> str:
    """取消息的文本概要，失败返回空串。"""
    try:
        return (event.get_message_outline() or "").strip()
    except Exception:
        return ""


def is_private_chat(chat_id: str) -> bool:
    """unified_msg_origin 形如 platform:MessageType:session_id。"""
    return "FriendMessage" in (chat_id or "")


def platform_of(event) -> str:
    """取事件所属的平台实例标识，用于拼装 unified_msg_origin。

    必须以 unified_msg_origin 的首段为准：那是 AstrBot 真正用于路由的平台实例
    ID。get_platform_name() 返回的是适配器类型名（如 aiocqhttp），未必等于实例
    ID，拿它拼 umo 会路由不到平台，发送时报
    「cannot find platform for session ...」。
    """
    umo = getattr(event, "unified_msg_origin", "") or ""
    prefix = umo.split(":", 1)[0].strip()
    if prefix:
        return prefix
    # umo 异常时退回适配器类型名，聊胜于无
    return call_str(event, "get_platform_name")


def describe_chat(event) -> tuple[str, str]:
    """返回 (chat_kind, chat_label)，标签用于分清消息来自哪里。"""
    from .short_term import build_chat_label

    chat_id = getattr(event, "unified_msg_origin", "") or ""
    if is_private_chat(chat_id):
        return "private", build_chat_label(
            "private", call_str(event, "get_sender_name"), chat_id
        )
    return "group", build_chat_label("group", call_str(event, "get_group_id"), chat_id)


def is_upstream_command_event(event) -> bool:
    """事件是否已命中 AstrBot 的指令 / skill 处理器。

    优先用 AstrBot 内部 filter 类型精确识别；取不到内部类型时退化为
    「消息以默认指令前缀开头」的粗略判断——宁可把普通消息误判成指令
    （放行），也好过把指令误判成普通消息（拦掉，指令直接失效）。
    """
    if _COMMAND_FILTERS:
        try:
            handlers = event.get_extra("activated_handlers", []) or []
        except Exception:
            handlers = []
        for handler in handlers:
            for event_filter in getattr(handler, "event_filters", []) or []:
                if isinstance(event_filter, _COMMAND_FILTERS):
                    return True
        return False

    text = get_message_text(event)
    return any(text.startswith(prefix) for prefix in _FALLBACK_COMMAND_PREFIXES)


def extract_reply_text(event) -> str:
    """从事件结果里取出实际发出去的文本。"""
    try:
        result = event.get_result() if hasattr(event, "get_result") else None
        chain = getattr(result, "chain", None) if result else None
        if not chain:
            return ""
        return "".join(
            text
            for text in (getattr(component, "text", None) for component in chain)
            if isinstance(text, str)
        ).strip()
    except Exception:
        return ""


def set_extra(event, key: str, value) -> None:
    """安全写入事件临时标记。"""
    setter = getattr(event, "set_extra", None)
    if not callable(setter):
        return
    try:
        setter(key, value)
    except Exception:
        pass


def get_extra(event, key: str, default=None):
    """安全读取事件临时标记。"""
    getter = getattr(event, "get_extra", None)
    if not callable(getter):
        return default
    try:
        value = getter(key, default)
    except Exception:
        return default
    return default if value is None else value


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
