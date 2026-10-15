"""AstrBot 事件对象的取用工具。

集中处理两类版本差异，避免各处重复写 try/except：
- 事件上某个取值方法在当前 AstrBot 版本可能不存在
- 上游指令识别所依赖的内部 filter 类型路径可能随版本变化
"""

from __future__ import annotations

import re

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
# 她本轮用 say 去了别的会话（值为目标 umo）。据此抑制原会话的收尾发言。
EXTRA_LEFT_TO = "shittim_left_to"


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


# 无文本消息的占位说明。用组件类名匹配，取不到就整体降级为空串。
_COMPONENT_LABELS = {
    "Image": "[图片]",
    "Face": "[表情]",
    "Poke": "[戳一戳]",
    "Video": "[视频]",
    "File": "[文件]",
    "Record": "[语音]",
    "Forward": "[转发消息]",
    "Json": "[卡片消息]",
    "Music": "[音乐]",
}


def describe_non_text(event) -> str:
    """为图片/表情这类无文本消息生成占位描述。

    这类消息不该触发回复，但完全不入账会让她的对话流出现空白——
    她会看到「刚才有人发了什么，但我不知道是什么」。占位描述既保留了
    「发生过什么」这个事实，又不会让决断层误以为有内容可回应。
    """
    try:
        components = event.get_messages() or []
    except Exception:
        return ""
    labels: list[str] = []
    for component in components:
        label = _COMPONENT_LABELS.get(type(component).__name__)
        if label and label not in labels:
            labels.append(label)
    return "".join(labels)


# AstrBot 把非文本消息转成占位标签（`_outline_chain`：图片→[图片]、表情→[表情:id]），
# 所以这些文本看着非空，其实什么也没说。
_PLACEHOLDER_RE = re.compile(
    r"\[(?:图片|表情[^\]]*|At[^\]]*|引用消息[^\]]*|转发消息|戳一戳|视频|文件|"
    r"语音|卡片消息|音乐|Image[^\]]*)\]"
)
# 表情符号与零宽连接符：只发表情同样没有可回应的内容
_EMOJI_RE = re.compile(r"[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200D\s]+")


def is_placeholder_only(text: str) -> bool:
    """整条消息是否只有占位标签、表情和空白，没有一句真正的话。

    存在的意义：`get_message_outline()` 对图片返回「[图片]」，非空，
    所以「没有可回应的文本内容」这类判断不能只看是否为空串。
    """
    cleaned = _PLACEHOLDER_RE.sub("", text or "")
    cleaned = _EMOJI_RE.sub("", cleaned)
    return not cleaned.strip()


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
