"""合并转发：把转发消息里的聊天记录读成文本，写回账本。

## 独立实现的原因

QQ 的合并转发在协议层只是一个 `Forward` 消息段，携带的仅有 `id`。AstrBot 把它
转成同名组件后，`get_message_outline()` 只会返回「[转发消息]」占位，内层聊天记录
从未进入账本。因此判定、跨会话语境与记忆页面都只能看到这四个字。

## 取回时机

`id` 在服务端有有效期，但不需要像图片那样当场读成本地文件——它只是一个索引。
记账时把 `id` 取出，随后调用适配器的 `get_forward_msg` 拉回节点列表，渲染成
「昵称: 内容」写回账本。调用方（`main._read_forward`）对这次取回做**有界等待**：
纯转发消息的 `message_str` 为空，AstrBot 会直接跳过主脑请求，必须在事件链上
拿到内容补进 prompt；超时则让任务继续在后台完成，只影响本轮 prompt。

## 平台差异

优先用事件自带的机器人客户端（aiocqhttp 事件上的 `bot`）；取不到时按
`unified_msg_origin` 找到平台实例再取客户端。两者都不可用时静默跳过，
账本保留「[转发消息]」占位，不影响其余链路。
"""

from __future__ import annotations

import json

from astrbot.api import logger

from ..common.event_utils import platform_of

# 单条转发消息最多读取的聊天记录条数
_DEFAULT_MAX_ITEMS = 20
# 渲染后写入账本的最大字符数
_DEFAULT_MAX_CHARS = 1200

# 非文本消息段在转发记录里的占位说明
_SEGMENT_LABELS = {
    "image": "[图片]",
    "face": "[表情]",
    "mface": "[表情]",
    "record": "[语音]",
    "video": "[视频]",
    "file": "[文件]",
    "forward": "[转发消息]",
    "node": "[转发消息]",
    "nodes": "[转发消息]",
    "json": "[卡片消息]",
    "xml": "[卡片消息]",
    "music": "[音乐]",
}


def collect_ids(event, limit: int = 1) -> list[str]:
    """从事件的消息链中提取合并转发 id，最多 limit 条。

    按类名而非 isinstance 判断，避免绑定特定 AstrBot 版本的导入路径。
    """
    try:
        components = event.get_messages() or []
    except Exception:
        return []
    ids: list[str] = []
    for component in components:
        if type(component).__name__ != "Forward":
            continue
        value = str(getattr(component, "id", "") or "").strip()
        if value and value not in ids:
            ids.append(value)
        if len(ids) >= max(int(limit), 1):
            break
    return ids


def _segments_text(segments) -> str:
    """把一段消息段渲染成文本；无法识别的段直接跳过。"""
    if isinstance(segments, str):
        return segments.strip()
    if not isinstance(segments, list):
        return ""
    parts: list[str] = []
    for segment in segments:
        if isinstance(segment, str):
            parts.append(segment)
            continue
        if not isinstance(segment, dict):
            continue
        kind = str(segment.get("type") or "").lower()
        data = segment.get("data")
        data = data if isinstance(data, dict) else {}
        if kind in ("text", "plain"):
            parts.append(str(data.get("text") or ""))
        elif kind == "at":
            name = str(data.get("name") or data.get("qq") or "").strip()
            parts.append(f"@{name}" if name else "@")
        elif kind in _SEGMENT_LABELS:
            parts.append(_SEGMENT_LABELS[kind])
    return "".join(parts).strip()


def _node_payload(node) -> dict:
    """取出节点自身的字段：兼容 `{sender, content}` 与 `{type, data:{...}}` 两种形态。"""
    if not isinstance(node, dict):
        return {}
    data = node.get("data")
    if isinstance(data, dict) and any(
        key in data for key in ("sender", "content", "message")
    ):
        return data
    return node


def _sender_name(node: dict) -> str:
    sender = node.get("sender")
    sender = sender if isinstance(sender, dict) else {}
    for key in ("card", "nickname", "name"):
        value = str(sender.get(key) or "").strip()
        if value:
            return value
    return str(sender.get("user_id") or "").strip()


def _node_list(payload) -> list:
    """从接口返回值里取出节点数组，兼容多种包装形态。

    NapCat 的 `data.message` 可能是数组，也可能是 JSON 字符串；部分实现会把
    整份结果再包一层 `data`。逐层展开，直到找到列表为止。
    """
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in ("messages", "message", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except (TypeError, ValueError):
                continue
            if isinstance(parsed, (list, dict)):
                nested = _node_list(parsed)
                if nested:
                    return nested
        elif isinstance(value, dict):
            nested = _node_list(value)
            if nested:
                return nested
    return []


class ForwardReader:
    """把一条合并转发读成一段可读文本。"""

    def __init__(self, config, astr_context) -> None:
        self._config = config
        self._astr_context = astr_context

    def update_config(self, config) -> None:
        self._config = config

    async def read(self, event, forward_id: str) -> str:
        """读回转发内容；失败返回空串（账本保留占位）。"""
        forward_id = str(forward_id or "").strip()
        if not forward_id:
            return ""
        client = self._client(event)
        if client is None:
            logger.debug("[什亭之枢] 没有可用的机器人客户端，跳过转发读取")
            return ""
        payload = await self._fetch(client, forward_id)
        if payload is None:
            return ""
        return self._render(_node_list(payload))

    # ── 客户端 ────────────────────────────────────────────

    def _client(self, event):
        """优先用事件自带的客户端，其次按平台实例取。"""
        bot = getattr(event, "bot", None)
        if bot is not None and callable(getattr(bot, "call_action", None)):
            return bot
        platform_id = platform_of(event)
        if not platform_id or self._astr_context is None:
            return None
        try:
            platform = self._astr_context.get_platform_inst(platform_id)
        except Exception:
            return None
        getter = getattr(platform, "get_client", None)
        if not callable(getter):
            return None
        try:
            return getter()
        except Exception:
            return None

    @staticmethod
    async def _fetch(client, forward_id: str):
        """调用合并转发查询接口。

        不同适配器对参数名的约定不一致（NapCat 用 `message_id`，部分实现用
        `id`），依次尝试；都失败时返回 None。
        """
        for params in ({"message_id": forward_id}, {"id": forward_id}):
            try:
                data = await client.call_action("get_forward_msg", **params)
            except Exception as error:
                logger.debug("[什亭之枢] 读取转发消息失败: %s", error)
                continue
            if data:
                return data
        return None

    # ── 渲染 ──────────────────────────────────────────────

    def _limits(self) -> tuple[int, int]:
        try:
            max_items = int(
                getattr(self._config, "forward_max_items", _DEFAULT_MAX_ITEMS)
            )
        except (TypeError, ValueError):
            max_items = _DEFAULT_MAX_ITEMS
        try:
            max_chars = int(
                getattr(self._config, "forward_max_chars", _DEFAULT_MAX_CHARS)
            )
        except (TypeError, ValueError):
            max_chars = _DEFAULT_MAX_CHARS
        return max(max_items, 1), max(max_chars, 1)

    def _render(self, nodes: list) -> str:
        """节点 → 「昵称: 内容」，用「；」串成一行，并受条数与字数上限约束。"""
        max_items, max_chars = self._limits()
        lines: list[str] = []
        used = 0
        for node in nodes[:max_items]:
            payload = _node_payload(node)
            text = _segments_text(payload.get("content"))
            if not text and "message" in payload:
                text = _segments_text(payload.get("message"))
            if not text:
                continue
            sender = _sender_name(payload)
            line = f"{sender}: {text}" if sender else text
            remaining = max_chars - used
            if remaining <= 0:
                break
            if len(line) > remaining:
                lines.append(line[:remaining].rstrip() + "…")
                used = max_chars
                break
            lines.append(line)
            used += len(line) + 1  # 分隔符也占字数
        return "；".join(lines).strip()[:max_chars]
