"""从 AstrBot 官方会话历史回填短期记忆。

我们的短期账本只记「插件在场时」的消息。插件装上之前、插件被禁用或出错期间
的对话，AstrBot 自己有记录，但对她不可见。这个模块把它们补进来，让她的
短期记忆尽量完整。

## 对应关系

    unified_msg_origin
        └─> conversation_manager.get_curr_conversation_id(umo)   → conversation_id
        └─> conversation_manager.get_conversation(umo, cid)      → Conversation
        └─> json.loads(conversation.history)                     → OpenAI 消息数组

## 已知的信息损失

AstrBot 的历史只存 role 与 content，**不含发送者**。所以回填进群聊的
user 消息无法知道是谁说的，只能标记为「旧记录」。这是数据源的硬限制，
不是实现偷懒。
"""

from __future__ import annotations

import json
from collections import deque

try:
    from astrbot.api import logger
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

# 只回填这两个角色：tool 结果与纯工具调用的 assistant 对记忆没有价值
_ALLOWED_ROLES = ("user", "assistant")


def extract_text(content) -> str:
    """从 OpenAI 格式的 content 里取出可读文本。"""
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type == "text":
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        elif part_type == "image_url":
            parts.append("[图片]")
    return " ".join(parts).strip()


def convert_records(records: list, limit: int) -> list[dict]:
    """把 AstrBot 历史转成短期账本能用的 [{role, content}]，按时间正序。

    从最新往回取够 limit 条，因此拿到的是「最近的」历史。
    """
    if not isinstance(records, list) or limit <= 0:
        return []
    selected: list[dict] = []
    for record in reversed(records):
        if not isinstance(record, dict):
            continue
        role = record.get("role")
        if role not in _ALLOWED_ROLES:
            continue
        text = extract_text(record.get("content"))
        if not text:
            # assistant 只调工具不說話、或内容为空的消息直接跳过
            continue
        selected.append({"role": role, "content": text})
        if len(selected) >= limit:
            break
    selected.reverse()
    return selected


def align_records(
    records: list[dict], known_rows: list, now: float
) -> list[tuple[dict, float, bool]]:
    """把 AstrBot 历史记录对齐到账本的时间轴。

    沿 AstrBot 历史的时间顺序与账本做锚点匹配：匹配上的记录跳过（我们已有），
    同时把它的时间当作锚点；没匹配上的记录就落到当前锚点稍后的位置。

    返回值 [(记录, 时间戳, 是否早于账本里的一切)]。第三项决定这条回填要不要
    标成 gated 由我们兜底注入——比账本里所有记录都早的那批，正是 AstrBot 最
    可能压缩掉的旧历史；而落在实时消息之间的，AstrBot 自己会保留。

    known_rows: 账本侧记录，需含 role / content / created_at 且按时间正序。
    """
    pools: dict[tuple[str, str], deque] = {}
    earliest_known: float | None = None
    for row in known_rows:
        created_at = float(row["created_at"])
        if earliest_known is None or created_at < earliest_known:
            earliest_known = created_at
        pools.setdefault((str(row["role"]), str(row["content"])), deque()).append(
            created_at
        )

    deferred: list[dict] = []  # 还没出现任何锚点前遇到的新记录
    placed: list[tuple[dict, float, bool]] = []
    cursor: float | None = None
    for record in records:
        pool = pools.get((record["role"], record["content"]))
        if pool:
            cursor = pool.popleft()
            continue
        if cursor is None:
            deferred.append(record)
        else:
            cursor += 1e-3
            placed.append((record, cursor, False))

    base = (earliest_known if earliest_known is not None else now) - len(deferred) - 1.0
    for index, record in enumerate(deferred):
        placed.append((record, base + index, True))

    placed.sort(key=lambda item: item[1])
    return placed


class ConversationHistoryReader:
    """读取 AstrBot 官方会话历史。"""

    def __init__(self, astr_context):
        self._astr_context = astr_context

    @property
    def available(self) -> bool:
        return (
            self._astr_context is not None
            and getattr(self._astr_context, "conversation_manager", None) is not None
        )

    async def read(self, chat_id: str, limit: int) -> list[dict]:
        """读某会话最近的历史，取不到就返回空列表（绝不抛异常）。"""
        if not chat_id or limit <= 0 or not self.available:
            return []
        manager = self._astr_context.conversation_manager
        try:
            conversation_id = await manager.get_curr_conversation_id(chat_id)
            if not conversation_id:
                return []
            conversation = await manager.get_conversation(chat_id, conversation_id)
            raw_history = getattr(conversation, "history", None) if conversation else None
            if not raw_history:
                return []
            records = json.loads(raw_history) if isinstance(raw_history, str) else raw_history
        except Exception as error:
            logger.warning("[什亭之枢] 读取 AstrBot 会话历史失败 | %s | %s", chat_id, error)
            return []
        return convert_records(records, limit)
