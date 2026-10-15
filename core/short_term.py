"""短期记忆：一段时间内她进入过的所有群聊/私聊的原始上下文。

与「按会话隔离」的做法不同，这里是**跨会话合并**的一本总账：所有会话的消息
按时间排在同一条流里，每条都带明确来源标记。这样她读到的是一段连续的
「生活片段」，而不是割裂的多个会话，符合统一人格的理念；同时来源标记保证
她不会把 A 群的话回到 B 群去。
"""

from __future__ import annotations

import sqlite3
import time

from .db import MemoryDB
from .tokenizer import estimate_tokens

_ROLE_LABEL_ASSISTANT = "你"

# 从 AstrBot 官方历史回填的消息用这个占位发送者：历史只存 role 与 content，
# 不含发送者，所以回填进群聊的 user 消息无法还原是谁说的。
HISTORY_SENDER_ID = "astrbot_history"


def build_chat_label(chat_kind: str, name: str, chat_id: str) -> str:
    """构造人类可读的会话来源标签，如「群聊·测试群·10001」。"""
    kind_label = "私聊" if chat_kind == "private" else "群聊"
    parts = [kind_label]
    if name and name.strip():
        parts.append(name.strip())
    # chat_id 形如 aiocqhttp:GroupMessage:10001，只取最后一段做辨识
    tail = (chat_id or "").rsplit(":", 1)[-1].strip()
    if tail and tail not in parts:
        parts.append(tail)
    return "·".join(parts)


def format_message_line(row: sqlite3.Row) -> str:
    """把一条短期消息格式化成带来源标记的一行，供上下文与提炼共用。"""
    role = str(row["role"] or "user")
    sender_id = str(row["sender_id"] or "")
    if role == "assistant":
        speaker = _ROLE_LABEL_ASSISTANT
    elif sender_id == HISTORY_SENDER_ID:
        # 从 AstrBot 历史回填的消息不知道发言人是谁，如实标注
        speaker = "旧记录"
    else:
        name = str(row["sender_name"] or "").strip() or sender_id
        speaker = name or "某人"
    content = str(row["content"] or "").replace("\n", " ").strip()
    return f"[{row['chat_label']}] {speaker}: {content}"


class ShortTermLedger:
    """短期账本的读写入口。"""

    def __init__(self, db: MemoryDB, config):
        self._db = db
        self._config = config

    # ── 写入 ──────────────────────────────────────────────

    def add(
        self,
        *,
        chat_id: str,
        chat_label: str,
        chat_kind: str,
        sender_id: str,
        sender_name: str,
        role: str,
        content: str,
        created_at: float | None = None,
        gated: bool = False,
    ) -> int:
        """记入一条短期消息，返回自增 id。内容为空的记录直接丢弃。

        gated=True 表示这条消息不在 AstrBot 的历史里（被决断层拦下，
        或由插件主动发出），跨会话上下文档需要把它补进去。
        """
        text = (content or "").strip()
        if not text:
            return 0
        cursor = self._db.execute(
            """
            INSERT INTO short_term_messages
                (chat_id, chat_label, chat_kind, sender_id, sender_name,
                 role, content, created_at, extracted, gated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
            """,
            (
                chat_id or "",
                chat_label or "",
                chat_kind or "group",
                str(sender_id or ""),
                sender_name or "",
                role or "user",
                text,
                float(created_at if created_at is not None else time.time()),
                1 if gated else 0,
            ),
        )
        return int(cursor.lastrowid or 0)

    def mark_extracted(self, ids: list[int]) -> None:
        """标记这批消息已提炼进长期层。"""
        self._mark_flag("extracted", ids)

    def mark_gated(self, ids: list[int]) -> None:
        """标记这批消息不在 AstrBot 的历史里（被拦下或插件主动发出）。"""
        self._mark_flag("gated", ids)

    def add_batch(self, rows: list[dict], extracted: bool = False) -> int:
        """批量写入。

        extracted=True 表示这批消息不参与长期记忆提炼。回填 AstrBot 历史时
        用它：回填可能一次几百条，若计入提炼会突然产生大量模型调用。
        """
        payload = []
        for row in rows:
            text = str(row.get("content") or "").strip()
            if not text:
                continue
            payload.append(
                (
                    row.get("chat_id") or "",
                    row.get("chat_label") or "",
                    row.get("chat_kind") or "group",
                    str(row.get("sender_id") or ""),
                    row.get("sender_name") or "",
                    row.get("role") or "user",
                    text,
                    float(row.get("created_at") or time.time()),
                    1 if extracted else 0,
                    1 if row.get("gated") else 0,
                )
            )
        self._db.executemany(
            """
            INSERT INTO short_term_messages
                (chat_id, chat_label, chat_kind, sender_id, sender_name,
                 role, content, created_at, extracted, gated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
        return len(payload)

    def _mark_flag(self, column: str, ids: list[int]) -> None:
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        self._db.execute(
            f"UPDATE short_term_messages SET {column} = 1 WHERE id IN ({placeholders})",
            tuple(ids),
        )

    # ── 读取 ──────────────────────────────────────────────

    def pending(self, limit: int) -> list[sqlite3.Row]:
        """取最早的、尚未提炼的消息（按时间正序，保证提炼顺序正确）。"""
        return self._db.query(
            """
            SELECT * FROM short_term_messages
            WHERE extracted = 0
            ORDER BY id ASC
            LIMIT ?
            """,
            (int(limit),),
        )

    def pending_count(self) -> int:
        row = self._db.query_one(
            "SELECT COUNT(*) AS n FROM short_term_messages WHERE extracted = 0"
        )
        return int(row["n"]) if row else 0

    def recent(self, limit: int) -> list[sqlite3.Row]:
        """取最近 N 条消息（返回按时间正序）。

        必须按时间而不是 id 排序：从官方历史回填的消息 id 更大但时间更早，
        按 id 排会把它们错误地当成最新消息。
        """
        rows = self._db.query(
            """
            SELECT * FROM short_term_messages
             ORDER BY created_at DESC, id DESC
             LIMIT ?
            """,
            (int(limit),),
        )
        return list(reversed(rows))

    def count(self) -> int:
        row = self._db.query_one("SELECT COUNT(*) AS n FROM short_term_messages")
        return int(row["n"]) if row else 0

    def count_of_chat(self, chat_id: str) -> int:
        row = self._db.query_one(
            "SELECT COUNT(*) AS n FROM short_term_messages WHERE chat_id = ?", (chat_id,)
        )
        return int(row["n"]) if row else 0

    def all_of_chat(self, chat_id: str) -> list[sqlite3.Row]:
        """取某会话账本的全部 (role, content, created_at)，按时间正序。

        供回填做锚点对齐：拿 AstrBot 历史与这份序列比对，才能给新记录定出
        真实的时间位置，而不是一律塞到时间线最前面。
        """
        return self._db.query(
            """
            SELECT role, content, created_at
              FROM short_term_messages
             WHERE chat_id = ?
             ORDER BY created_at ASC, id ASC
            """,
            (chat_id,),
        )

    def build_recent_block(
        self,
        limit: int | None = None,
        token_budget: int | None = None,
        exclude_chat_id: str | None = None,
    ) -> str:
        """把最近的原始消息拼成带来源标记的上下文块，受 token 预算约束。

        exclude_chat_id 用于排除当前会话：当前会话里**能到达主脑**的历史已由
        AstrBot 原样放进 req.contexts，重复注入会让模型在两种格式里看到同一条
        消息。但被决断层拦下的消息（gated=1）主脑从未看到，必须补进来，
        否则会出现「刚才说的那件事」无指代可依的上下文空洞。
        """
        limit = self._config.short_term_context_messages if limit is None else limit
        token_budget = (
            self._config.short_term_context_tokens if token_budget is None else token_budget
        )
        if limit <= 0 or token_budget <= 0:
            return ""

        rows = self.recent(limit * 3 if exclude_chat_id else limit)
        if exclude_chat_id:
            rows = [
                row
                for row in rows
                if str(row["chat_id"]) != exclude_chat_id or int(row["gated"] or 0) == 1
            ][-limit:]
        else:
            rows = rows[-limit:]

        selected: list[str] = []
        used = 0
        # 从最新往回取，直到超出预算；再翻转回时间正序
        for row in reversed(rows):
            line = format_message_line(row)
            cost = estimate_tokens(line)
            if selected and used + cost > token_budget:
                break
            selected.append(line)
            used += cost
        selected.reverse()
        return "\n".join(selected)

    def recent_of_chat(self, chat_id: str, limit: int) -> list[sqlite3.Row]:
        """取某个会话最近的对话（按时间正序）。"""
        rows = self._db.query(
            """
            SELECT * FROM short_term_messages
             WHERE chat_id = ?
             ORDER BY created_at DESC, id DESC
             LIMIT ?
            """,
            (chat_id, int(limit)),
        )
        return list(reversed(rows))

    def list_known_chats(self, limit: int = 20) -> list[sqlite3.Row]:
        """列出活跃过的会话，供她决定往哪里发消息。"""
        return self._db.query(
            """
            SELECT chat_id,
                   chat_label,
                   chat_kind,
                   COUNT(*)      AS message_count,
                   MAX(created_at) AS last_at
              FROM short_term_messages
             GROUP BY chat_id
             ORDER BY last_at DESC
             LIMIT ?
            """,
            (int(limit),),
        )

    def resolve_chat(self, token: str) -> str | None:
        """把「群号 / QQ号 / 会话标签 / 完整 umo」解析成完整的 unified_msg_origin。"""
        token = (token or "").strip()
        if not token:
            return None
        # 1) 完整 umo
        row = self._db.query_one(
            "SELECT chat_id FROM short_term_messages WHERE chat_id = ? LIMIT 1", (token,)
        )
        if row:
            return str(row["chat_id"])
        # 2) umo 末段（群号或 QQ 号）
        row = self._db.query_one(
            """
            SELECT chat_id FROM short_term_messages
             WHERE chat_id LIKE ?
             ORDER BY id DESC LIMIT 1
            """,
            (f"%:{token}",),
        )
        if row:
            return str(row["chat_id"])
        # 3) 会话标签模糊匹配
        row = self._db.query_one(
            """
            SELECT chat_id FROM short_term_messages
             WHERE chat_label LIKE ?
             ORDER BY id DESC LIMIT 1
            """,
            (f"%{token}%",),
        )
        return str(row["chat_id"]) if row else None

    def label_for_chat(self, chat_id: str) -> str:
        """取该会话最近用过的来源标签；没有记录则按会话 ID 现造一个。"""
        row = self._db.query_one(
            """
            SELECT chat_label FROM short_term_messages
             WHERE chat_id = ?
             ORDER BY id DESC LIMIT 1
            """,
            (chat_id,),
        )
        if row and str(row["chat_label"] or "").strip():
            return str(row["chat_label"])
        kind = "private" if "FriendMessage" in (chat_id or "") else "group"
        return build_chat_label(kind, "", chat_id)

    def update_config(self, config) -> None:
        self._config = config

    # ── 淘汰 ──────────────────────────────────────────────

    def prune(self, max_per_chat: int | None = None) -> int:
        """淘汰每个会话中最早的消息，只保留最近 N 条。

        刻意按会话分组而不是全局按时间排序：全局淘汰会让一个话痨群把其它
        会话的历史全部挤掉，跨会话记忆就废了。
        """
        max_per_chat = (
            self._config.short_term_max_messages if max_per_chat is None else max_per_chat
        )
        if max_per_chat <= 0:
            return 0
        cursor = self._db.execute(
            """
            DELETE FROM short_term_messages
             WHERE id IN (
                 SELECT id FROM (
                     SELECT id,
                            ROW_NUMBER() OVER (
                                PARTITION BY chat_id
                                ORDER BY created_at DESC, id DESC
                            ) AS rn
                       FROM short_term_messages
                 )
                  WHERE rn > ?
             )
            """,
            (int(max_per_chat),),
        )
        return int(cursor.rowcount or 0)
