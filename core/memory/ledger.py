"""短期记忆：一段时间内所有群聊 / 私聊会话的原始上下文。

与「按会话隔离」的做法不同，这里采用**跨会话合并**的单本总账：所有会话的消息
按时间排在同一条流中，每条均带明确的来源标记。因此上下文是一段连续的
「生活片段」而非割裂的多个会话，符合统一人格的设计；来源标记则保证
不会把 A 群的消息回复到 B 群。
"""

from __future__ import annotations

import json
import sqlite3
import time

from ..common.db import MemoryDB
from ..common.event_utils import build_chat_label
from ..common.time_utils import format_clock
from ..common.tokenizer import estimate_tokens

_ROLE_LABEL_ASSISTANT = "你"

# 从 AstrBot 官方历史回填的消息用这个占位发送者：历史只存 role 与 content，
# 不含发送者，所以回填进群聊的 user 消息无法还原是谁说的。
HISTORY_SENDER_ID = "astrbot_history"

# 消息来源：实时记录 / 睡眠期间收到
SOURCE_LIVE = "live"
# 睡眠期间收到的消息：不参与整理（起床后 staged 置 0 才纳入）。
SOURCE_SLEEP = "sleep"

# 「这条仍在短期记忆中有效」——即当前仍应被记住的条目。
# **所有面向「现在记得什么」的查询都必须带上它**：缺少任一条件，
# 已归档或已压入摘要的记录就会重新出现在上下文、判定与页面中，且不会报错。
# 短期记忆只有两条出路：压入摘要（`condensed_at`）或随深夜整理清空
# （`archived_at`）。只有两者之外的行才是当前的经历。
_LIVE_SQL = "archived_at = 0 AND condensed_at = 0"


def _entry_sort_key(entry: dict) -> tuple:
    """时间线条目的排序键：按时间，同一时刻让摘要排在前（它代表更早的那一段）。"""
    return (float(entry.get("at") or 0.0), 0 if entry.get("kind") == "summary" else 1)


def _caption_of(row) -> str:
    """取这一行的图片转述；老库的行没有该列，取不到就当没有。"""
    try:
        return str(row["image_caption"] or "").strip()
    except (KeyError, IndexError, TypeError):
        return ""


def _source_of(row) -> str:
    """取这一行的来源；老库的行没有该列，取不到按实时消息算。"""
    try:
        return str(row["source"] or "")
    except (KeyError, IndexError, TypeError):
        return ""


def format_speech(row) -> str:
    """该行**说出的内容本身**，不含「谁、在哪儿、多久前」这些前缀。

    时间线需要将助手发言还原为干净的 assistant 回合——上一条消息中已标明
    会话与发言人，若在正文中重复「你」会形成自我陈述。
    """
    content = str(row["content"] or "").replace("\n", " ").strip()
    caption = _caption_of(row)
    if caption:
        content = (
            f"{content}（图片内容：{caption}）" if content else f"[图片]（图片内容：{caption}）"
        )
    if _source_of(row) == SOURCE_SLEEP:
        # 睡眠期消息需明确标注来源，使主模型读取时能区分其产生时间，
        # 避免将睡眠期消息误判为当前对话。
        content = f"{content}（睡觉时收到）"
    return content


def format_message_line(row: sqlite3.Row) -> str:
    """把一条短期消息格式化成带来源与时间标记的一行。

    时间标签采用**绝对时间戳**而非相对时间：相对时间每轮都在变化，
    同一行在两次请求中的文本不一致，会降低 prompt 缓存命中率。

    图片描述同理：AstrBot 对图片只提供「[图片]」占位，若不补充描述，
    该图片在上下文中即不可见。
    """
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
    content = format_speech(row)

    time_tag = ""
    try:
        created_at = float(row["created_at"] or 0.0)
    except (KeyError, IndexError, TypeError, ValueError):
        created_at = 0.0
    if created_at > 0:
        time_tag = f" ({format_clock(created_at)})"
    return f"[{row['chat_label']}] {speaker}{time_tag}: {content}"


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
        source: str = SOURCE_LIVE,
        staged: bool = False,
    ) -> int:
        """记入一条短期消息，返回自增 id。内容为空的记录直接丢弃。

        gated=True 表示这条消息不在 AstrBot 的历史里（被决断层拦下，
        或由插件主动发出）——现在只用于页面与排障，上下文由全局时间线整体
        重建，不再读它。
        source 区分「实时记录」与「睡眠期间收到」。
        staged=True 表示这条消息还在睡眠期暂存区（不参与整理），起床时统一放行。
        """
        text = (content or "").strip()
        if not text:
            return 0
        cursor = self._db.execute(
            """
            INSERT INTO short_term_messages
                (chat_id, chat_label, chat_kind, sender_id, sender_name,
                 role, content, created_at, gated, source, staged)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                source or SOURCE_LIVE,
                1 if staged else 0,
            ),
        )
        return int(cursor.lastrowid or 0)

    def release_staged(self) -> int:
        """起床收口：把睡眠期暂存的消息放行，纳入正常流程。

        只翻 `staged` 标记，**不动 `source`**——"这条是睡觉时收到的"是永久事实，
        上下文里还要靠它显示「（睡觉时收到）」。
        """
        cursor = self._db.execute(
            "UPDATE short_term_messages SET staged = 0 WHERE staged = 1"
        )
        return int(cursor.rowcount or 0)

    def staged_chat_ids(self) -> list[str]:
        """睡眠期间产生过消息的会话列表。

        起床时据此刷新焦点，使这些会话恢复主动发言资格。**必须在
        `release_staged()` 之前调用**，否则查询不到结果。
        """
        rows = self._db.query(
            """
            SELECT DISTINCT chat_id FROM short_term_messages
             WHERE staged = 1 AND chat_id <> ''
            """
        )
        return [str(row["chat_id"]) for row in rows]

    def sleep_digest(
        self, since: float, max_chats: int = 10
    ) -> tuple[int, list[tuple[str, int, str]]]:
        """本轮睡眠期间的消息摘要：`(总条数, [(会话标签, 条数, 最后一条)])`。

        起床提示据此回答「睡眠期间有哪些会话找过我」。采用聚合而非返回数百条
        原文：一夜积压可能很多，而提示只需要会话列表、各会话条数与最后一句。

        会话按最近发言时间倒序，最多 `max_chats` 个；`since` 传入本轮入睡时刻
        （见 `schedule.last_sleep_start()`），否则会混入前几晚的消息。
        """
        if since <= 0:
            return 0, []
        total_row = self._db.query_one(
            f"""
            SELECT COUNT(*) AS n FROM short_term_messages
             WHERE source = ? AND created_at >= ? AND {_LIVE_SQL}
            """,
            (SOURCE_SLEEP, float(since)),
        )
        total = int(total_row["n"] or 0) if total_row is not None else 0
        if not total:
            return 0, []

        rows = self._db.query(
            f"""
            SELECT chat_id, chat_label, COUNT(*) AS cnt, MAX(id) AS last_id
              FROM short_term_messages
             WHERE source = ? AND chat_id <> '' AND created_at >= ?
               AND {_LIVE_SQL}
             GROUP BY chat_id
             ORDER BY MAX(created_at) DESC, last_id DESC
             LIMIT ?
            """,
            (SOURCE_SLEEP, float(since), max(int(max_chats), 1)),
        )
        digest: list[tuple[str, int, str]] = []
        for row in rows:
            last = self._db.query_one(
                "SELECT * FROM short_term_messages WHERE id = ?",
                (int(row["last_id"]),),
            )
            line = format_message_line(last) if last is not None else ""
            digest.append(
                (
                    str(row["chat_label"] or "") or str(row["chat_id"] or ""),
                    int(row["cnt"] or 0),
                    line,
                )
            )
        return total, digest

    def set_image_caption(self, message_id: int, caption: str) -> None:
        """为已入库的消息补充图片描述。

        图片转述在后台异步执行：记账必须保持低延迟，不能等待视觉模型。
        描述返回后再回填该行。
        """
        if not message_id or not caption:
            return
        self._db.execute(
            "UPDATE short_term_messages SET image_caption = ? WHERE id = ?",
            (caption, int(message_id)),
        )

    def mark_gated(self, ids: list[int]) -> None:
        """标记这批消息不在 AstrBot 的历史里（被拦下或插件主动发出）。"""
        self._mark_flag("gated", ids)

    def _mark_flag(self, column: str, ids: list[int]) -> None:
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        self._db.execute(
            f"UPDATE short_term_messages SET {column} = 1 WHERE id IN ({placeholders})",
            tuple(ids),
        )

    # ── 读取 ──────────────────────────────────────────────

    def recent(self, limit: int) -> list[sqlite3.Row]:
        """取最近 N 条消息（返回按时间正序）。

        必须按时间而不是 id 排序：从官方历史回填的消息 id 更大但时间更早，
        按 id 排会把它们错误地当成最新消息。
        """
        rows = self._db.query(
            f"""
            SELECT * FROM short_term_messages
             WHERE {_LIVE_SQL}
             ORDER BY created_at DESC, id DESC
             LIMIT ?
            """,
            (int(limit),),
        )
        return list(reversed(rows))

    def count(self) -> int:
        row = self._db.query_one(
            f"SELECT COUNT(*) AS n FROM short_term_messages WHERE {_LIVE_SQL}"
        )
        return int(row["n"]) if row else 0

    # ── 条目（消息 + 摘要）─────────────────────────────────
    #
    # 短期记忆的「条目」有两类：原始对话消息，以及被压缩段落的替身（摘要）。
    # 两者共同构成时间线，也共同参与下一次压缩的取材——旧摘要只是一个占名额的
    # 普通条目，这正是「二次合并无需特殊处理」的实现方式。

    def timeline_entries_balanced(
        self,
        *,
        per_chat: int,
        max_chats: int,
        limit: int,
        current_chat_id: str = "",
    ) -> list[dict]:
        """按**会话配额**取时间线条目，再整体按时间正序返回。

        与 `live_entries()` 共用同一份 `_LIVE_SQL` 与 `staged = 0` 过滤条件，
        区别只在于取样：

        - 直接取「全局最新 limit 条」时，一个刷屏的群聊会占满窗口，其它会话
          被挤出去——切回那些会话时就显得「断片」。这里先给每个会话留出最近
          `per_chat` 条，再合并排序，保证每个会话的连续性都不被邻居挤掉。
        - 只纳入最近活跃的 `max_chats` 个会话；**当前会话始终入选**，且它被
          拦下（`gated = 1`）的消息全量保留——这些消息没进官方历史，丢掉就
          等于主脑没看见那几句话。
        - 全部活摘要一并保留：它们是压缩后的概要，没有 `chat_id`，
          无法按会话归属。

        `limit` 仍作为最终条数上限，超出时从最旧条目开始丢弃。
        """
        if limit <= 0:
            return []
        per_chat = max(int(per_chat), 1)
        max_chats = max(int(max_chats), 1)

        summaries: list[dict] = []
        grouped: dict[str, list[dict]] = {}
        for entry in self.live_entries():
            if entry.get("kind") == "summary":
                summaries.append(entry)
                continue
            row = entry["row"]
            grouped.setdefault(str(row["chat_id"] or ""), []).append(entry)

        # 会话按最近活跃排序；当前会话必须入选，否则切回来时那几句话就丢了
        order = sorted(
            grouped, key=lambda chat: grouped[chat][-1]["at"], reverse=True
        )
        picked = set(order[:max_chats])
        if current_chat_id in grouped:
            picked.add(current_chat_id)

        kept: list[dict] = []
        for chat in picked:
            rows = grouped[chat]
            kept.extend(rows[-per_chat:])
            if chat == current_chat_id:
                # 被拦下的消息通常就在最近几条里，这里仍单独补一次，
                # 避免一次防抖窗口内连发多条时被配额截掉
                kept.extend(
                    entry
                    for entry in rows
                    if int(entry["row"]["gated"] or 0) == 1
                )

        merged = summaries + kept
        merged.sort(key=_entry_sort_key)
        deduped: list[dict] = []
        seen: set[tuple[str, int]] = set()
        for entry in merged:
            key = (str(entry.get("kind")), int(entry["row"]["id"] or 0))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(entry)
        return deduped[-int(limit) :]

    def oldest_entries(self, limit: int) -> list[dict]:
        """取**最旧**的 limit 个条目，按时间正序（短期压缩的取材范围）。"""
        if limit <= 0:
            return []
        entries = self._entry_rows(order="ASC", limit=int(limit))
        entries.sort(key=_entry_sort_key)
        return entries[: int(limit)]

    def live_entries(self) -> list[dict]:
        """短期层当前**全部**条目（消息 + 摘要），按时间正序。

        与 `timeline_entries` 的区别是**不分页**：深夜收口需要全部压缩，
        任何上限都会导致超出部分在清空时静默丢失。
        """
        entries = self._entry_rows(order="ASC")
        entries.sort(key=_entry_sort_key)
        return entries

    def live_entry_count(self) -> int:
        """短期记忆里活着的条目总数（消息 + 摘要），压缩阈值的判据。"""
        messages = self._db.query_one(
            f"SELECT COUNT(*) AS n FROM short_term_messages "
            f"WHERE {_LIVE_SQL} AND staged = 0"
        )
        summaries = self._db.query_one(
            "SELECT COUNT(*) AS n FROM short_term_summaries WHERE condensed_at = 0"
        )
        return int(messages["n"] or 0 if messages else 0) + int(
            summaries["n"] or 0 if summaries else 0
        )

    def _entry_rows(self, *, order: str, limit: int | None = None) -> list[dict]:
        """消息与摘要各取一份再合并；给了 limit 时整体最新/最旧的 limit 个
        一定落在其中。"""
        tail = " LIMIT ?" if limit is not None else ""
        params: tuple = (int(limit),) if limit is not None else ()
        messages = self._db.query(
            f"""
            SELECT * FROM short_term_messages
             WHERE {_LIVE_SQL} AND staged = 0
             ORDER BY created_at {order}, id {order}{tail}
            """,
            params,
        )
        summaries = self._db.query(
            f"""
            SELECT * FROM short_term_summaries
             WHERE condensed_at = 0
             ORDER BY covered_at {order}, id {order}{tail}
            """,
            params,
        )
        entries = [
            {"kind": "message", "at": float(row["created_at"] or 0.0), "row": row}
            for row in messages
        ]
        entries.extend(
            {"kind": "summary", "at": float(row["covered_at"] or 0.0), "row": row}
            for row in summaries
        )
        entries.sort(key=_entry_sort_key)
        return entries

    def condense_records(self, entries: list[dict]) -> str:
        """把待压缩的条目渲染成给模型的对话记录。"""
        lines: list[str] = []
        for entry in entries:
            if entry.get("kind") == "summary":
                # 旧摘要在压缩时以「上一层的概括」身份出现：模型据此再概括一层
                content = str(entry["row"]["content"] or "").strip()
                lines.append(f"【更早的一段概括】{content}")
                continue
            lines.append(format_message_line(entry["row"]))
        return "\n".join(line for line in lines if line)

    def condense_context(self, entries: list[dict]) -> dict:
        """待压缩区段的归属信息，供写入长期记忆时带上来源。"""
        name_to_id: dict[str, str] = {}
        chat_ids: set[str] = set()
        chat_label = ""
        for entry in entries:
            if entry.get("kind") != "message":
                continue
            row = entry["row"]
            name = str(row["sender_name"] or "").strip()
            if name:
                name_to_id.setdefault(name, str(row["sender_id"] or ""))
            chat_ids.add(str(row["chat_id"] or ""))
            chat_label = str(row["chat_label"] or "") or chat_label
        single = chat_ids.pop() if len(chat_ids) == 1 else ""
        return {
            "chat_id": single or None,
            "chat_label": chat_label if single else None,
            "name_to_id": name_to_id,
        }

    def apply_condensation(
        self, *, summary: str, entries: list[dict], covered_count: int
    ) -> bool:
        """落盘一次压缩：写摘要 + 把被覆盖的条目移出短期记忆。

        多步写入必须原子：中途可见半成品会让那些条目既不在原始列表、
        也不在摘要里——静默丢一段对话。

        被覆盖的**消息**同时标 `condensed_at`：它们的内容已在本次调用里
        一并处理过（压缩即提炼），既不再进上下文，也不会被搜出来。
        """
        text = (summary or "").strip()
        if not text or not entries:
            return False
        now = time.time()
        covered_at = max(float(entry.get("at") or 0.0) for entry in entries)
        inner_ids = [
            int(entry["row"]["id"]) for entry in entries if entry.get("kind") == "summary"
        ]
        message_ids = [
            int(entry["row"]["id"]) for entry in entries if entry.get("kind") == "message"
        ]
        with self._db.lock:
            cursor = self._db.execute(
                """
                INSERT INTO short_term_summaries
                    (content, created_at, covered_at, covered_count, covered_items)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    text,
                    now,
                    covered_at,
                    max(int(covered_count), len(entries)),
                    json.dumps(inner_ids, ensure_ascii=False),
                ),
            )
            if message_ids:
                placeholders = ",".join("?" for _ in message_ids)
                self._db.execute(
                    f"""
                    UPDATE short_term_messages
                       SET condensed_at = ?
                     WHERE id IN ({placeholders})
                    """,
                    (now, *message_ids),
                )
            if inner_ids:
                placeholders = ",".join("?" for _ in inner_ids)
                self._db.execute(
                    f"""
                    UPDATE short_term_summaries SET condensed_at = ?
                     WHERE id IN ({placeholders})
                    """,
                    (now, *inner_ids),
                )
        return bool(cursor.lastrowid)

    def list_summaries(self, limit: int = 200) -> list[sqlite3.Row]:
        """当前活着的摘要，按时间正序（页面展示用）。"""
        rows = self._db.query(
            """
            SELECT * FROM short_term_summaries
             WHERE condensed_at = 0
             ORDER BY covered_at DESC, id DESC
             LIMIT ?
            """,
            (int(limit),),
        )
        return list(reversed(rows))

    def build_recent_block(
        self,
        per_chat: int | None = None,
        max_chats: int | None = None,
        token_budget: int | None = None,
        exclude_chat_id: str | None = None,
    ) -> str:
        """将其他会话的近期消息拼成带来源标记的文本，受 token 预算约束。

        仅用于**决断层**（判断某个群是否接话时，需要了解其他会话的近期情况）。
        主脑上下文使用全局时间线，不经过此处。

        按**会话配额**读取，而非全局取最近 N 条：全局取会导致高频会话挤占
        其他会话的展示配额。每个会话保留代表消息，以维持跨会话连续性。

        `exclude_chat_id` 排除正在判定的会话：该段内容由 `messages_since`
        单独提供，重复注入会使模型误判为两条消息。
        """
        per_chat = (
            self._config.short_term_context_per_chat if per_chat is None else per_chat
        )
        max_chats = (
            self._config.short_term_context_chats if max_chats is None else max_chats
        )
        token_budget = (
            self._config.short_term_context_tokens if token_budget is None else token_budget
        )
        if per_chat <= 0 or max_chats <= 0 or token_budget <= 0:
            return ""

        rows = self._context_rows(per_chat, max_chats, exclude_chat_id)

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

    def _context_rows(
        self, per_chat: int, max_chats: int, exclude_chat_id: str | None
    ) -> list[sqlite3.Row]:
        """取最近活跃的若干会话，各取最近 per_chat 条，按时间正序返回。

        正在判定的会话整个排除：它这一段的内容由 `messages_since` 单独喂给
        判定，这里再放一份只会让模型以为同一句话出现了两次。
        """
        # 多取一个会话，防止被排除的会话占掉名额后实际只有 max_chats-1 个
        chat_rows = self._db.query(
            f"""
            SELECT chat_id
              FROM short_term_messages
             WHERE {_LIVE_SQL}
             GROUP BY chat_id
             ORDER BY MAX(created_at) DESC
             LIMIT ?
            """,
            (int(max_chats) + (1 if exclude_chat_id else 0),),
        )
        chat_ids = [str(row["chat_id"]) for row in chat_rows]
        if not chat_ids:
            return []
        placeholders = ",".join("?" for _ in chat_ids)
        return self._db.query(
            f"""
            SELECT * FROM (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY chat_id
                           ORDER BY created_at DESC, id DESC
                       ) AS _rank
                  FROM short_term_messages
                 WHERE chat_id IN ({placeholders})
                   AND {_LIVE_SQL}
            )
             WHERE _rank <= ?
               AND chat_id != ?
             ORDER BY created_at ASC, id ASC
            """,
            (*chat_ids, int(per_chat), exclude_chat_id or ""),
        )

    def recent_of_chat(self, chat_id: str, limit: int) -> list[sqlite3.Row]:
        """取某个会话最近的对话（按时间正序）。"""
        rows = self._db.query(
            f"""
            SELECT * FROM short_term_messages
             WHERE chat_id = ? AND {_LIVE_SQL}
             ORDER BY created_at DESC, id DESC
             LIMIT ?
            """,
            (chat_id, int(limit)),
        )
        return list(reversed(rows))

    def messages_before(
        self,
        chat_id: str,
        before_at: float,
        before_id: int,
        limit: int = 6,
    ) -> list[sqlite3.Row]:
        """取某会话在指定条目**之前**的若干条消息（按时间正序）。

        判定对象是「上次判定之后的增量」，而增量往往只有一两句
        （「不懂」「那他呢」）。缺少前文时，模型无法确定代词指向，
        容易判定为「无明确上下文」并保持沉默。此处补充窗口开始前的
        那段对话，使判据从「一句话」变为「一段话」。
        """
        if limit <= 0:
            return []
        rows = self._db.query(
            f"""
            SELECT * FROM short_term_messages
             WHERE chat_id = ? AND {_LIVE_SQL}
               AND (created_at < ? OR (created_at = ? AND id < ?))
             ORDER BY created_at DESC, id DESC
             LIMIT ?
            """,
            (chat_id, float(before_at), float(before_at), int(before_id), int(limit)),
        )
        return list(reversed(rows))

    def messages_since(
        self, chat_id: str, since: float, limit: int = 30
    ) -> list[sqlite3.Row]:
        """取某会话在 since 之后的消息（按时间正序）。

        巡检用它把「这段时间里发生了什么」交给决策模型：判定的对象是
        一整段对话，而不是被切成一条条、彼此脱节的碎片。
        """
        rows = self._db.query(
            f"""
            SELECT * FROM short_term_messages
             WHERE chat_id = ? AND created_at >= ? AND {_LIVE_SQL}
             ORDER BY created_at DESC, id DESC
             LIMIT ?
            """,
            (chat_id, float(since), int(limit)),
        )
        return list(reversed(rows))

    def search_history(
        self,
        *,
        chat_id: str | None,
        keyword: str = "",
        since: float = 0.0,
        limit: int = 20,
    ) -> list[sqlite3.Row]:
        """按关键词查询历史消息，返回时间正序。

        **刻意不使用 `_LIVE_SQL`**：该工具用于历史回查，已清空
        （`archived_at`）与已压入摘要（`condensed_at`）的行正是查询目标。
        唯一排除睡眠期暂存（`staged = 1`），这些消息在起床前不应对模型可见。

        关键词同时匹配正文、发言人名称与图片描述，使图片内容也可通过描述检索。
        """
        limit = max(int(limit), 1)
        clauses = ["staged = 0", "created_at >= ?"]
        params: list = [float(since)]
        if chat_id:
            clauses.append("chat_id = ?")
            params.append(chat_id)
        text = (keyword or "").strip()
        if text:
            like = f"%{text}%"
            clauses.append(
                "(content LIKE ? OR sender_name LIKE ? OR image_caption LIKE ?)"
            )
            params.extend([like, like, like])
        params.append(limit)
        rows = self._db.query(
            f"""
            SELECT * FROM short_term_messages
             WHERE {' AND '.join(clauses)}
             ORDER BY created_at DESC, id DESC
             LIMIT ?
            """,
            tuple(params),
        )
        return list(reversed(rows))

    def page(
        self, chat_id: str | None, limit: int, offset: int
    ) -> tuple[list[sqlite3.Row], int]:
        """分页读短期账本（可按会话过滤），返回 (本页消息, 总数)。

        走 `_LIVE_SQL`：页面与上下文**同源**——已归档、已压入摘要的行
        不应仅在此处出现。上下文读取与页面展示必须是同一批条目。
        """
        limit = max(int(limit), 1)
        offset = max(int(offset), 0)
        if chat_id:
            total_row = self._db.query_one(
                f"SELECT COUNT(*) AS n FROM short_term_messages "
                f"WHERE chat_id = ? AND {_LIVE_SQL}",
                (chat_id,),
            )
            rows = self._db.query(
                f"""
                SELECT * FROM short_term_messages
                 WHERE chat_id = ? AND {_LIVE_SQL}
                 ORDER BY created_at DESC, id DESC
                 LIMIT ? OFFSET ?
                """,
                (chat_id, limit, offset),
            )
        else:
            total_row = self._db.query_one(
                f"SELECT COUNT(*) AS n FROM short_term_messages WHERE {_LIVE_SQL}"
            )
            rows = self._db.query(
                f"""
                SELECT * FROM short_term_messages
                 WHERE {_LIVE_SQL}
                 ORDER BY created_at DESC, id DESC
                 LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )
        total = int(total_row["n"]) if total_row else 0
        return rows, total

    def list_known_chats(self, limit: int = 20) -> list[sqlite3.Row]:
        """列出有过活动记录的会话，供 `say` 选择目标会话。"""
        return self._db.query(
            f"""
            SELECT chat_id,
                   chat_label,
                   chat_kind,
                   COUNT(*)      AS message_count,
                   MAX(created_at) AS last_at
              FROM short_term_messages
             WHERE {_LIVE_SQL}
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

    def retire_summary(self, summary_id: int) -> None:
        """把一条摘要移出短期层（内容已被提升为长期记忆）。"""
        if not summary_id:
            return
        self._db.execute(
            "UPDATE short_term_summaries SET condensed_at = ? WHERE id = ?",
            (time.time(), int(summary_id)),
        )

    # ── 归档 ──────────────────────────────────────────────

    def clear_short_term(self, now: float) -> tuple[int, int]:
        """清空短期层，返回 (归档消息数, 收起摘要数)。

        这是深夜收口的最后一步：当天摘要已写入长期层，短期层不应再保留内容——
        「整理后短期记忆为空」是这套设计的验收点。

        **不是物理删除**：原始消息标记 `archived_at`、摘要标记 `condensed_at`，
        数据均保留供 `search_history` 回查（保留期见 `purge_expired`）。

        仅处理 `staged = 0` 的消息：睡眠期暂存消息需等起床后走正常流程，
        不应在归档时一并清理。
        """
        with self._db.lock:
            messages = self._db.execute(
                """
                UPDATE short_term_messages SET archived_at = ?
                 WHERE staged = 0 AND archived_at = 0
                """,
                (float(now),),
            )
            summaries = self._db.execute(
                """
                UPDATE short_term_summaries SET condensed_at = ?
                 WHERE condensed_at = 0
                """,
                (float(now),),
            )
        return int(messages.rowcount or 0), int(summaries.rowcount or 0)

    def purge_expired(self, retention_days: int) -> int:
        """物理删除超过保留期的旧内容，返回删除的消息条数。

        两类数据都会过期，均为「已离开上下文、等待回查窗口结束」的记录：

        - **已归档**的原始消息（每日整理时移出上下文的批次）
        - **已压入摘要**的原始消息（压缩时移出上下文的批次）

        缺少此步骤会导致数据库无限增长：分会话淘汰（`prune`）退役后，这里是唯一
        的清理路径。已被吸收的旧摘要（`condensed_at > 0`）一并删除。
        """
        days = max(int(retention_days), 0)
        if days <= 0:
            return 0
        cutoff = time.time() - days * 86400
        with self._db.lock:
            cursor = self._db.execute(
                """
                DELETE FROM short_term_messages
                 WHERE (archived_at > 0 AND archived_at < ?)
                    OR (condensed_at > 0 AND condensed_at < ?)
                """,
                (cutoff, cutoff),
            )
            self._db.execute(
                """
                DELETE FROM short_term_summaries
                 WHERE condensed_at > 0 AND condensed_at < ?
                """,
                (cutoff,),
            )
        return int(cursor.rowcount or 0)

    # ── 淘汰 ──────────────────────────────────────────────
    #
    # 分会话淘汰（每会话保留最近 N 条）已随位置榜一同退役：该策略会挤掉摘要，
    # 且与「条目总数超阈值才压缩」存在规则冲突。
    # 当前短期层体量由「压缩 + 保留期清理」共同约束，见 condense/apply_condensation
    # 与 purge_expired。
