"""会话焦点：当前处于关注状态的会话。

## 焦点的作用范围

**只约束主动发言**，不约束被动接话。

- 被动接话：任何会话收到消息都会触发唤醒，交由决断层判断是否回复，不受焦点限制。
- 主动发言：只能向拥有焦点的会话主动发言（`say`）。

该设计的目的是让主动行为符合人类习惯：有人对自己说话时随时可以回应，
但不会前往一个早已无人发言的会话自说自话。

## 生命周期

- 收到某会话的新消息         → 该会话获得 / 刷新焦点
- 主动向某会话发言（say）    → 该会话获得 / 刷新焦点
- 某会话安静超过 timeout     → 焦点失效
- 主动 leave / rest          → 立即失效

注意：焦点失效不等于「离场」——下一条消息仍会触发唤醒，仍会进入决断层判断，
它只收回「主动开口」的资格。
"""

from __future__ import annotations

import time

from ..common.db import MemoryDB

DEFAULT_TIMEOUT_MINUTES = 30.0
# 焦点记录超过该天数就清理，避免表无限增长
_PRUNE_AFTER_DAYS = 14.0


class FocusManager:
    """会话焦点的读写。"""

    def __init__(self, db: MemoryDB, config):
        self._db = db
        self._config = config

    def update_config(self, config) -> None:
        self._config = config

    # ── 写入 ──────────────────────────────────────────────

    def touch(self, chat_id: str, chat_label: str | None = None) -> None:
        """使某会话获得或刷新焦点（新消息到达、被叫、主动前往）。"""
        if not chat_id:
            return
        now = time.time()
        row = self._db.query_one(
            "SELECT chat_id FROM session_focus WHERE chat_id = ?", (chat_id,)
        )
        if row is None:
            self._db.execute(
                """
                INSERT INTO session_focus (chat_id, chat_label, focused_at, left)
                VALUES (?, ?, ?, 0)
                """,
                (chat_id, chat_label or "", now),
            )
            return
        self._db.execute(
            """
            UPDATE session_focus
               SET chat_label = COALESCE(NULLIF(?, ''), chat_label),
                   focused_at = ?,
                   left = 0
             WHERE chat_id = ?
            """,
            (chat_label or "", now, chat_id),
        )

    def leave(self, chat_id: str) -> bool:
        """主动离开某会话：收回该会话的主动发言资格。"""
        if not self.get_record(chat_id):
            return False
        self._db.execute(
            "UPDATE session_focus SET left = 1 WHERE chat_id = ?", (chat_id,)
        )
        return True

    def rest(self) -> int:
        """主动进入休息：一次性收回所有会话的主动发言资格。"""
        cursor = self._db.execute(
            "UPDATE session_focus SET left = 1 WHERE left = 0"
        )
        return int(cursor.rowcount or 0)

    def prune(self, max_age_days: float = _PRUNE_AFTER_DAYS) -> int:
        """清理长期未动的焦点记录。"""
        cutoff = time.time() - max(float(max_age_days), 1.0) * 86400.0
        cursor = self._db.execute(
            "DELETE FROM session_focus WHERE focused_at < ?", (cutoff,)
        )
        return int(cursor.rowcount or 0)

    # ── 读取 ──────────────────────────────────────────────

    def get_record(self, chat_id: str):
        if not chat_id:
            return None
        return self._db.query_one(
            "SELECT * FROM session_focus WHERE chat_id = ?", (chat_id,)
        )

    def is_focused(self, chat_id: str) -> bool:
        """该会话是否仍在关注中（未主动离开，且未超时）。"""
        row = self.get_record(chat_id)
        if row is None or int(row["left"] or 0) == 1:
            return False
        return self.idle_seconds(chat_id) <= self._timeout_seconds

    def idle_seconds(self, chat_id: str) -> float:
        """距上次获得焦点已过去多少秒；无记录返回正无穷。"""
        row = self.get_record(chat_id)
        if row is None:
            return float("inf")
        return max(time.time() - float(row["focused_at"] or 0), 0.0)

    def list_focused(self, limit: int = 50) -> list:
        """当前仍在关注中的会话，按最近活跃排序。"""
        cutoff = time.time() - self._timeout_seconds
        return self._db.query(
            """
            SELECT * FROM session_focus
             WHERE left = 0 AND focused_at >= ?
             ORDER BY focused_at DESC
             LIMIT ?
            """,
            (cutoff, int(limit)),
        )

    # ── 内部 ──────────────────────────────────────────────

    @property
    def _timeout_seconds(self) -> float:
        try:
            minutes = float(self._config.timeout_minutes)
        except (TypeError, ValueError):
            minutes = DEFAULT_TIMEOUT_MINUTES
        return max(minutes, 0.0) * 60.0
