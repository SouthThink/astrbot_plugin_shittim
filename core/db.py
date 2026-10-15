"""SQLite 存储层：建表、schema 版本与连接管理。

设计取舍：
- 单连接 + 可重入锁，跨线程复用（sqlite3 的 check_same_thread=False）。
  什亭之枢的全部读写都经由这一个入口，避免多连接写冲突。
- 记忆存储是跨重启的长期数据，因此落盘而非纯内存。
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

SCHEMA_VERSION = 3

# 增量列迁移：用于给已存在的库补列，避免升级插件时丢数据
_COLUMN_MIGRATIONS = (
    # gated=1 表示这条消息不在 AstrBot 的历史里：要么被决断层拦下，
    # 要么是插件主动发出的。跨会话上下文档据此把主脑看不到的内容补进去。
    ("short_term_messages", "gated", "INTEGER NOT NULL DEFAULT 0"),
)

_DDL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- 短期记忆：一段时间内所有群聊/私聊的原始消息，带来源标记以防搞混
CREATE TABLE IF NOT EXISTS short_term_messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id     TEXT    NOT NULL,
    chat_label  TEXT    NOT NULL,
    chat_kind   TEXT    NOT NULL,
    sender_id   TEXT    NOT NULL DEFAULT '',
    sender_name TEXT    NOT NULL DEFAULT '',
    role        TEXT    NOT NULL,
    content     TEXT    NOT NULL,
    created_at  REAL    NOT NULL,
    extracted   INTEGER NOT NULL DEFAULT 0,
    gated       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_stm_created ON short_term_messages (created_at);
CREATE INDEX IF NOT EXISTS idx_stm_chat    ON short_term_messages (chat_id, created_at);
CREATE INDEX IF NOT EXISTS idx_stm_pending ON short_term_messages (extracted, id);

-- 长期 / 永久记忆
CREATE TABLE IF NOT EXISTS memories (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    tier             TEXT    NOT NULL DEFAULT 'long',
    content          TEXT    NOT NULL,
    source_quote     TEXT    NOT NULL DEFAULT '',
    tags             TEXT    NOT NULL DEFAULT '[]',
    keywords         TEXT    NOT NULL DEFAULT '[]',
    chat_id          TEXT,
    chat_label       TEXT,
    user_id          TEXT,
    user_name        TEXT,
    strength         REAL    NOT NULL DEFAULT 5.0,
    recall_count     INTEGER NOT NULL DEFAULT 0,
    useful_count     INTEGER NOT NULL DEFAULT 0,
    cross_chat_count INTEGER NOT NULL DEFAULT 1,
    manual           INTEGER NOT NULL DEFAULT 0,
    created_at       REAL    NOT NULL,
    last_recalled_at REAL,
    last_decay_at    REAL    NOT NULL,
    merged_from      TEXT    NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_mem_tier  ON memories (tier, strength);
CREATE INDEX IF NOT EXISTS idx_mem_user  ON memories (user_id);
CREATE INDEX IF NOT EXISTS idx_mem_stale ON memories (last_recalled_at);

-- 关键词倒排表：召回时先在此表做候选筛选，避免全表扫描
CREATE TABLE IF NOT EXISTS memory_keywords (
    memory_id INTEGER NOT NULL,
    keyword   TEXT    NOT NULL,
    PRIMARY KEY (memory_id, keyword)
);
CREATE INDEX IF NOT EXISTS idx_mk_keyword ON memory_keywords (keyword);

-- 用户账本：跨会话识别同一个人
CREATE TABLE IF NOT EXISTS user_ledger (
    platform   TEXT NOT NULL,
    user_id    TEXT NOT NULL,
    user_names TEXT NOT NULL DEFAULT '[]',
    first_seen REAL NOT NULL,
    last_seen  REAL NOT NULL,
    PRIMARY KEY (platform, user_id)
);

-- 会话焦点：她当前在关注哪些会话（只约束主动发言，不约束被动接话）
CREATE TABLE IF NOT EXISTS session_focus (
    chat_id    TEXT PRIMARY KEY,
    chat_label TEXT NOT NULL DEFAULT '',
    focused_at REAL NOT NULL,
    left       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_focus_recent ON session_focus (left, focused_at);
"""


class MemoryDB:
    """什亭之枢记忆库的连接与建表入口。"""

    def __init__(self, data_dir: Path | str):
        self.path = Path(data_dir) / "shittim_memory.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_DDL)
            self._conn.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            self._conn.commit()
            self._migrate_columns()

    def _migrate_columns(self) -> None:
        """给已存在的库补齐新增列。必须是幂等的。"""
        for table, column, column_type in _COLUMN_MIGRATIONS:
            existing = {
                row["name"]
                for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if column not in existing:
                self._conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} {column_type}"
                )
        self._conn.execute(
            "UPDATE meta SET value = ? WHERE key = 'schema_version'",
            (str(SCHEMA_VERSION),),
        )
        self._conn.commit()

    @property
    def lock(self) -> threading.RLock:
        """对外暴露的锁，供需要「多次操作保持原子」的调用方使用。"""
        return self._lock

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cursor = self._conn.execute(sql, params)
            self._conn.commit()
            return cursor

    def executemany(self, sql: str, rows: list[tuple]) -> None:
        if not rows:
            return
        with self._lock:
            self._conn.executemany(sql, rows)
            self._conn.commit()

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def query_one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass
