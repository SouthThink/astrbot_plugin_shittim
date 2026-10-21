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

SCHEMA_VERSION = 10

# 增量列迁移：用于给已存在的库补列，避免升级插件时丢数据
_COLUMN_MIGRATIONS = (
    # gated=1 表示这条消息不在 AstrBot 的历史里：要么被决断层拦下，
    # 要么是插件主动发出的。跨会话上下文档据此把主脑看不到的内容补进去。
    ("short_term_messages", "gated", "INTEGER NOT NULL DEFAULT 0"),
    # source 取值 live / backfill / sleep。回填的历史数据量大且时间久远，
    # 全部注入 req.contexts 会超出上下文窗口，因此合并回上下文时只处理 live；
    # 睡眠期收到的消息同样不参与合并，改由起床提示块按 token 预算呈现。
    ("short_term_messages", "source", "TEXT NOT NULL DEFAULT 'live'"),
    # 图片消息的转述文本。AstrBot 只会把图片转成「[图片]」占位，
    # 因此账本需额外记录一句描述，否则该图片在上下文中不可见。
    ("short_term_messages", "image_caption", "TEXT NOT NULL DEFAULT ''"),
    # 合并转发的聊天记录。AstrBot 只会把合并转发转成「[转发消息]」占位，
    # 内层内容需另调接口取回后单独记录。
    ("short_term_messages", "forward_content", "TEXT NOT NULL DEFAULT ''"),
    # staged=1 表示该消息仍处于「睡眠期暂存区」：不参与整理、不参与归档。
    # 起床时统一置 0，纳入正常流程。与 source 分开是因为 source 记录的是永久事实
    # （「睡觉时收到」始终成立），而「是否参与整理」仅在睡眠期间成立。
    ("short_term_messages", "staged", "INTEGER NOT NULL DEFAULT 0"),
    # archived_at > 0 表示这条消息已归档（每日整理时移出常规路径）。
    # "整理完短期记忆为空"是**从常规查询与上下文中排除**，不是物理删除；
    # 数据保留 archive_retention_days 天，供 search_history 回查。
    ("short_term_messages", "archived_at", "REAL NOT NULL DEFAULT 0"),
    # condensed_at > 0 表示这条消息已被压缩进一条短期摘要（见 short_term_summaries）：
    # 它不再进上下文，但数据保留——摘要可能压得过狠，回查工具还要用它。
    ("short_term_messages", "condensed_at", "REAL NOT NULL DEFAULT 0"),
    # 长期记忆的上浮轴：每晚整理产出、每次被召回都刷新为当下，
    # 查询按它倒序得到「位置榜」。取代了原先的 strength 权重（strength 已退役，
    # 列保留只为兼容旧库，不再读写）。
    ("memories", "promoted_at", "REAL NOT NULL DEFAULT 0"),
    # 清晰度积分（§9.6）：新建 = clarity_full(4)，每次召回 +recall_gain(3)，
    # 每轮整理 -1（下限 -1）。等级由分数推导，不单独存字段，避免两者不同步。
    ("memories", "curation_score", "INTEGER NOT NULL DEFAULT 4"),
    # 累计被降级的次数：遗忘阶梯判据里的「陈旧度」
    ("memories", "curation_count", "INTEGER NOT NULL DEFAULT 0"),
    ("memories", "last_curated_at", "REAL NOT NULL DEFAULT 0"),
    # 被明确要求忘记（§10.3）：立即停止召回，等下一轮整理物理删除
    ("memories", "forget_requested", "INTEGER NOT NULL DEFAULT 0"),
    ("memories", "forget_requested_at", "REAL NOT NULL DEFAULT 0"),
)

# 建在「迁移后才存在的列」上的索引：**必须**在补列之后创建。
# 放进 _DDL 会让老库当场报 "no such column"——建表脚本先于补列执行，
# 那时这些列还不存在。
_POST_MIGRATION_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_stm_live ON short_term_messages (archived_at, id)",
    "CREATE INDEX IF NOT EXISTS idx_stm_condensed "
    "ON short_term_messages (condensed_at, archived_at, id)",
    "CREATE INDEX IF NOT EXISTS idx_mem_promoted "
    "ON memories (tier, promoted_at DESC)",
    # strength 已退役（权重召回被位置榜取代），旧库上的该索引只会造成误导
    "DROP INDEX IF EXISTS idx_mem_tier",
    # extracted 的「待提炼」通道已并入短期收口（压缩即提炼），该索引只会增加记账开销
    "DROP INDEX IF EXISTS idx_stm_pending",
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
    -- 已退役：原先标记「这条已提炼进长期层」。提炼通道并入短期收口后，
    -- 「是否已被处理过」只看 condensed_at / archived_at，此列不再读写。
    extracted   INTEGER NOT NULL DEFAULT 0,
    gated       INTEGER NOT NULL DEFAULT 0,
    source      TEXT    NOT NULL DEFAULT 'live',
    image_caption TEXT  NOT NULL DEFAULT '',
    forward_content TEXT NOT NULL DEFAULT '',
    staged      INTEGER NOT NULL DEFAULT 0,
    archived_at REAL    NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_stm_created ON short_term_messages (created_at);
CREATE INDEX IF NOT EXISTS idx_stm_chat    ON short_term_messages (chat_id, created_at);

-- 图片转述缓存：同一个引用只问一次视觉模型，省下重复调用的钱和时间
CREATE TABLE IF NOT EXISTS image_captions (
    ref_hash   TEXT PRIMARY KEY,
    caption    TEXT NOT NULL,
    created_at REAL NOT NULL
);

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
    merged_from      TEXT    NOT NULL DEFAULT '[]',
    curation_score   INTEGER NOT NULL DEFAULT 4,
    curation_count   INTEGER NOT NULL DEFAULT 0,
    last_curated_at  REAL    NOT NULL DEFAULT 0,
    forget_requested INTEGER NOT NULL DEFAULT 0,
    forget_requested_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_mem_tier  ON memories (tier, strength);
CREATE INDEX IF NOT EXISTS idx_mem_user  ON memories (user_id);
CREATE INDEX IF NOT EXISTS idx_mem_stale ON memories (last_recalled_at);

-- 短期摘要：把一段较早的对话（或更早的摘要）压成一条，构成短期层的「远期背景」。
-- 它是分层压缩的产物，也是二次合并的输入——被吸收时同样置 condensed_at。
-- 刻意**不参与每日归档**：平常日子（不到阈值）不产生摘要，起床后短期层仍为空；
-- 爆发日留下的摘要则跨天留着，继续被更高一层的摘要吸收。
CREATE TABLE IF NOT EXISTS short_term_summaries (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    content       TEXT    NOT NULL,
    created_at    REAL    NOT NULL,               -- 生成时刻
    covered_at    REAL    NOT NULL,               -- 覆盖区间的末尾时间，决定它在时间线里的位置
    covered_count INTEGER NOT NULL DEFAULT 0,     -- 被它压掉的条目数（供页面与日志）
    covered_items TEXT    NOT NULL DEFAULT '[]',  -- 被压掉的摘要 id，供溯源
    condensed_at  REAL    NOT NULL DEFAULT 0      -- >0 表示已被更高一层的摘要吸收
);
CREATE INDEX IF NOT EXISTS idx_sts_live ON short_term_summaries (condensed_at, covered_at);

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

-- 会话焦点：当前处于关注状态的会话（只约束主动发言，不约束被动接话）
CREATE TABLE IF NOT EXISTS session_focus (
    chat_id      TEXT PRIMARY KEY,
    chat_label   TEXT NOT NULL DEFAULT '',
    focused_at   REAL NOT NULL,
    left         INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_focus_recent ON session_focus (left, focused_at);

-- 被要求遗忘的内容指纹：防止同一条信息在下一次压缩时被重新写回来（§13.4）
CREATE TABLE IF NOT EXISTS suppressed_memories (
    keyword_hash TEXT PRIMARY KEY,
    content      TEXT NOT NULL,
    created_at   REAL NOT NULL
);

-- 全局作息状态（单行）。状态本身由时间轴推导，这里只持久化「事实」：
-- 本轮窗口的起止、被叫醒/宽限时刻、待注入的起床提示。这样重启后算一次即可恢复。
CREATE TABLE IF NOT EXISTS sleep_state (
    id                   INTEGER PRIMARY KEY CHECK (id = 1),
    plan_date            TEXT    NOT NULL DEFAULT '',   -- 本轮作息所属日期（入睡日）
    sleep_at             REAL    NOT NULL DEFAULT 0,    -- 本轮入睡时刻
    wake_at              REAL    NOT NULL DEFAULT 0,    -- 本轮起床时刻
    phase                TEXT    NOT NULL DEFAULT '',   -- 上次观察到的阶段，用于识别转换
    postponed_hours      REAL    NOT NULL DEFAULT 0,    -- 本轮已推迟小时数
    grace_until          REAL    NOT NULL DEFAULT 0,    -- 清醒宽限截止
    awakened_at          REAL    NOT NULL DEFAULT 0,    -- 最近一次被叫醒时刻
    wake_notice_pending  INTEGER NOT NULL DEFAULT 0,    -- 起床提示待注入
    wake_notice_fired_at REAL    NOT NULL DEFAULT 0,    -- 起床提示已注入时刻
    last_curation_at     REAL    NOT NULL DEFAULT 0,    -- 上次深睡整理时刻
    last_cleared_at      REAL    NOT NULL DEFAULT 0,    -- 上次归档时刻
    updated_at           REAL    NOT NULL DEFAULT 0
);
"""


class _NullCursor:
    """连接已关闭时的空游标。

    插件卸载（terminate）会关闭连接，但此时可能仍有 handler 停留在 await 中
    （例如等待决策模型返回）。其恢复后会继续读写，从而触发「Cannot operate on
    a closed database」。与其在每个调用点判空，不如在此提供一个无副作用的
    替身对象，使迟到的写入静默失效。
    """

    lastrowid = 0
    rowcount = 0

    def fetchall(self) -> list:
        return []

    def fetchone(self):
        return None

    def fetchmany(self, size: int = 1) -> list:
        return []

    def __iter__(self):
        return iter(())


class MemoryDB:
    """什亭之枢记忆库的连接与建表入口。"""

    def __init__(self, data_dir: Path | str):
        self.path = Path(data_dir) / "shittim_memory.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._closed = False
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
        # 索引要等列补齐之后再建：老库此时才第一次拥有这些列
        for statement in _POST_MIGRATION_INDEXES:
            self._conn.execute(statement)
        self._backfill_promoted_at()
        self._conn.execute(
            "UPDATE meta SET value = ? WHERE key = 'schema_version'",
            (str(SCHEMA_VERSION),),
        )
        self._conn.commit()

    def _backfill_promoted_at(self) -> None:
        """老库补上浮轴。

        新列的默认值是 0，整榜的先后会退化成「按 id」，与它们原本的清晰度无关。
        所以用已有的时间信息回填一次：先按最后一轮整理时刻，再退回创建时刻。
        只在 `promoted_at = 0` 的行上动手，重启不会反复改写。
        """
        self._conn.execute(
            """
            UPDATE memories
               SET promoted_at = CASE
                       WHEN last_curated_at > 0 THEN last_curated_at
                       WHEN last_recalled_at > 0 THEN last_recalled_at
                       ELSE created_at
                   END
             WHERE promoted_at = 0
            """
        )

    @property
    def lock(self) -> threading.RLock:
        """对外暴露的锁，供需要「多次操作保持原子」的调用方使用。"""
        return self._lock

    @property
    def closed(self) -> bool:
        """连接是否已关闭。调用方可据此避免把「丢弃」报告成「成功」。"""
        return self._closed

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor | _NullCursor:
        if self._closed:
            return _NullCursor()
        with self._lock:
            cursor = self._conn.execute(sql, params)
            self._conn.commit()
            return cursor

    def executemany(self, sql: str, rows: list[tuple]) -> None:
        if not rows or self._closed:
            return
        with self._lock:
            self._conn.executemany(sql, rows)
            self._conn.commit()

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        if self._closed:
            return []
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def query_one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        if self._closed:
            return None
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def close(self) -> None:
        """关闭连接。幂等；关闭后所有读写静默失效而不是抛异常。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._conn.close()
            except Exception:
                pass
