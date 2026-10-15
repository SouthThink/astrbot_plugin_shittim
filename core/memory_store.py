"""长期 / 永久记忆的存储、召回打分与去重。

三层语义：
- long      : 长期记忆，会随时间衰减、参与合并、可能被遗忘
- permanent : 永久记忆，不衰减、不合并、不遗忘

召回采用双通道：
- 消息通道：按当前消息的相似度召回（「我记得好像……」）
- 用户通道：按发送者召回这个人的相关记忆

两条通道都只是「辅助参考」，是否真的进入回复由主模型决定，
注入时另有反过拟合指令约束（见 core/injector.py）。
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field

from .db import MemoryDB
from .tokenizer import extract_keywords, overlap_score, tokenize

TIER_LONG = "long"
TIER_PERMANENT = "permanent"

# 权重：关键词相关性 > 时间近因 ≈ 强度 ≈ 用户匹配
_W_KEYWORD = 0.55
_W_RECENCY = 0.15
_W_STRENGTH = 0.15
_W_USER = 0.15

# 用户通道：偏重重要性，但仍让话题相关的记忆优先浮出
_WU_KEYWORD = 0.55
_WU_IMPORTANCE = 0.45

# 判定「同一条记忆」的关键词重合度
_DUPLICATE_OVERLAP = 0.6

_MAX_QUERY_TOKENS = 48


def _json_list(raw: str | None) -> list:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return value if isinstance(value, list) else []


@dataclass
class Memory:
    """一条长期/永久记忆。"""

    id: int = 0
    tier: str = TIER_LONG
    content: str = ""
    source_quote: str = ""
    tags: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    chat_id: str | None = None
    chat_label: str | None = None
    user_id: str | None = None
    user_name: str | None = None
    strength: float = 5.0
    recall_count: int = 0
    useful_count: int = 0
    cross_chat_count: int = 1
    manual: int = 0
    created_at: float = 0.0
    last_recalled_at: float | None = None
    last_decay_at: float = 0.0
    merged_from: list[int] = field(default_factory=list)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Memory":
        return cls(
            id=int(row["id"]),
            tier=str(row["tier"]),
            content=str(row["content"] or ""),
            source_quote=str(row["source_quote"] or ""),
            tags=[str(t) for t in _json_list(row["tags"])],
            keywords=[str(k) for k in _json_list(row["keywords"])],
            chat_id=row["chat_id"],
            chat_label=row["chat_label"],
            user_id=row["user_id"],
            user_name=row["user_name"],
            strength=float(row["strength"] or 0.0),
            recall_count=int(row["recall_count"] or 0),
            useful_count=int(row["useful_count"] or 0),
            cross_chat_count=int(row["cross_chat_count"] or 1),
            manual=int(row["manual"] or 0),
            created_at=float(row["created_at"] or 0.0),
            last_recalled_at=row["last_recalled_at"],
            last_decay_at=float(row["last_decay_at"] or 0.0),
            merged_from=[int(i) for i in _json_list(row["merged_from"])],
        )

    def activity_at(self) -> float:
        """最近一次「活跃」时间：召回过就用召回时间，否则用创建时间。"""
        return float(self.last_recalled_at or self.created_at or 0.0)


@dataclass
class RecalledMemory:
    """召回结果：记忆本体 + 得分 + 来源通道，便于日志排障。"""

    memory: Memory
    score: float
    channel: str  # "message" | "user"


class MemoryStore:
    """记忆表的读写与召回。"""

    def __init__(self, db: MemoryDB, config):
        self._db = db
        self._config = config

    @property
    def lock(self):
        """供需要「多步写入保持原子」的调用方（如合并）获取库级锁。"""
        return self._db.lock

    def update_config(self, config) -> None:
        self._config = config

    # ── 写入 ──────────────────────────────────────────────

    def add(
        self,
        *,
        content: str,
        source_quote: str = "",
        tags: list[str] | None = None,
        keywords: list[str] | None = None,
        chat_id: str | None = None,
        chat_label: str | None = None,
        user_id: str | None = None,
        user_name: str | None = None,
        strength: float = 5.0,
        tier: str = TIER_LONG,
        manual: bool = False,
        merged_from: list[int] | None = None,
        created_at: float | None = None,
    ) -> int:
        """写入一条记忆，返回自增 id。

        manual=True 表示「她/用户明确要求记住」，直接落永久层，不等阈值晋升。
        """
        now = float(created_at if created_at is not None else time.time())
        keyword_list = [k for k in (keywords or extract_keywords(content)) if k]
        if manual:
            tier = TIER_PERMANENT
        cursor = self._db.execute(
            """
            INSERT INTO memories
                (tier, content, source_quote, tags, keywords, chat_id, chat_label,
                 user_id, user_name, strength, recall_count, useful_count,
                 cross_chat_count, manual, created_at, last_recalled_at,
                 last_decay_at, merged_from)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 1, ?, ?, NULL, ?, ?)
            """,
            (
                tier,
                content,
                source_quote,
                json.dumps(tags or [], ensure_ascii=False),
                json.dumps(keyword_list, ensure_ascii=False),
                chat_id,
                chat_label,
                user_id,
                user_name,
                float(strength),
                1 if manual else 0,
                now,
                now,
                json.dumps(merged_from or [], ensure_ascii=False),
            ),
        )
        memory_id = int(cursor.lastrowid or 0)
        self._index_keywords(memory_id, keyword_list)
        return memory_id

    def reinforce_or_add(
        self,
        *,
        content: str,
        source_quote: str = "",
        tags: list[str] | None = None,
        keywords: list[str] | None = None,
        chat_id: str | None = None,
        chat_label: str | None = None,
        user_id: str | None = None,
        user_name: str | None = None,
        strength: float = 5.0,
    ) -> tuple[int, bool]:
        """去重写入。

        若已存在高度相似的长期记忆，则强化它（提升强度、必要时累加跨会话计数），
        否则新建。返回 (记忆 id, 是否新建)。
        """
        keyword_list = [k for k in (keywords or extract_keywords(content)) if k]
        duplicate = self._find_duplicate(keyword_list, user_id=user_id)
        if duplicate is None:
            return (
                self.add(
                    content=content,
                    source_quote=source_quote,
                    tags=tags,
                    keywords=keyword_list,
                    chat_id=chat_id,
                    chat_label=chat_label,
                    user_id=user_id,
                    user_name=user_name,
                    strength=strength,
                ),
                True,
            )

        now = time.time()
        cross_delta = 1 if (chat_id and duplicate.chat_id and chat_id != duplicate.chat_id) else 0
        self._db.execute(
            """
            UPDATE memories
               SET strength = MIN(strength + ?, 20.0),
                   cross_chat_count = cross_chat_count + ?,
                   last_decay_at = ?
             WHERE id = ?
            """,
            (max(float(strength) * 0.5, 0.5), cross_delta, now, duplicate.id),
        )
        return duplicate.id, False

    def update(self, memory_id: int, **fields) -> None:
        """按字段名更新记忆，仅允许白名单字段。"""
        allowed = {
            "tier", "content", "source_quote", "tags", "keywords", "chat_id",
            "chat_label", "user_id", "user_name", "strength", "recall_count",
            "useful_count", "cross_chat_count", "manual", "last_recalled_at",
            "last_decay_at", "merged_from",
        }
        assignments = {k: v for k, v in fields.items() if k in allowed}
        if not assignments:
            return
        # JSON 字段统一序列化
        for key in ("tags", "keywords", "merged_from"):
            if key in assignments and not isinstance(assignments[key], str):
                assignments[key] = json.dumps(assignments[key], ensure_ascii=False)
        columns = ", ".join(f"{key} = ?" for key in assignments)
        self._db.execute(
            f"UPDATE memories SET {columns} WHERE id = ?",
            (*assignments.values(), memory_id),
        )
        if "keywords" in assignments:
            self._index_keywords(memory_id, json.loads(assignments["keywords"]))

    def delete(self, memory_id: int) -> None:
        with self._db.lock:
            self._db.execute("DELETE FROM memory_keywords WHERE memory_id = ?", (memory_id,))
            self._db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))

    def mark_recalled(self, memory_ids: list[int]) -> None:
        """召回计数 +1 并刷新最近召回时间（用于衰减与晋升判定）。"""
        if not memory_ids:
            return
        now = time.time()
        placeholders = ",".join("?" for _ in memory_ids)
        self._db.execute(
            f"""
            UPDATE memories
               SET recall_count = recall_count + 1,
                   last_recalled_at = ?
             WHERE id IN ({placeholders})
            """,
            (now, *memory_ids),
        )

    def mark_useful(self, memory_id: int) -> None:
        """标记为「确实用上了」，是比召回次数更可信的重要性信号。"""
        self._db.execute(
            """
            UPDATE memories
               SET useful_count = useful_count + 1,
                   strength = MIN(strength + 2.0, 20.0)
             WHERE id = ?
            """,
            (memory_id,),
        )

    def promote(self, memory_id: int) -> None:
        self._db.execute(
            "UPDATE memories SET tier = ? WHERE id = ?", (TIER_PERMANENT, memory_id)
        )

    # ── 读取 ──────────────────────────────────────────────

    def get(self, memory_id: int) -> Memory | None:
        row = self._db.query_one("SELECT * FROM memories WHERE id = ?", (memory_id,))
        return Memory.from_row(row) if row else None

    def get_many(self, memory_ids: list[int]) -> list[Memory]:
        if not memory_ids:
            return []
        placeholders = ",".join("?" for _ in memory_ids)
        rows = self._db.query(
            f"SELECT * FROM memories WHERE id IN ({placeholders})", tuple(memory_ids)
        )
        by_id = {int(r["id"]): Memory.from_row(r) for r in rows}
        return [by_id[i] for i in memory_ids if i in by_id]

    def list_by_tier(self, tier: str, limit: int = 500) -> list[Memory]:
        rows = self._db.query(
            "SELECT * FROM memories WHERE tier = ? ORDER BY strength DESC LIMIT ?",
            (tier, int(limit)),
        )
        return [Memory.from_row(r) for r in rows]

    def list_long_memories(self) -> list[Memory]:
        rows = self._db.query(
            "SELECT * FROM memories WHERE tier = ? ORDER BY id ASC", (TIER_LONG,)
        )
        return [Memory.from_row(r) for r in rows]

    def count_by_tier(self) -> dict[str, int]:
        rows = self._db.query("SELECT tier, COUNT(*) AS n FROM memories GROUP BY tier")
        return {str(r["tier"]): int(r["n"]) for r in rows}

    # ── 召回 ──────────────────────────────────────────────

    def recall(
        self,
        *,
        query_text: str,
        user_id: str | None = None,
        top_k: int | None = None,
        user_top_k: int | None = None,
        min_score: float | None = None,
        exclude_ids: set[int] | None = None,
    ) -> list[RecalledMemory]:
        """双通道召回：消息相关 + 发送者相关，合并去重后返回。"""
        top_k = self._config.recall_top_k if top_k is None else top_k
        user_top_k = self._config.recall_user_top_k if user_top_k is None else user_top_k
        min_score = self._config.recall_min_score if min_score is None else min_score
        exclude_ids = exclude_ids or set()

        query_tokens = set(tokenize(query_text)[:_MAX_QUERY_TOKENS])

        results: list[RecalledMemory] = []
        seen: set[int] = set()

        # 通道一：与当前消息相关的记忆
        message_hits = 0
        for memory, score in self._rank_by_keywords(query_tokens, limit=max(top_k * 6, 24)):
            if message_hits >= top_k:
                break
            if memory.id in exclude_ids or memory.id in seen or score < min_score:
                continue
            seen.add(memory.id)
            message_hits += 1
            results.append(RecalledMemory(memory=memory, score=score, channel="message"))

        # 通道二：这个人的相关记忆（无关键词下限，按重要性兜底）
        if user_id and user_top_k > 0:
            for memory, score in self._rank_user_memories(
                user_id, query_tokens, limit=user_top_k
            ):
                if memory.id in exclude_ids or memory.id in seen:
                    continue
                seen.add(memory.id)
                results.append(RecalledMemory(memory=memory, score=score, channel="user"))

        return results

    def _rank_by_keywords(
        self, query_tokens: set[str], limit: int
    ) -> list[tuple[Memory, float]]:
        """用倒排表筛出候选，再按综合分排序。"""
        now = time.time()
        candidates = self._candidates_by_keywords(query_tokens, limit)
        if not candidates:
            return []
        scored = [
            (memory, self._score(memory, query_tokens, now, user_id=None))
            for memory in candidates
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def _rank_user_memories(
        self, user_id: str, query_tokens: set[str], limit: int
    ) -> list[tuple[Memory, float]]:
        """某个用户的记忆，按「话题相关 + 重要性」排序，无阈值兜底。"""
        rows = self._db.query(
            "SELECT * FROM memories WHERE user_id = ? ORDER BY strength DESC LIMIT ?",
            (user_id, max(int(limit) * 8, 24)),
        )
        now = time.time()
        scored: list[tuple[Memory, float]] = []
        for row in rows:
            memory = Memory.from_row(row)
            keyword_score = overlap_score(query_tokens, memory.keywords)
            score = _WU_KEYWORD * keyword_score + _WU_IMPORTANCE * self._importance(
                memory, now
            )
            scored.append((memory, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: int(limit)]

    def _candidates_by_keywords(
        self, query_tokens: set[str], limit: int
    ) -> list[Memory]:
        """借倒排表先取候选，避免全表扫描；命中太少时用高强度记忆兜底。"""
        tokens = [t for t in query_tokens if t][: _MAX_QUERY_TOKENS]
        memory_ids: list[int] = []
        if tokens:
            placeholders = ",".join("?" for _ in tokens)
            rows = self._db.query(
                f"""
                SELECT memory_id, COUNT(*) AS hits
                  FROM memory_keywords
                 WHERE keyword IN ({placeholders})
                 GROUP BY memory_id
                 ORDER BY hits DESC
                 LIMIT ?
                """,
                (*tokens, int(limit)),
            )
            memory_ids = [int(r["memory_id"]) for r in rows]

        if len(memory_ids) < int(limit):
            # 兜底：把高强度/常被召回的长期记忆并入候选，缓解关键词漏召
            fallback = self._db.query(
                """
                SELECT id FROM memories
                 WHERE tier = ?
                 ORDER BY strength DESC, recall_count DESC
                 LIMIT ?
                """,
                (TIER_PERMANENT, int(limit)),
            )
            for row in fallback:
                candidate_id = int(row["id"])
                if candidate_id not in memory_ids:
                    memory_ids.append(candidate_id)

        return self.get_many(memory_ids)

    def _score(
        self, memory: Memory, query_tokens: set[str], now: float, user_id: str | None
    ) -> float:
        keyword_score = overlap_score(query_tokens, memory.keywords)
        recency = self._recency(memory, now)
        strength = min(memory.strength / 10.0, 1.0)
        user_factor = 1.0 if (user_id and memory.user_id == user_id) else 0.0
        return (
            _W_KEYWORD * keyword_score
            + _W_RECENCY * recency
            + _W_STRENGTH * strength
            + _W_USER * user_factor
        )

    def _recency(self, memory: Memory, now: float) -> float:
        """半衰期衰减：越久没被用到，时间分越低。永久记忆不衰减。"""
        if memory.tier == TIER_PERMANENT:
            return 1.0
        half_life = max(self._config.decay_half_life_days, 1.0)
        idle_days = max(now - memory.activity_at(), 0.0) / 86400.0
        return 0.5 ** (idle_days / half_life)

    def _importance(self, memory: Memory, now: float) -> float:
        """重要度 = 强度 + 近因 + 用过的次数。"""
        strength = min(memory.strength / 10.0, 1.0)
        useful = min(memory.useful_count / 3.0, 1.0)
        return 0.5 * strength + 0.3 * self._recency(memory, now) + 0.2 * useful

    # ── 去重与索引 ────────────────────────────────────────

    def _find_duplicate(self, keywords: list[str], user_id: str | None) -> Memory | None:
        """在倒排表里找关键词高度重合的既有长期记忆。"""
        tokens = [k for k in keywords if k]
        if not tokens:
            return None
        placeholders = ",".join("?" for _ in tokens)
        rows = self._db.query(
            f"""
            SELECT m.* FROM memories m
              JOIN memory_keywords mk ON mk.memory_id = m.id
             WHERE mk.keyword IN ({placeholders})
               AND m.tier = ?
             GROUP BY m.id
             ORDER BY COUNT(*) DESC
             LIMIT 8
            """,
            (*tokens, TIER_LONG),
        )
        best: Memory | None = None
        best_overlap = 0.0
        for row in rows:
            memory = Memory.from_row(row)
            # 用户维度不一致的不算同一条（避免把 A 的偏好并进 B）
            if (memory.user_id or "") != (user_id or ""):
                continue
            score = overlap_score(set(tokens), memory.keywords)
            if score >= _DUPLICATE_OVERLAP and score > best_overlap:
                best, best_overlap = memory, score
        return best

    def _index_keywords(self, memory_id: int, keywords: list[str]) -> None:
        with self._db.lock:
            self._db.execute(
                "DELETE FROM memory_keywords WHERE memory_id = ?", (memory_id,)
            )
            unique = {k for k in keywords if k}
            self._db.executemany(
                "INSERT OR IGNORE INTO memory_keywords (memory_id, keyword) VALUES (?, ?)",
                [(memory_id, k) for k in unique],
            )

    # ── 用户账本 ──────────────────────────────────────────

    def upsert_user(
        self, platform: str, user_id: str, user_name: str, now: float | None = None
    ) -> None:
        """登记见过的用户，使其可跨会话被识别为同一个人。"""
        if not user_id:
            return
        now = float(now if now is not None else time.time())
        row = self._db.query_one(
            "SELECT user_names FROM user_ledger WHERE platform = ? AND user_id = ?",
            (platform, user_id),
        )
        if row is None:
            self._db.execute(
                """
                INSERT INTO user_ledger (platform, user_id, user_names, first_seen, last_seen)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    platform,
                    user_id,
                    json.dumps([user_name] if user_name else [], ensure_ascii=False),
                    now,
                    now,
                ),
            )
            return
        names = [str(n) for n in _json_list(row["user_names"])]
        if user_name and user_name not in names:
            names.append(user_name)
        self._db.execute(
            "UPDATE user_ledger SET user_names = ?, last_seen = ? WHERE platform = ? AND user_id = ?",
            (json.dumps(names, ensure_ascii=False), now, platform, user_id),
        )
