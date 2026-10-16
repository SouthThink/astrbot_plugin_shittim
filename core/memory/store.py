"""长期 / 永久记忆的存储、召回打分与去重。

三层语义：
- long      : 长期记忆，排在一张**位置榜**上，尾部会变模糊、最终被遗忘
- permanent : 永久记忆，不参与榜序、不降级、不遗忘

## 位置榜取代权重

长期记忆不再有「强度」这个权重。它只有一根轴：**最后一次被想起的时刻**
（`promoted_at`）——每晚整理产出的记忆直接上浮到最前，每次被召回也上浮到最前。
用得少的自然沉下去，落到榜尾的逐级变模糊，沉到底且长期无人问津的才被遗忘。

这样做的理由：排序是**确定**的，能一眼看出「她现在记得最牢的是什么」；
而权重衰减要靠一堆参数和半衰期去凑，既难调也难解释。

## 两条召回路径的依据不同

- **自动插入**（拼上下文）：按榜序取前 N 条，与话题无关。
- **工具召回**（`recall`，表现为「我想想」）：按**关键词相关性**排序。
  若这里也按榜序，她会想起「榜上靠前但与当前话题无关」的事。

注入时另有反过拟合指令约束（见 core/memory/injector.py）。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field

from ..common.db import MemoryDB
from ..common.tokenizer import extract_keywords, overlap_score, tokenize

TIER_LONG = "long"
TIER_PERMANENT = "permanent"

# ── 位置榜（§3）──────────────────────────────────────────
# 榜序：谁最后被想起谁排前面。id 兜底保证同刻写入的顺序稳定。
_BOARD_ORDER = "promoted_at DESC, id DESC"

# 等级：0 完整 / 1 精简 / 2 关键词
LEVEL_FULL = 0
LEVEL_BRIEF = 1
LEVEL_KEYWORDS = 2

# 名次分档：榜前 60% 完整、到 85% 精简、之外只剩关键词
DEGRADE_BRIEF_RATIO = 0.60
DEGRADE_KEYWORD_RATIO = 0.85

# 遗忘只看榜尾：名次落在这个比例之后才算「沉到底」
_FORGET_TAIL_RATIO = 0.05
# 榜尾不足这么多条时不做尾部判定：20 条记忆的「后 5%」没有意义
_FORGET_TAIL_MIN = 20


def curation_level(rank: int, total: int) -> int:
    """由**名次**推导等级。

    纯函数、不落库：名次本身是每次查询算出来的，等级再存一份必然不同步。
    rank 为 0 表示名次未知（单条读取、页面直读），一律按「完整」对待——
    宁可多给信息，也不要凭空把一条记忆显示成模糊的。
    """
    if rank <= 0 or total <= 0:
        return LEVEL_FULL
    ratio = rank / total
    if ratio <= DEGRADE_BRIEF_RATIO:
        return LEVEL_FULL
    if ratio <= DEGRADE_KEYWORD_RATIO:
        return LEVEL_BRIEF
    return LEVEL_KEYWORDS


def is_tail(rank: int, total: int) -> bool:
    """名次是否落在榜尾（遗忘候选）。"""
    if rank <= 0 or total < _FORGET_TAIL_MIN:
        return False
    return (total - rank + 1) <= max(int(total * _FORGET_TAIL_RATIO), 1)


# 用户通道/消息通道的打分：关键词相关性为主，时间近因与用户匹配为辅。
# 强度不参与——它已退役，榜序只在自动插入时起作用。
_W_KEYWORD = 0.70
_W_RECENCY = 0.15
_W_USER = 0.15

# 用户通道：偏重重要性，但仍让话题相关的记忆优先浮出
_WU_KEYWORD = 0.55
_WU_IMPORTANCE = 0.45

# 判定「同一条记忆」的关键词重合度
_DUPLICATE_OVERLAP = 0.6
# 判定「这就是被要求忘掉的那件事」的关键词重合度（Jaccard）
_SUPPRESS_OVERLAP = 0.6

_MAX_QUERY_TOKENS = 48


def _json_list(raw: str | None) -> list:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return value if isinstance(value, list) else []


def _content_fingerprint(content: str) -> str:
    """「被要求遗忘」的内容指纹。

    用**关键词集合**而不是原文做指纹：她再次听到同一件事时，模型几乎不可能
    逐字复述出上次那句话，但关键词会高度重合。宁可多压掉一点，也不能让
    主人明确要求忘掉的事从别的说法里复活。

    关键词取不到时退回原文规范化后的哈希。
    """
    text = (content or "").strip()
    if not text:
        return ""
    keywords = sorted({k.strip() for k in extract_keywords(text) if k and k.strip()})
    payload = "|".join(keywords) if keywords else "".join(text.split())
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


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
    recall_count: int = 0
    useful_count: int = 0
    cross_chat_count: int = 1
    manual: int = 0
    created_at: float = 0.0
    last_recalled_at: float | None = None
    last_decay_at: float = 0.0
    merged_from: list[int] = field(default_factory=list)
    curation_score: int = 0
    curation_count: int = 0
    last_curated_at: float = 0.0
    forget_requested: int = 0
    forget_requested_at: float = 0.0
    # 位置榜的轴：最后一次被想起（整理产出 / 被召回）的时刻
    promoted_at: float = 0.0
    # 名次与榜长：由查询填进来，不在表里。升/降级形态靠它们推导。
    rank: int = 0
    rank_total: int = 0

    @property
    def level(self) -> int:
        """当前清晰度等级（由名次推导，见 curation_level）。"""
        return curation_level(self.rank, self.rank_total)

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
            recall_count=int(row["recall_count"] or 0),
            useful_count=int(row["useful_count"] or 0),
            cross_chat_count=int(row["cross_chat_count"] or 1),
            manual=int(row["manual"] or 0),
            created_at=float(row["created_at"] or 0.0),
            last_recalled_at=row["last_recalled_at"],
            last_decay_at=float(row["last_decay_at"] or 0.0),
            merged_from=[int(i) for i in _json_list(row["merged_from"])],
            curation_score=int(row["curation_score"] or 0),
            curation_count=int(row["curation_count"] or 0),
            last_curated_at=float(row["last_curated_at"] or 0.0),
            forget_requested=int(row["forget_requested"] or 0),
            forget_requested_at=float(row["forget_requested_at"] or 0.0),
            promoted_at=float(row["promoted_at"] or 0.0),
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
        tier: str = TIER_LONG,
        manual: bool = False,
        merged_from: list[int] | None = None,
        created_at: float | None = None,
    ) -> int:
        """写入一条记忆，返回自增 id。

        manual=True 表示「她/用户明确要求记住」，直接落永久层，不等阈值晋升。
        新记忆一律**上浮到榜首**（promoted_at = now）：刚被写进来就是最鲜活的。
        传了旧 `created_at` 的（如导入历史）自然排在旧位置，不会顶上去。
        """
        now = float(created_at if created_at is not None else time.time())
        keyword_list = [k for k in (keywords or extract_keywords(content)) if k]
        if manual:
            tier = TIER_PERMANENT
        cursor = self._db.execute(
            """
            INSERT INTO memories
                (tier, content, source_quote, tags, keywords, chat_id, chat_label,
                 user_id, user_name, recall_count, useful_count,
                 cross_chat_count, manual, created_at, last_recalled_at,
                 last_decay_at, merged_from, promoted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 1, ?, ?, NULL, ?, ?, ?)
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
                1 if manual else 0,
                now,
                now,
                json.dumps(merged_from or [], ensure_ascii=False),
                now,
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
    ) -> tuple[int, bool]:
        """去重写入。

        若已存在高度相似的长期记忆，则**上浮**它（同一件事又被说了一次＝又被
        想起一次），必要时累加跨会话计数；否则新建。返回 (记忆 id, 是否新建)。
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
                ),
                True,
            )

        now = time.time()
        cross_delta = 1 if (chat_id and duplicate.chat_id and chat_id != duplicate.chat_id) else 0
        self._db.execute(
            """
            UPDATE memories
               SET promoted_at = ?,
                   cross_chat_count = cross_chat_count + ?
             WHERE id = ?
            """,
            (now, cross_delta, duplicate.id),
        )
        return duplicate.id, False

    def update(self, memory_id: int, **fields) -> None:
        """按字段名更新记忆，仅允许白名单字段。"""
        allowed = {
            "tier", "content", "source_quote", "tags", "keywords", "chat_id",
            "chat_label", "user_id", "user_name", "recall_count",
            "useful_count", "cross_chat_count", "manual", "last_recalled_at",
            "last_decay_at", "merged_from", "promoted_at", "curation_count",
            "last_curated_at", "forget_requested", "forget_requested_at",
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
        """召回计数 +1、刷新最近召回时间，并把它们**上浮到榜首**。

        被想起＝又被用了一次，榜序就该往前挪——这是位置榜唯一的「加权」来源。
        """
        if not memory_ids:
            return
        now = time.time()
        placeholders = ",".join("?" for _ in memory_ids)
        self._db.execute(
            f"""
            UPDATE memories
               SET recall_count = recall_count + 1,
                   last_recalled_at = ?,
                   promoted_at = ?
             WHERE id IN ({placeholders})
            """,
            (now, now, *memory_ids),
        )

    def request_forget(self, memory_id: int) -> bool:
        """「被明确要求忘记」：从永久层拉回长期层，并要求遗忘（§10.3）。

        不在这里物理删除：删除留给整理管线统一做，同时把内容指纹写进
        suppressed，免得下一次压缩又把同一件事重新写回来。
        """
        memory = self.get(memory_id)
        if memory is None:
            return False
        self.update(
            memory_id,
            tier=TIER_LONG,
            forget_requested=1,
            forget_requested_at=time.time(),
        )
        self.suppress(memory.content)
        return True

    def suppress(self, content: str) -> None:
        """记下「这条内容已被要求遗忘」，压缩写入前会先比对该表。"""
        text = (content or "").strip()
        if not text:
            return
        self._db.execute(
            """
            INSERT OR REPLACE INTO suppressed_memories (keyword_hash, content, created_at)
            VALUES (?, ?, ?)
            """,
            (_content_fingerprint(text), text, time.time()),
        )

    def is_suppressed(self, content: str) -> bool:
        """这条内容是否属于「已被要求遗忘」的范围。

        先走指纹（同一条信息被换了个说法重新压回来时，指纹可能不同），
        再退回关键词重合度兜底：宁可多压掉一点，也不能让主人明确要求忘掉的事
        从另一种说法里复活。被抑制的记录极少（只来自明确要求），全表比对可接受。
        """
        text = (content or "").strip()
        if not text:
            return False
        row = self._db.query_one(
            "SELECT 1 FROM suppressed_memories WHERE keyword_hash = ?",
            (_content_fingerprint(text),),
        )
        if row is not None:
            return True
        tokens = set(tokenize(text))
        if not tokens:
            return False
        for entry in self._db.query("SELECT content FROM suppressed_memories"):
            candidate = set(tokenize(str(entry["content"] or "")))
            if not candidate:
                continue
            union = tokens | candidate
            if union and len(tokens & candidate) / len(union) >= _SUPPRESS_OVERLAP:
                return True
        return False

    def search_by_keywords(self, query_text: str, limit: int = 5) -> list[Memory]:
        """按关键词匹配记忆，按重合度排序（不设阈值、不限层级）。

        与 `recall()` 的区别：召回是给注入用的、要过滤掉「已被要求忘记」的，
        而这个方法是给「遗忘」这类需要精确定位的场景用的——正因为如此，
        它必须能看见长期层与永久层里的全部记忆。代价是一次全表扫描，
        但调用频率极低（只在有人明确要求忘掉某事时）。
        """
        tokens = set(tokenize(query_text or ""))
        if not tokens:
            return []
        scored: list[tuple[Memory, float]] = []
        for row in self._db.query("SELECT * FROM memories WHERE forget_requested = 0"):
            memory = Memory.from_row(row)
            score = overlap_score(tokens, memory.keywords)
            if score > 0:
                scored.append((memory, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [memory for memory, _ in scored[: max(int(limit), 1)]]

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
        return self.ranked(tier, limit=int(limit))

    def ranked(self, tier: str = TIER_LONG, limit: int | None = None) -> list[Memory]:
        """按**位置榜**读某一层，并把名次与榜长填进每条记忆。

        名次是 `Memory.level` 的唯一输入，所以凡是要用等级的地方都得走这里
        （页面、降级、遗忘都算）。
        """
        sql = f"SELECT * FROM memories WHERE tier = ? ORDER BY {_BOARD_ORDER}"
        params: tuple = (tier,)
        if limit is not None:
            sql += " LIMIT ?"
            params = (tier, max(int(limit), 1))
        rows = self._db.query(sql, params)
        total = self._board_size(tier)
        memories: list[Memory] = []
        for index, row in enumerate(rows):
            memory = Memory.from_row(row)
            memory.rank = index + 1
            memory.rank_total = total
            memories.append(memory)
        return memories

    def board(self, tier: str = TIER_LONG) -> list[Memory]:
        """整张榜（名次已填好）。降级与遗忘需要看到全榜才能判断「尾部」。"""
        return self.ranked(tier)

    def auto_injection(self) -> tuple[list[Memory], list[Memory]]:
        """自动插入用的两段：榜前 N 条给**正文**，其后 M 条只给**关键词**。

        第二段是「元记忆」：她得知道自己还忘了些什么，才会去主动回想。
        只给一行关键词，代价极小；不给的话第 N+1 条之后就彻底不在她视野里，
        她永远不会去 `recall` 一个不知道存在的东西。

        返回 (正文条目, 索引条目)，两段的形态由 injector 决定。
        被明确要求忘记的不进任何一段。
        """
        top_n = max(int(self._config.auto_top), 0)
        index_n = max(int(self._config.auto_index), 0)
        if top_n <= 0 and index_n <= 0:
            return [], []
        rows = self.ranked(TIER_LONG, limit=max(top_n + index_n, 1))
        total = rows[0].rank_total if rows else 0
        live = [m for m in rows if not m.forget_requested]
        for memory in live:
            memory.rank_total = total
        return live[:top_n], live[top_n : top_n + index_n]

    def permanents(self) -> list[Memory]:
        """永久记忆**全量**读出。

        永久层不参与榜单、不降级、不遗忘，所以她确信的事永远在场，
        插入时排在最早（长期背景的最前一段）。
        """
        rows = self._db.query(
            "SELECT * FROM memories WHERE tier = ? ORDER BY id ASC",
            (TIER_PERMANENT,),
        )
        memories = [Memory.from_row(row) for row in rows]
        for index, memory in enumerate(memories):
            memory.rank = index + 1
            memory.rank_total = len(memories)
        return memories

    def count_by_tier(self) -> dict[str, int]:
        rows = self._db.query("SELECT tier, COUNT(*) AS n FROM memories GROUP BY tier")
        return {str(r["tier"]): int(r["n"]) for r in rows}

    def _board_size(self, tier: str) -> int:
        row = self._db.query_one(
            "SELECT COUNT(*) AS n FROM memories WHERE tier = ?", (tier,)
        )
        return int(row["n"]) if row else 0

    def list_page(
        self, tier: str, limit: int, offset: int
    ) -> tuple[list[Memory], int]:
        """分页读某一层记忆（按位置榜倒序），返回 (本页记忆, 该层总数)。

        供记忆页面展示：榜就是她脑子的排序，页面和上下文必须同源——
        上下文中看到的顺序，在页面上逐条对得上。
        """
        limit = max(int(limit), 1)
        offset = max(int(offset), 0)
        total = self._board_size(tier)
        rows = self._db.query(
            f"""
            SELECT * FROM memories
             WHERE tier = ?
             ORDER BY {_BOARD_ORDER}
             LIMIT ? OFFSET ?
            """,
            (tier, limit, offset),
        )
        memories: list[Memory] = []
        for index, row in enumerate(rows):
            memory = Memory.from_row(row)
            memory.rank = offset + index + 1
            memory.rank_total = total
            memories.append(memory)
        return memories, total

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
        """某个用户的记忆，按「话题相关 + 时间近因」排序，无阈值兜底。"""
        rows = self._db.query(
            f"SELECT * FROM memories WHERE user_id = ? AND forget_requested = 0 "
            f"ORDER BY {_BOARD_ORDER} LIMIT ?",
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
        """借倒排表先取候选，避免全表扫描；命中太少时用永久记忆兜底。"""
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
            # 兜底：把永久记忆并入候选，缓解关键词漏召。
            # 长期记忆不兜底——榜上靠前的那些本来就在上下文里，重复召回只是浪费。
            fallback = self._db.query(
                f"""
                SELECT id FROM memories
                 WHERE tier = ?
                 ORDER BY {_BOARD_ORDER}
                 LIMIT ?
                """,
                (TIER_PERMANENT, int(limit)),
            )
            for row in fallback:
                candidate_id = int(row["id"])
                if candidate_id not in memory_ids:
                    memory_ids.append(candidate_id)

        # 被明确要求忘记的记忆立即停止召回（§10.3）：否则她会在「已要求忘记」
        # 到「下一轮整理真正删除」之间的窗口里，又把这件事说出来。
        return [
            memory
            for memory in self.get_many(memory_ids)
            if not memory.forget_requested
        ]

    def _score(
        self, memory: Memory, query_tokens: set[str], now: float, user_id: str | None
    ) -> float:
        keyword_score = overlap_score(query_tokens, memory.keywords)
        recency = self._recency(memory, now)
        user_factor = 1.0 if (user_id and memory.user_id == user_id) else 0.0
        return (
            _W_KEYWORD * keyword_score
            + _W_RECENCY * recency
            + _W_USER * user_factor
        )

    def _recency(self, memory: Memory, now: float) -> float:
        """时间近因：越久没被用到，时间分越低。永久记忆不衰减。"""
        if memory.tier == TIER_PERMANENT:
            return 1.0
        half_life = max(self._config.decay_half_life_days, 1.0)
        idle_days = max(now - memory.activity_at(), 0.0) / 86400.0
        return 0.5 ** (idle_days / half_life)

    def _importance(self, memory: Memory, now: float) -> float:
        """重要度 = 被召回的次数 + 时间近因。

        强度已退役，所以「重要」只剩两个可观测证据：被想起过多少次、
        最近一次是什么时候。前者是累计证据，后者防止久不用的旧记忆靠老本一直压着。
        """
        recall = min(memory.recall_count / 5.0, 1.0)
        return 0.6 * recall + 0.4 * self._recency(memory, now)

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
