"""长期 / 永久记忆的存储、召回打分与去重。

三层语义：
- long      : 长期记忆，排在一张**位置榜**上，尾部会变模糊、最终被遗忘
- permanent : 永久记忆，不参与榜序、不降级、不遗忘

## 位置榜取代权重

长期记忆不使用「强度」权重，仅依据**最后一次被想起的时刻**（`promoted_at`）
排序。新提炼的记忆按它覆盖的对话时间入榜，当天摘要、合并结果与每次主动召回
落在最前；使用较少的记忆自然下沉，进入榜尾后逐级模糊，长期未被召回的才会被遗忘。

选择该方案的原因：排序结果**确定**，可直接反映记忆强度；
而基于权重衰减的方案依赖较多参数与半衰期，难以调整与解释。

## 三条召回路径的依据不同

- **自动插入**（拼上下文）：按榜序取前 N 条，与话题无关。
- **主脑自动相关召回**：按当前消息做关键词匹配，只补榜前没覆盖的部分；
  不计数、不改变榜序。
- **工具召回**（`recall`，表现为「我想想」）：按**关键词相关性**排序，
  且会更新召回次数与榜序。若也按榜序，会召回「榜上靠前但与当前话题无关」的记忆。

注入时另有反过拟合指令约束（见 core/memory/injector.py）。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field

from ..common.db import MemoryDB
from ..common.timeexpr import day_key
from ..common.tokenizer import extract_keywords, overlap_score, tokenize

TIER_LONG = "long"
TIER_PERMANENT = "permanent"

# ── 位置榜（§3）──────────────────────────────────────────
# 榜序：谁最后被想起谁排前面。id 兜底保证同刻写入的顺序稳定。
_BOARD_ORDER = "promoted_at DESC, id DESC"

# 等级：0 完整 / 1 精简（一句）/ 2 只出现在模糊索引里（正文保留）
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
    rank 为 0 表示名次未知（单条读取、页面直读），按「完整」处理：
    优先提供完整信息，避免将记忆错误显示为模糊状态。
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


def _days_of(hint) -> list[str]:
    """把时间区间展开成一组本地日期键（`YYYY-MM-DD`）。

    区间太宽时只取**最后** `_MAX_TIME_DAYS` 天：用户说「前阵子」时，他更可能
    在说最近那几天，一次捞出三十天的记忆只会把上下文灌满。
    """
    start = float(getattr(hint, "start", 0.0) or 0.0)
    end = float(getattr(hint, "end", 0.0) or 0.0)
    if start <= 0 or end <= start:
        return []
    days: list[str] = []
    cursor = start
    # 用 86400 步进而不是日历运算：区间边界本来就是按自然日算好的，
    # 这里只需要覆盖 [start, end) 里出现的每一个日期键。
    while cursor < end and len(days) < _MAX_TIME_DAYS * 2:
        key = day_key(cursor)
        if key and key not in days:
            days.append(key)
        cursor += 86400.0
    tail_key = day_key(end - 1.0)
    if tail_key and tail_key not in days:
        days.append(tail_key)
    return days[-_MAX_TIME_DAYS:]


# ── 联想打分（取代旧的关键词+近因加权）───────────────��
# 旧公式是三个权重的加权和：0.70×关键词 + 0.15×近因 + 0.15×同人。
# 问题在于「加权和」让三个维度可以互相补偿——一条完全无关但最近的记忆，
# 靠近因也能拿到 0.15 分混进结果。改成**连乘**之后，任何一维接近 0，
# 整条就起不来，这才是「相关」应有的语义。
#
# 公式：score = 相关度 × (0.5 + 0.5×强度) × 时间衰减
#   - 相关度：关键词重合率（0~1），为 0 时整条必然为 0
#   - 强度：她在这件事上的态度有多强烈（0~1）。没有态度时按 0 算，
#     系数落在 0.5——「我说不上来我喜欢不喜欢，但它就是发生过」
#   - 时间衰减：久远的事慢慢淡，但留一个地板，不让它彻底归零
_INTENSITY_WEIGHT = 0.5
_RECENCY_FLOOR = 0.35
# 时间通道的相关度基准：用户明确说了「昨天」，那批记忆就一定相关
_TIME_RELEVANCE = 0.7
# 时间通道最多覆盖多少天（区间太宽时按最近的几天取，避免一次捞出上百条）
_MAX_TIME_DAYS = 7
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

    使用**关键词集合**而非原文作为指纹：同一件事再次出现时，模型几乎无法
    逐字复述上次的句子，但关键词会高度重合。该策略有意偏向多匹配，以确保
    用户明确要求忘记的内容不会以其它表述重新出现。

    关键词为空时退回到原文规范化后的哈希。
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
    # 「判断为没有信息价值」的标记：整理时只打标记，下一次整理才删除。
    # 期间只要被召回、被重新提到或被手工编辑，标记立即解除。
    worthless: int = 0
    worthless_at: float = 0.0
    worthless_checked_at: float = 0.0
    # 位置榜的轴：最后一次被想起（整理产出 / 被召回）的时刻
    promoted_at: float = 0.0
    # 这条记忆覆盖的是哪一天（本地日期 YYYY-MM-DD）。
    # 与 promoted_at 的区别：promoted_at 会被召回推到现在，covered_date 永远
    # 指向「这段话是什么时候说的」。问「昨天」时按它查才查得准。
    covered_date: str = ""
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
            worthless=int(row["worthless"] or 0),
            worthless_at=float(row["worthless_at"] or 0.0),
            worthless_checked_at=float(row["worthless_checked_at"] or 0.0),
            promoted_at=float(row["promoted_at"] or 0.0),
            covered_date=str(row["covered_date"] or ""),
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
        promoted_at: float | None = None,
        covered_at: float | None = None,
    ) -> int:
        """写入一条记忆，返回自增 id。

        manual=True 表示「用户或插件明确要求记住」，直接写入永久层，不经过阈值晋升。

        `created_at` 是这条记忆**产生的时间**（导入历史时保留原时间）；
        `promoted_at` 是**入榜位置**，默认等于 `created_at`——新写入即最新。
        整理时提炼出的记忆会把入榜位置设为「这段发言覆盖的末尾时间」，
        这样每晚整理不会把所有新条目一起压到榜首、把旧榜整版刷下去。
        只有当天摘要与合并结果才显式落在当下。

        `covered_at` 是这段记忆**说的是哪段时间**的事。它与 `promoted_at`
        分成两个字段是必要的：被召回会把 `promoted_at` 推到现在，而「昨天说过
        什么」问的是 `covered_date`。缺省时按 `promoted_at` 归属日期。
        """
        now = time.time()
        created = float(created_at if created_at is not None else now)
        promoted = float(promoted_at if promoted_at is not None else created)
        covered = day_key(float(covered_at) if covered_at else promoted)
        keyword_list = [k for k in (keywords or extract_keywords(content)) if k]
        if manual:
            tier = TIER_PERMANENT
        cursor = self._db.execute(
            """
            INSERT INTO memories
                (tier, content, source_quote, tags, keywords, chat_id, chat_label,
                 user_id, user_name, recall_count, useful_count,
                 cross_chat_count, manual, created_at, last_recalled_at,
                 last_decay_at, merged_from, promoted_at, covered_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 1, ?, ?, NULL, ?, ?, ?, ?)
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
                created,
                created,
                json.dumps(merged_from or [], ensure_ascii=False),
                promoted,
                covered,
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
        promoted_at: float | None = None,
        covered_at: float | None = None,
    ) -> tuple[int, bool]:
        """去重写入。

        若已存在高度相似的长期记忆，则**上浮**它（同一件事又被说了一次＝又被
        想起一次），必要时累加跨会话计数；否则新建。返回 (记忆 id, 是否新建)。

        新建时可用 `promoted_at` 指定入榜位置：整理提炼出的记忆按发言覆盖的
        时间落位，而不是把整批新条目一起压到榜首。`covered_at` 是这条记忆说的
        是哪段时间的事，用于「问昨天」时的时间路检索。
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
                    promoted_at=promoted_at,
                    covered_at=covered_at,
                ),
                True,
            )

        now = time.time()
        cross_delta = 1 if (chat_id and duplicate.chat_id and chat_id != duplicate.chat_id) else 0
        self._db.execute(
            """
            UPDATE memories
               SET promoted_at = ?,
                   cross_chat_count = cross_chat_count + ?,
                   worthless = 0,
                   worthless_at = 0
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
            "worthless", "worthless_at", "worthless_checked_at", "covered_date",
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

    def update_content(self, memory_id: int, content: str) -> bool:
        """手工改写一条记忆的正文，并重建它的关键词索引。

        关键词是从正文推导出来的，正文一改就必须重算：否则 `recall` 仍按
        旧关键词命中，页面上的词也和新内容对不上。标签与来源原样保留——
        手工编辑针对的是「内容写错了」，不是重新归类。
        """
        text = (content or "").strip()
        if not text:
            return False
        keywords = [k for k in extract_keywords(text) if k]
        self.update(memory_id, content=text, keywords=keywords)
        # 人工改写说明这条记忆仍然重要：解除「无价值」标记，别让整理把它删掉
        self.clear_worthless([memory_id])
        return True

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
                   promoted_at = ?,
                   worthless = 0,
                   worthless_at = 0,
                   worthless_checked_at = ?
             WHERE id IN ({placeholders})
            """,
            (now, now, now, *memory_ids),
        )

    def worthless_candidates(self, limit: int = 20) -> list[Memory]:
        """取出这一轮要判断「还有没有信息价值」的长期记忆。

        永久层不参与：它不降级、不遗忘，也不该因为「判断无价值」被删掉。
        按 `worthless_checked_at` 升序轮转——判断过但仍有价值的条目会被推到
        队尾，否则每晚都会拿同一批旧记忆重复问模型。
        """
        rows = self._db.query(
            """
            SELECT * FROM memories
             WHERE tier = ? AND forget_requested = 0 AND worthless = 0
             ORDER BY worthless_checked_at ASC, promoted_at ASC
             LIMIT ?
            """,
            (TIER_LONG, max(int(limit), 1)),
        )
        return [Memory.from_row(row) for row in rows]

    def mark_worthless(self, memory_ids: list[int]) -> int:
        """打上「无价值」标记。**不删除**：留给下一次整理执行。"""
        if not memory_ids:
            return 0
        now = time.time()
        placeholders = ",".join("?" for _ in memory_ids)
        cursor = self._db.execute(
            f"""
            UPDATE memories
               SET worthless = 1,
                   worthless_at = ?,
                   worthless_checked_at = ?
             WHERE id IN ({placeholders})
            """,
            (now, now, *memory_ids),
        )
        return int(cursor.rowcount or 0)

    def mark_worthless_checked(self, memory_ids: list[int]) -> None:
        """记录「已经判断过、结论是仍有价值」，让取材轮转到后面的条目。"""
        if not memory_ids:
            return
        placeholders = ",".join("?" for _ in memory_ids)
        self._db.execute(
            f"UPDATE memories SET worthless_checked_at = ? WHERE id IN ({placeholders})",
            (time.time(), *memory_ids),
        )

    def clear_worthless(self, memory_ids: list[int]) -> int:
        """解除「无价值」标记（被召回 / 被重新提到 / 被手工编辑时调用）。"""
        if not memory_ids:
            return 0
        placeholders = ",".join("?" for _ in memory_ids)
        cursor = self._db.execute(
            f"""
            UPDATE memories
               SET worthless = 0, worthless_at = 0, worthless_checked_at = ?
             WHERE id IN ({placeholders})
            """,
            (time.time(), *memory_ids),
        )
        return int(cursor.rowcount or 0)

    def delete_worthless(self, before_ts: float, limit: int = 50) -> int:
        """删除标记已过期（`worthless_at <= before_ts`）的无价值记忆。

        只删长期层：万一标记之后被晋升成永久记忆，就不该再按这条规则消失。
        """
        rows = self._db.query(
            """
            SELECT id FROM memories
             WHERE tier = ? AND worthless = 1
               AND worthless_at > 0 AND worthless_at <= ?
             ORDER BY worthless_at ASC
             LIMIT ?
            """,
            (TIER_LONG, float(before_ts), max(int(limit), 1)),
        )
        ids = [int(row["id"]) for row in rows]
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        with self._db.lock:
            self._db.execute(
                f"DELETE FROM memory_keywords WHERE memory_id IN ({placeholders})",
                tuple(ids),
            )
            self._db.execute(
                f"DELETE FROM memories WHERE id IN ({placeholders})", tuple(ids)
            )
        return len(ids)

    def request_forget(self, memory_id: int) -> bool:
        """「被明确要求忘记」：从永久层拉回长期层，并要求遗忘（§10.3）。

        此处不执行物理删除：删除由整理管线统一处理，同时将内容指纹写入
        suppressed，避免下一次压缩重新产出同一条记忆。
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

        先按指纹匹配（同一信息换用不同表述重新压缩时，指纹可能不同），
        再以关键词重合度兜底：优先扩大匹配范围，确保用户明确要求忘记的内容
        不会通过其他表述重新进入记忆。被抑制的记录数量很少（仅来自明确要求），
        全表比对可接受。
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

    def memories_of_user(self, user_id: str, limit: int = 20) -> list[Memory]:
        """某个人的长期与永久记忆，供画像整理取材。

        永久层排前面：它们是最确定的认知。其余按位置榜倒序，保证「最近还在
        用的」优先进入画像素材。已要求遗忘的条目不参与。
        """
        if not user_id:
            return []
        rows = self._db.query(
            f"""
            SELECT * FROM memories
             WHERE user_id = ? AND forget_requested = 0
             ORDER BY CASE tier WHEN '{TIER_PERMANENT}' THEN 0 ELSE 1 END,
                      {_BOARD_ORDER}
             LIMIT ?
            """,
            (user_id, max(int(limit), 1)),
        )
        return [Memory.from_row(row) for row in rows]

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
        """整张榜（名次已填好）。降级与遗忘需要全榜数据才能判断「尾部」。"""
        return self.ranked(tier)

    def auto_injection(self) -> tuple[list[Memory], list[Memory]]:
        """自动插入用的两段：榜前 N 条给**正文**，其后 M 条只给**关键词**。

        第二段是「元记忆」：模型需要知道自己还忘记了哪些内容，才会主动回想。
        该段只给一行关键词，代价极小；若省略，第 N+1 条之后的内容将完全不在
        上下文中，也不会被 `recall` 检索到。

        返回 (正文条目, 索引条目)，两段的具体形态由 injector 决定。
        被明确要求忘记的条目不出现在任何一段。
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

        永久层不参与榜序、不降级、不遗忘，因此其内容始终在场，
        插入时排在最前（作为长期背景的第一段）。
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

        供记忆页面展示：位置榜即上下文中的记忆顺序，页面与上下文必须同源，
        保证页面上的条目顺序与上下文中的顺序逐条对应。
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
        time_hint=None,
        stances: list | None = None,
    ) -> list[RecalledMemory]:
        """多通道联想：关键词 + 时间区间 + 发送者，合并去重后返回。

        与旧版「召回」的区别不只是多了通道，而是**打分换了语义**：旧版是
        三个权重的加权和，条目的分数可以靠「最近」或「同一个人」补上来；
        新版用连乘，相关度为零就是零。这样联想出来的东西才会「都是有点关系的」，
        而不是「凑够三条」。

        `time_hint` 是 `timeexpr.TimeHint`：用户说了「昨天」「上周三」时，
        插件已经把它算成区间，这里按 `covered_date` 直接取那几天的记忆。
        时间不能当关键词——日期串在 2-gram 分词里会被切碎，倒排表里也没有它。

        `stances` 是 `Stance` 列表（可为空）：她对这些事的态度强度会抬高
        相关记忆的分数。有立场的事更容易被想起来，这是联想与人味的一部分。
        """
        top_k = self._config.recall_top_k if top_k is None else top_k
        user_top_k = self._config.recall_user_top_k if user_top_k is None else user_top_k
        min_score = self._config.recall_min_score if min_score is None else min_score
        exclude_ids = exclude_ids or set()
        stances = stances or []

        query_tokens = set(tokenize(query_text)[:_MAX_QUERY_TOKENS])

        results: list[RecalledMemory] = []
        seen: set[int] = set()

        # 通道一：与当前消息相关的记忆
        message_hits = 0
        for memory, score in self._rank_by_keywords(
            query_tokens, limit=max(top_k * 6, 24), stances=stances
        ):
            if message_hits >= top_k:
                break
            if memory.id in exclude_ids or memory.id in seen or score < min_score:
                continue
            seen.add(memory.id)
            message_hits += 1
            results.append(RecalledMemory(memory=memory, score=score, channel="message"))

        # 通道二：用户提到了具体时间，把那几天的记忆直接取回来
        if time_hint is not None:
            for memory, score in self._rank_by_time(time_hint, limit=top_k):
                if memory.id in exclude_ids or memory.id in seen:
                    continue
                seen.add(memory.id)
                results.append(RecalledMemory(memory=memory, score=score, channel="time"))

        # 通道三：这个人的相关记忆（无关键词下限，按重要性兜底）
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
        self, query_tokens: set[str], limit: int, *, stances: list | None = None
    ) -> list[tuple[Memory, float]]:
        """用倒排表筛出候选，再按联想分排序。"""
        now = time.time()
        candidates = self._candidates_by_keywords(query_tokens, limit)
        if not candidates:
            return []
        stances = stances or []
        scored = [
            (memory, self._score(memory, query_tokens, now, stances))
            for memory in candidates
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def _rank_by_time(self, hint, limit: int) -> list[tuple[Memory, float]]:
        """按时间区间取记忆：用户说了「昨天」，就把昨天的记忆找出来。

        时间区间只覆盖 `covered_date`（那段话是哪天说的），不看 `promoted_at`
        ——后者会被召回推到现在，用它查「昨天」会查出今天刚想起的旧事。
        """
        days = _days_of(hint)
        if not days:
            return []
        placeholders = ",".join("?" for _ in days)
        rows = self._db.query(
            f"""
            SELECT * FROM memories
             WHERE covered_date IN ({placeholders}) AND forget_requested = 0
             ORDER BY {_BOARD_ORDER}
             LIMIT ?
            """,
            (*days, max(int(limit) * 4, 16)),
        )
        now = time.time()
        scored: list[tuple[Memory, float]] = []
        for row in rows:
            memory = Memory.from_row(row)
            # 时间通道的相关度是给死的：用户点了名，这批记忆就算相关。
            # 分数仍然乘时间衰减，所以区间里更靠后的日子会排在前面。
            score = _TIME_RELEVANCE * self._recency_factor(memory, now)
            scored.append((memory, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: int(limit)]

    def _rank_user_memories(
        self, user_id: str, query_tokens: set[str], limit: int
    ) -> list[tuple[Memory, float]]:
        """某个用户的记忆，按「话题相关 + 重要度」排序，无阈值兜底。"""
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

        # 被明确要求忘记的记忆立即停止召回（§10.3）：避免在「已要求忘记」
        # 到「下一轮整理真正删除」之间再次被提及。
        return [
            memory
            for memory in self.get_many(memory_ids)
            if not memory.forget_requested
        ]

    def _score(
        self,
        memory: Memory,
        query_tokens: set[str],
        now: float,
        stances: list | None = None,
    ) -> float:
        """联想分：相关度 × 态度强度 × 时间衰减，三项连乘。

        连乘而不是加权和是有意的：加权和允许「完全不相关但很新」的记忆拿到分数，
        只要把近因权重调大就能混进结果。连乘下相关度为 0 就整条为 0，联想出来
        的东西至少都沾点边。

        态度强度那一项从 0.5 起步而不是 0：没有态度不代表没关系，
        「说不上来喜不喜欢，但它就是发生过」也该被想起。
        """
        relevance = overlap_score(query_tokens, memory.keywords)
        if relevance <= 0:
            return 0.0
        intensity = self._stance_intensity(memory, stances or [])
        return relevance * (_INTENSITY_WEIGHT + (1.0 - _INTENSITY_WEIGHT) * intensity) * (
            self._recency_factor(memory, now)
        )

    def _stance_intensity(self, memory: Memory, stances: list) -> float:
        """这条记忆关联着多强的态度（0~1）。

        只在同一个人 / 同一批关键词上找关联，不做语义匹配：零依赖的分词扛不住
        语义，硬凑出来的关联比没有更糟。找不到就返回 0，让分数只由相关度决定。

        `user` 按 QQ 号比，`topic` 按关键词比：话题态度用的是「爬山」「相亲」
        这类词，而记忆的 keywords 正是同一套词法切出来的，两边能对上。
        """
        if not stances:
            return 0.0
        best = 0.0
        keywords = memory.keywords or []
        for stance in stances:
            key = stance.subject_key
            if not key:
                continue
            if stance.subject_type == "user" and memory.user_id:
                if key == memory.user_id:
                    best = max(best, stance.effective_intensity())
            elif stance.subject_type == "topic":
                if key in keywords or key in memory.content:
                    best = max(best, stance.effective_intensity())
        return min(best, 1.0)

    def _recency_factor(self, memory: Memory, now: float) -> float:
        """时间衰减：久远的事慢慢淡，但留一个地板不让它彻底归零。

        有地板是因为「旧事」本身就有被想起的价值：一段三个月前的对话当然比
        昨天的更模糊，但用户主动提起它时，它就该能被联动起来。
        """
        return _RECENCY_FLOOR + (1.0 - _RECENCY_FLOOR) * self._recency(memory, now)

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
