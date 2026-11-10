"""态度：她对人对事的看法、感受与关系。

记忆回答「发生了什么」，态度回答「我怎么想」。两者生命周期完全不同：

- 记忆会**遗忘**：沉到榜尾、被模糊、最终删掉。
- 态度只会**改写**：新证据来了就重写同一行，永远不删。把「我觉得他其实挺
  怕麻烦的」压缩成关键词碎片，或者按名次把它淡掉，都是不合适的——人对别人
  的看法不会因为「最近没想起来」就消失。

因此态度单独一张表，与 memories 不共享任何排序或清理机制。

## 三个字段的语义（不要混用）

- `stance`  正文。必须第一人称、口语、带具体依据，例如「嘴上说随便，真让他
  选又挑半天，我觉得他是怕担责任」。写成客观陈述（「小明怕担责任」）就退化
  成了记忆，失去「她觉得」这层。
- `intensity` 0~1，这份态度有多**强烈**。「讨厌加班」和「烦死了」不是一回事。
- `confidence` 0~1，她有多**确信自己没看错**。这个值不由模型自由发挥：
  `apply()` 会按证据条数加一个上限（见 `_confidence_cap`），一次情绪冲不上
  高置信度，避免「说了一次就当成定论」。

## 衰减只发生在一处

`effective_intensity()` 是注入与排序时用的强度：久没被重新证实就缓慢回落，
但**有下限**（`_INTENSITY_FLOOR`）。落库的 `intensity` 不动，回落是算出来的，
所以重新想起时它自己就回来了。
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field

from ..common.db import MemoryDB

# 态度的对象
SUBJECT_USER = "user"      # 某个人的 QQ 号
SUBJECT_TOPIC = "topic"    # 一件事 / 一个话题，用关键词做键
SUBJECT_SELF = "self"      # 她自己的处境（「我最近睡得不太好」）

# 同一对象可有多个维度
DIM_VIEW = "view"          # 看法：我觉得他是怎样的人 / 这件事怎么样
DIM_RELATION = "relation"  # 关系：我跟他相处起来是什么感觉

_SUBJECT_TYPES = {SUBJECT_USER, SUBJECT_TOPIC, SUBJECT_SELF}
_DIMENSIONS = {DIM_VIEW, DIM_RELATION}

# 正文长度上限：态度是一句话的立场，不是一段描述
_MAX_STANCE_CHARS = 120
# 保留的依据条数：太多会把一行撑爆，太少又看不出「她为什么这么想」
_MAX_BASIS = 5
_MAX_BASIS_QUOTE = 80

# 强度回落：半衰期 45 天，但不会低于地板值——淡掉的是鲜明度，不是「她还记不记得」
_INTENSITY_HALF_LIFE_DAYS = 45.0
_INTENSITY_FLOOR = 0.15

# 新证据与旧态度融合时的权重：偏新，但不让一次情绪掀翻既有判断
_BLEND_NEW = 0.6

# 置信度上限：0.35 + 0.12×证据数。一次证据封顶 0.47，五次才到 0.95。
# 「一次情绪不算立场」这条规则就落在这里，不要改成模型直接说了算。
_CONFIDENCE_BASE = 0.35
_CONFIDENCE_PER_EVIDENCE = 0.12


def _confidence_cap(evidence_count: int) -> float:
    return min(_CONFIDENCE_BASE + _CONFIDENCE_PER_EVIDENCE * max(evidence_count, 1), 0.95)


def _json_list(raw: str | None) -> list:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return value if isinstance(value, list) else []


def _clamp(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def normalize_key(key: str) -> str:
    """对象键规范化：去掉多余空白，避免「小明 」和「小明」各占一行。"""
    return " ".join(str(key or "").split())


@dataclass
class Stance:
    """一条态度。"""

    id: int = 0
    subject_type: str = SUBJECT_TOPIC
    subject_key: str = ""
    subject_name: str = ""
    dimension: str = DIM_VIEW
    stance: str = ""
    basis: list[dict] = field(default_factory=list)
    intensity: float = 0.5
    confidence: float = 0.4
    evidence_count: int = 1
    revision: int = 1
    first_at: float = 0.0
    updated_at: float = 0.0

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Stance":
        return cls(
            id=int(row["id"]),
            subject_type=str(row["subject_type"] or SUBJECT_TOPIC),
            subject_key=str(row["subject_key"] or ""),
            subject_name=str(row["subject_name"] or ""),
            dimension=str(row["dimension"] or DIM_VIEW),
            stance=str(row["stance"] or ""),
            basis=[b for b in _json_list(row["basis"]) if isinstance(b, dict)],
            intensity=_clamp(row["intensity"] or 0.0),
            confidence=_clamp(row["confidence"] or 0.0),
            evidence_count=int(row["evidence_count"] or 1),
            revision=int(row["revision"] or 1),
            first_at=float(row["first_at"] or 0.0),
            updated_at=float(row["updated_at"] or 0.0),
        )

    @property
    def label(self) -> str:
        """给人看的主语：优先用记住的名字，没有就用键本身。"""
        return self.subject_name or self.subject_key

    def effective_intensity(self, now: float | None = None) -> float:
        """当前的鲜明度：久未证实就缓慢回落，但不低于地板。

        落库的 `intensity` 是「最近一次她这么想时的强度」，这个函数才是注入与
        排序该用的值。两者的差别就是「她上一次这么说」与「她现在还这么觉得吗」。
        """
        now = float(now if now is not None else time.time())
        idle_days = max(now - (self.updated_at or self.first_at or now), 0.0) / 86400.0
        decay = 0.5 ** (idle_days / _INTENSITY_HALF_LIFE_DAYS)
        return _INTENSITY_FLOOR + (self.intensity - _INTENSITY_FLOOR) * decay


class StanceStore:
    """态度表的读写。

    与 `MemoryStore` 并列，共用同一个库与锁。业务层不直接拼 SQL。
    """

    def __init__(self, db: MemoryDB, config=None):
        self._db = db
        self._config = config

    def update_config(self, config) -> None:
        self._config = config

    # ── 写入 ──────────────────────────────────────────────

    def get(
        self,
        subject_type: str,
        subject_key: str,
        dimension: str = DIM_VIEW,
    ) -> Stance | None:
        row = self._db.query_one(
            """
            SELECT * FROM stances
             WHERE subject_type = ? AND subject_key = ? AND dimension = ?
            """,
            (subject_type, normalize_key(subject_key), dimension),
        )
        return Stance.from_row(row) if row else None

    def apply(self, items: list[dict], *, name_to_id: dict[str, str] | None = None) -> dict:
        """写入一批从压缩里提炼出的态度。

        `items` 的形态与 `extractor.condense()` 的 `stances` 段一致：
        `{"subject_type","subject","stance","basis_quote","intensity","confidence"}`。

        三条落库规则：

        1. **同对象同维度只有一行**，新证据来就改写它（`revision + 1`），
           不新增。所以「她对这个人的看法」永远只有一个当前版本。
        2. **强度按 6:4 融合**，不直接覆盖：一次激动的发言不该把长期的判断整个
           翻过去，但它确实会推高或压低这个值。
        3. **置信度受证据条数封顶**（`_confidence_cap`）：模型给多少都不作数，
           单条证据到不了 0.5 以上。
        """
        name_to_id = name_to_id or {}
        now = time.time()
        created = 0
        updated = 0
        for item in items:
            subject_type = str(item.get("subject_type") or "").strip()
            if subject_type not in _SUBJECT_TYPES:
                # 模型没给类型时按对象猜：QQ 号 → user，其余 → topic。
                # 猜错不如不写：类型错了会污染「对这个人的态度」这条查询。
                continue
            dimension = str(item.get("dimension") or DIM_VIEW).strip()
            if dimension not in _DIMENSIONS:
                dimension = DIM_VIEW
            raw_key = str(item.get("subject") or "").strip()
            key = normalize_key(raw_key)
            if not key:
                continue
            text = " ".join(str(item.get("stance") or "").split())
            if not text:
                continue
            text = text[:_MAX_STANCE_CHARS]

            if subject_type == SUBJECT_USER:
                # 名字 → QQ 号。映射不到的人不写：按名字建行会让同一个人
                # 在不同会话里长出几条互不相干的态度。
                key = normalize_key(name_to_id.get(raw_key, "")) or (
                    key if key.isdigit() else ""
                )
                if not key:
                    continue
                subject_name = raw_key[:_MAX_STANCE_CHARS]
            else:
                subject_name = raw_key[:_MAX_STANCE_CHARS]

            basis = self._basis_entry(item, now)
            existing = self.get(subject_type, key, dimension)
            if existing is None:
                self._insert(
                    subject_type=subject_type,
                    subject_key=key,
                    subject_name=subject_name,
                    dimension=dimension,
                    stance=text,
                    basis=[basis] if basis else [],
                    intensity=_clamp(item.get("intensity", 0.5)),
                    evidence_count=1,
                    now=now,
                )
                created += 1
                continue

            merged_basis = ([basis] if basis else []) + existing.basis
            evidence = existing.evidence_count + 1
            self._rewrite(
                stance_id=existing.id,
                stance=text,
                basis=merged_basis[:_MAX_BASIS],
                intensity=_blend(existing.intensity, _clamp(item.get("intensity", 0.5))),
                confidence=min(
                    _clamp(item.get("confidence", existing.confidence)),
                    _confidence_cap(evidence),
                ),
                evidence_count=evidence,
                subject_name=subject_name or existing.subject_name,
                now=now,
            )
            updated += 1
        return {"created": created, "updated": updated}

    def edit(
        self,
        stance_id: int,
        *,
        stance: str | None = None,
        intensity: float | None = None,
    ) -> bool:
        """手工改写。不动 `evidence_count` 与 `revision`：

        修订数是「她自己改了多少次想法」的证据，手工纠正错字不该算进去。
        """
        fields: dict[str, object] = {}
        if stance is not None:
            text = " ".join(str(stance).split())[:_MAX_STANCE_CHARS]
            if text:
                fields["stance"] = text
        if intensity is not None:
            fields["intensity"] = _clamp(intensity)
        if not fields:
            return False
        columns = ", ".join(f"{key} = ?" for key in fields)
        cursor = self._db.execute(
            f"UPDATE stances SET {columns}, updated_at = ? WHERE id = ?",
            (*fields.values(), time.time(), int(stance_id)),
        )
        return bool(cursor.rowcount)

    def delete(self, stance_id: int) -> bool:
        cursor = self._db.execute("DELETE FROM stances WHERE id = ?", (int(stance_id),))
        return bool(cursor.rowcount)

    def _insert(
        self,
        *,
        subject_type: str,
        subject_key: str,
        subject_name: str,
        dimension: str,
        stance: str,
        basis: list[dict],
        intensity: float,
        evidence_count: int,
        now: float,
    ) -> None:
        self._db.execute(
            """
            INSERT INTO stances
                (subject_type, subject_key, subject_name, dimension, stance, basis,
                 intensity, confidence, evidence_count, revision, first_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
            """,
            (
                subject_type,
                subject_key,
                subject_name,
                dimension,
                stance,
                json.dumps(basis, ensure_ascii=False),
                intensity,
                _confidence_cap(evidence_count),
                evidence_count,
                now,
                now,
            ),
        )

    def _rewrite(
        self,
        *,
        stance_id: int,
        stance: str,
        basis: list[dict],
        intensity: float,
        confidence: float,
        evidence_count: int,
        subject_name: str,
        now: float,
    ) -> None:
        self._db.execute(
            """
            UPDATE stances
               SET stance = ?, basis = ?, intensity = ?, confidence = ?,
                   evidence_count = ?, subject_name = ?, revision = revision + 1,
                   updated_at = ?
             WHERE id = ?
            """,
            (
                stance,
                json.dumps(basis, ensure_ascii=False),
                intensity,
                confidence,
                evidence_count,
                subject_name,
                now,
                stance_id,
            ),
        )

    def _basis_entry(self, item: dict, now: float) -> dict:
        """一条依据 = 逐字原话 + 时间。

        没有逐字原话的依据不记：模型很容易给出「他经常抱怨」这种概括，
        而概括是记忆该干的事。原话是「她为什么这么想」的唯一硬证据。
        """
        quote = " ".join(str(item.get("basis_quote") or "").split())
        if not quote:
            return {}
        return {"quote": quote[:_MAX_BASIS_QUOTE], "at": now}

    # ── 读取 ──────────────────────────────────────────────

    def of_subject(self, subject_type: str, subject_key: str) -> list[Stance]:
        rows = self._db.query(
            """
            SELECT * FROM stances
             WHERE subject_type = ? AND subject_key = ?
             ORDER BY intensity DESC, updated_at DESC
            """,
            (subject_type, normalize_key(subject_key)),
        )
        return [Stance.from_row(row) for row in rows]

    def of_user(self, user_id: str) -> list[Stance]:
        return self.of_subject(SUBJECT_USER, user_id) if user_id else []

    def recent(self, limit: int = 20) -> list[Stance]:
        """最近被动过脑子的态度：注入与页面用。

        按 `updated_at` 倒序而不是 `intensity`：注入要的是「她最近在想什么」，
        一条强度很高但半年前就没再变过的旧判断，不该每轮都占位置。
        """
        rows = self._db.query(
            "SELECT * FROM stances ORDER BY updated_at DESC, id DESC LIMIT ?",
            (max(int(limit), 1),),
        )
        return [Stance.from_row(row) for row in rows]

    def ranked(self, limit: int = 20) -> list[Stance]:
        """按「鲜活度」排序：强度 × 新鲜度。联想与页面共用。"""
        now = time.time()
        stances = self.recent(limit=max(int(limit) * 3, 30))
        stances.sort(
            key=lambda s: s.effective_intensity(now) * _freshness(s, now),
            reverse=True,
        )
        return stances[: max(int(limit), 1)]

    def count(self) -> int:
        row = self._db.query_one("SELECT COUNT(*) AS n FROM stances")
        return int(row["n"]) if row else 0

    def list_page(self, limit: int, offset: int) -> tuple[list[Stance], int]:
        limit = max(int(limit), 1)
        offset = max(int(offset), 0)
        total = self.count()
        rows = self._db.query(
            "SELECT * FROM stances ORDER BY updated_at DESC, id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        return [Stance.from_row(row) for row in rows], total


def _blend(old: float, new: float) -> float:
    return _clamp(_BLEND_NEW * new + (1.0 - _BLEND_NEW) * old)


def _freshness(stance: Stance, now: float) -> float:
    """30 天为尺度的新鲜度，只用于排序，不参与注入正文。"""
    idle_days = max(now - (stance.updated_at or stance.first_at or now), 0.0) / 86400.0
    return 0.5 ** (idle_days / 30.0)
