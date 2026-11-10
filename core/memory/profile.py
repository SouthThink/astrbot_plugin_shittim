"""用户画像：以 QQ 号识别同一个人，跨私聊与群聊积累印象。

## 身份边界

同一个人在私聊、群 A、群 B 里可能有不同的昵称，但 `sender_id`（QQ 号）
始终相同。因此画像的身份是 **`user_id`**，与 `chat_id`、`platform` 都无关：

- 群聊与私聊、乃至不同机器人实例里的同一个 QQ，共用同一行画像；
- **名字**由 `user_ledger.user_names` 实时累积（每收一条消息就记一次），
  读取时按 `user_id` 跨平台合并，用于识别「小绿」和「绿」是同一个人；
- **印象**（描述 + 关键词）只在这里维护，由整理记忆时的轻量模型逐步完善；
- `platform` 只是「第一次见到他时来自哪个平台实例」的来源标记，不参与判定，
  也不参与主键。

不能用 `(platform, user_id)` 做主键：平台实例 ID 取自 `unified_msg_origin`
首段，同一个 QQ 在群聊与私聊、或两套机器人配置下可能落到不同实例上，
那样一次对话就会长出两份画像。

两者分开存：改名只会更新名字列表，不会触发画像重写；画像也不会因为一次
改名就丢失此前的认知。

## 更新节拍

画像不逐条更新——那样每条消息都要调用模型，成本不可接受。更新发生在
`MemoryMaintainer.run()` 之后（深睡整理 / 兜底维护）：

- 只处理最近 `profile_window_hours` 内说过话的人；
- 自上次更新后新增发言达到 `profile_min_messages` 条才调用模型；
- 单次最多处理 `profile_max_users_per_run` 人，按最近活跃排序。

## 素材与吸收

模型拿到三样东西：已有画像、这个人的**长期 / 永久记忆**、他最近的发言。
长期记忆必须参与——很多认知早已不在短期账本里（被压缩、被清空），只喂
发言会让画像反复从头建立。

模型输出**完整的新描述 + 新关键词**（整体替换），因此删除关键词就是
「新列表里不再出现它」。此外它还返回 `absorbed`：那些内容已被新画像完整
覆盖、不再有独立价值的记忆编号。调用方据此**删除**这些条目——画像接住了
它们的信息，留着只是重复。`manual=1`（明确要求记住的）永不参与删除。

## 观察与态度分开

画像有**两半**，混在一起就会变成「该用户为人友善，我很喜欢他」这种不像人话
的东西：

- `summary` / `keywords`：观察到的他。第三人称、客观、可核验。
- `stance` / `dimensions`：她对他的看法。第一人称、主观、允许多次改写。

`dimensions` 只有三个：`familiarity` / `trust` / `closeness`，都是 0~1。
它们在落库前要过两道处理，缺一不可：

1. **缓慢回归**（`merge_dimensions()`）：新值只占 35%，旧值占 65%。模型看到
   一段开心的对话就写 trust=0.95，第二天看到一次争吵又写 trust=0.1，不回归的
   话关系状态会在两天里来回横跳，表现成「她对他忽冷忽热」。回归让长期状态
   只随持续的证据漂移。
2. **不注入数值**：注入侧只把数值翻译成「很熟 / 一般 / 刚认识」这类描述，
   见 `injector.build_profile_block()`。把 0.73 这种数字递给主脑，它会开始
   表演这个数字——「我今天对你亲近度 73%」不是人说话的方式。

## 失败处理

模型不可用、调用失败或输出不可解析时**不更新** `profiled_at`，下一轮整理
会再次尝试；已生成的画像保持不变，不会被半成品覆盖。
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field

from astrbot.api import logger

from ..common.db import MemoryDB
from .ledger import format_message_line

# 单次画像更新带入的发言总字数上限：超出后从**最旧**的开始丢
_MAX_RECORD_CHARS = 6000
# 单行最多保留的字符数：图片描述与转发内容可能很长，会挤掉真正有信息量的发言
_MAX_LINE_CHARS = 200
# 单次带入的长期 / 永久记忆条数：画像是摘要而不是档案，素材宁精不宁多
_MAX_PROFILE_MEMORIES = 20

# 关系维度：与 extractor._RELATION_DIMENSIONS 一一对应。
# 初始值取 0.3 / 0.3 / 0.25：刚认识的人不熟、也谈不上信不信任，
# 但不能是 0——0 会让「亲近感」永远起步不了（回归是乘法式的）。
DIMENSIONS = ("familiarity", "trust", "closeness")
_DIMENSION_SEEDS: dict[str, float] = {
    "familiarity": 0.3,
    "trust": 0.3,
    "closeness": 0.25,
}
# 新观测只占这么多，其余来自旧值：关系是慢变量，一次对话推不动它
_DIMENSION_RATE = 0.35
# 三个维度都还在种子值这个带宽内，就当作「还没处出关系来」，不写进画像块
_DIMENSION_SEED_BAND = 0.06
# 关系描述不用数值，用词。档位与 injector 的渲染一一对应。
# 最后一档的门槛是 0：任何取值都要能翻成一个词，否则注入侧会出现空行。
_DIMENSION_WORDS: dict[str, tuple[tuple[float, str], ...]] = {
    "familiarity": ((0.7, "很熟"), (0.4, "聊过不少"), (0.0, "刚认识")),
    "trust": ((0.7, "信得过"), (0.4, "还谈不上完全放心"), (0.0, "摸不准")),
    "closeness": ((0.7, "亲近"), (0.4, "相处还算自然"), (0.0, "还没熟络起来")),
}


def merge_dimensions(
    old: dict[str, float] | None, new: dict[str, float] | None
) -> dict[str, float]:
    """把模型给的新关系值缓慢合进旧值。

    纯函数。缺失的维度沿用旧值（新人也从种子值起步），未知键直接丢弃——
    存下没人读的东西只会让下次的提示词里多出一行噪音。
    """
    merged: dict[str, float] = {}
    old = old or {}
    new = new or {}
    for name in DIMENSIONS:
        base = old.get(name)
        if not isinstance(base, (int, float)):
            base = _DIMENSION_SEEDS[name]
        candidate = new.get(name)
        if isinstance(candidate, (int, float)):
            value = _DIMENSION_RATE * float(candidate) + (1.0 - _DIMENSION_RATE) * float(base)
        else:
            value = float(base)
        merged[name] = max(0.0, min(value, 1.0))
    return merged


def describe_dimension(name: str, value: float) -> str:
    """把 0~1 的数值翻译成一个词。

    注入侧只给词、不给数：把 0.73 递给模型，它会照着这个数字表演。
    """
    for threshold, word in _DIMENSION_WORDS.get(name, ()):
        if value >= threshold:
            return word
    return ""


def describe_relation(dimensions: dict[str, float] | None) -> str:
    """把三个关系维度拼成一句话，交给画像块。

    拼法是「你跟他很熟，信得过，也亲近」这类**第一人称的关系描述**，不是
    「familiarity=0.7」。原因同 `describe_dimension`：数值会被表演，词不会。

    三个维度都还在种子值附近时返回空串：那是「还没处出关系来」，
    此时说一句「你跟他：刚认识，摸不准，还没熟络起来」只是噪音——
    刚认识的人本来就是这样，不需要专门告诉她一遍。
    """
    dimensions = dimensions or {}
    if not dimensions:
        return ""
    near_seed = all(
        abs(float(dimensions.get(name, _DIMENSION_SEEDS[name])) - _DIMENSION_SEEDS[name])
        <= _DIMENSION_SEED_BAND
        for name in DIMENSIONS
        if isinstance(dimensions.get(name), (int, float))
    )
    if near_seed:
        return ""
    words: list[str] = []
    for name in DIMENSIONS:
        value = dimensions.get(name)
        if not isinstance(value, (int, float)):
            continue
        word = describe_dimension(name, float(value))
        if word:
            words.append(word)
    if not words:
        return ""
    return "你跟他：" + "，".join(words) + "。"


def _json_obj(raw) -> dict:
    if not raw:
        return {}
    try:
        value = json.loads(str(raw))
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _json_list(raw) -> list:
    if not raw:
        return []
    try:
        value = json.loads(str(raw))
    except (TypeError, ValueError):
        return []
    return value if isinstance(value, list) else []


@dataclass
class UserProfile:
    """一个人的画像。`aliases` 来自用户账本，其余来自画像表。"""

    platform: str = ""
    user_id: str = ""
    display_name: str = ""
    summary: str = ""
    keywords: list[str] = field(default_factory=list)
    # 她对他的看法（第一人称）与关系状态（0~1）。见模块开头的「观察与态度分开」。
    stance: str = ""
    dimensions: dict[str, float] = field(default_factory=dict)
    aliases: list[str] = field(default_factory=list)
    message_count: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    profiled_at: float = 0.0
    created_at: float = 0.0
    updated_at: float = 0.0

    @property
    def has_content(self) -> bool:
        """是否有值得注入 / 展示的「印象」。

        只有名字、没有描述与关键词时不算画像——名字本身已经随消息出现，
        单独注入只是重复。态度也算内容：有些人她答不上来「他是谁」，
        但已经能说出「我不太喜欢跟他说话」。
        """
        return bool(self.summary or self.keywords or self.stance)

    def label(self) -> str:
        """页面与日志用的可读名字。"""
        return self.display_name or self.user_id


def _profile_from_row(row: sqlite3.Row, aliases: list[str]) -> UserProfile:
    display_name = str(row["display_name"] or "").strip()
    if not display_name and aliases:
        # 名字列表按出现顺序追加，末尾是最新用过的名字
        display_name = aliases[-1]
    return UserProfile(
        platform=str(row["platform"] or ""),
        user_id=str(row["user_id"] or ""),
        display_name=display_name,
        summary=str(row["summary"] or ""),
        keywords=[str(k) for k in _json_list(row["keywords"]) if str(k).strip()],
        stance=str(row["stance"] or ""),
        dimensions={
            str(k): float(v)
            for k, v in _json_obj(row["dimensions"]).items()
            if isinstance(v, (int, float))
        },
        aliases=aliases,
        message_count=int(row["message_count"] or 0),
        first_seen=float(row["first_seen"] or 0.0),
        last_seen=float(row["last_seen"] or 0.0),
        profiled_at=float(row["profiled_at"] or 0.0),
        created_at=float(row["created_at"] or 0.0),
        updated_at=float(row["updated_at"] or 0.0),
    )


_SELECT_ALL = "SELECT * FROM user_profiles"


class ProfileStore:
    """用户画像的读写。名字由 `user_ledger` 提供，这里只管印象。"""

    def __init__(self, db: MemoryDB, config):
        self._db = db
        self._config = config
        self._backfill()

    def update_config(self, config) -> None:
        self._config = config

    def _backfill(self) -> None:
        """补行并合并历史遗留的重复画像。

        老库只有 `user_ledger`，没有画像表；更早的版本还按
        `(platform, user_id)` 建过画像，同一个 QQ 会留下多行。补行只填空记录、
        不猜任何印象；合并只保留一行。两步都必须幂等，每次启动都能安全重跑。
        """
        try:
            self._db.execute(
                """
                INSERT INTO user_profiles
                    (platform, user_id, message_count, first_seen, last_seen,
                     created_at, updated_at)
                SELECT l.platform, l.user_id, 0,
                       MIN(l.first_seen), MAX(l.last_seen),
                       MIN(l.first_seen), MAX(l.last_seen)
                  FROM user_ledger l
                 WHERE NOT EXISTS (
                       SELECT 1 FROM user_profiles p WHERE p.user_id = l.user_id)
                 GROUP BY l.user_id
                """
            )
        except Exception as error:
            logger.error("[什亭之枢] 用户画像回填失败: %s", error)
        self._merge_duplicates()

    def _merge_duplicates(self) -> None:
        """把同一个 QQ 的多行画像并成一行。

        保留已有印象的那一行（都没有则取最近活跃的），发言数累加、首次与最近
        时间取极值；其余行删除。合并之后身份就是 `user_id`，与平台实例无关。
        """
        try:
            rows = self._db.query(
                _SELECT_ALL + " ORDER BY last_seen DESC, updated_at DESC"
            )
        except Exception as error:
            logger.error("[什亭之枢] 读取用户画像失败: %s", error)
            return

        groups: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            user_id = str(row["user_id"] or "")
            if user_id:
                groups.setdefault(user_id, []).append(row)

        for user_id, group in groups.items():
            if len(group) < 2:
                continue
            keeper = next(
                (
                    row
                    for row in group
                    if str(row["summary"] or "").strip()
                    or _json_list(row["keywords"])
                ),
                group[0],
            )
            summary = str(keeper["summary"] or "")
            keywords = _json_list(keeper["keywords"])
            if not summary.strip() and not keywords:
                # 保留行没有印象时，借用同组里唯一有内容的那份
                for row in group:
                    if str(row["summary"] or "").strip() or _json_list(
                        row["keywords"]
                    ):
                        summary = str(row["summary"] or "")
                        keywords = _json_list(row["keywords"])
                        break
            display_name = next(
                (
                    str(row["display_name"] or "").strip()
                    for row in group
                    if str(row["display_name"] or "").strip()
                ),
                "",
            )
            firsts = [
                float(row["first_seen"] or 0.0)
                for row in group
                if float(row["first_seen"] or 0.0) > 0
            ]
            created = [
                float(row["created_at"] or 0.0)
                for row in group
                if float(row["created_at"] or 0.0) > 0
            ]
            self._db.execute(
                """
                UPDATE user_profiles
                   SET display_name = ?, summary = ?, keywords = ?,
                       message_count = ?, first_seen = ?, last_seen = ?,
                       profiled_at = ?, created_at = ?, updated_at = ?
                 WHERE platform = ? AND user_id = ?
                """,
                (
                    display_name,
                    summary,
                    json.dumps(keywords, ensure_ascii=False),
                    sum(int(row["message_count"] or 0) for row in group),
                    min(firsts) if firsts else 0.0,
                    max(float(row["last_seen"] or 0.0) for row in group),
                    max(float(row["profiled_at"] or 0.0) for row in group),
                    min(created) if created else 0.0,
                    time.time(),
                    str(keeper["platform"] or ""),
                    user_id,
                ),
            )
            for row in group:
                if str(row["platform"] or "") == str(keeper["platform"] or ""):
                    continue
                self._db.execute(
                    "DELETE FROM user_profiles WHERE platform = ? AND user_id = ?",
                    (str(row["platform"] or ""), user_id),
                )
            logger.info(
                "[什亭之枢] 已合并重复的用户画像 | %s | %d 行并作一行",
                user_id,
                len(group),
            )

    # ── 写入 ──────────────────────────────────────────────

    def touch(self, platform: str, user_id: str, user_name: str = "") -> None:
        """登记一次发言：累加条数、刷新活跃时间与最新名字。

        以 `user_id` 定位而不是 `(platform, user_id)`：同一个人在私聊与群聊、
        乃至不同平台实例下都是同一行。`platform` 只在首次建行时记下来源。
        画像内容不在这里生成——逐条调用模型成本过高，见模块开头的说明。
        """
        if not user_id:
            return
        now = time.time()
        row = self._db.query_one(
            "SELECT platform FROM user_profiles WHERE user_id = ?"
            " ORDER BY last_seen DESC LIMIT 1",
            (user_id,),
        )
        if row is None:
            self._db.execute(
                """
                INSERT INTO user_profiles
                    (platform, user_id, display_name, message_count,
                     first_seen, last_seen, created_at, updated_at)
                VALUES (?, ?, ?, 1, ?, ?, ?, ?)
                """,
                (platform or "", user_id, user_name or "", now, now, now, now),
            )
            return
        self._db.execute(
            """
            UPDATE user_profiles
               SET display_name = COALESCE(NULLIF(?, ''), display_name),
                   message_count = message_count + 1,
                   last_seen = ?,
                   updated_at = ?
             WHERE user_id = ?
            """,
            (user_name or "", now, now, user_id),
        )

    def update(
        self,
        user_id: str,
        *,
        summary: str,
        keywords: list[str],
        stance: str | None = None,
        dimensions: dict[str, float] | None = None,
    ) -> bool:
        """用模型产出的完整结果替换描述与关键词。

        刻意做**整体替换**而非合并：删除关键词只能靠「新列表里没有它」来表达。

        `stance` 传 None 表示「模型这次没给」→ 保留旧值；传空串表示「明确
        清空」（例如刚认识、还没形成看法）。两者不能混用：`_update_one()` 只在
        模型确实给了非空态度时才传值，其余情况传 None。

        `dimensions` 走 `merge_dimensions()` 的缓慢回归，不整体替换——
        关系是累积出来的，不是每轮重算的。
        """
        if not user_id:
            return False
        now = time.time()
        current = self.get(user_id)
        merged_dims = merge_dimensions(
            current.dimensions if current else None, dimensions
        )
        if stance is None:
            stance_sql = "stance = stance"
            stance_params: tuple = ()
        else:
            stance_sql = "stance = ?"
            stance_params = (stance.strip(),)
        cursor = self._db.execute(
            f"""
            UPDATE user_profiles
               SET summary = ?,
                   keywords = ?,
                   {stance_sql},
                   dimensions = ?,
                   profiled_at = ?,
                   updated_at = ?
             WHERE user_id = ?
            """,
            (
                (summary or "").strip(),
                json.dumps(
                    [k for k in keywords if str(k).strip()], ensure_ascii=False
                ),
                *stance_params,
                json.dumps(merged_dims, ensure_ascii=False),
                now,
                now,
                user_id,
            ),
        )
        return bool(getattr(cursor, "rowcount", 0))

    def edit(
        self,
        user_id: str,
        *,
        summary: str,
        keywords: list[str],
        stance: str | None = None,
    ) -> bool:
        """页面手工改写印象（描述与关键词，可选态度）。

        与 `update()` 的唯一区别是**不推进 `profiled_at`**：那是「素材读到哪」
        的水位线，由整理流程维护。手工改动若把它抬到当下，此后到今天为止他
        真正说过的话就再也读不到了，画像会长期停在手工版本。
        """
        if not user_id:
            return False
        if stance is None:
            stance_sql = "stance = stance"
            stance_params: tuple = ()
        else:
            stance_sql = "stance = ?"
            stance_params = (stance.strip(),)
        cursor = self._db.execute(
            f"""
            UPDATE user_profiles
               SET summary = ?,
                   keywords = ?,
                   {stance_sql},
                   updated_at = ?
             WHERE user_id = ?
            """,
            (
                (summary or "").strip(),
                json.dumps(
                    [str(k).strip() for k in keywords if str(k).strip()],
                    ensure_ascii=False,
                ),
                *stance_params,
                time.time(),
                user_id,
            ),
        )
        return bool(getattr(cursor, "rowcount", 0))

    def delete(self, user_id: str) -> bool:
        """删除一份画像。

        只删印象本身：`user_ledger` 里的名字与发言记录一律不动，所以他下次
        说话时会重新建一行，之后再按新的素材慢慢长出印象。
        """
        if not user_id:
            return False
        cursor = self._db.execute(
            "DELETE FROM user_profiles WHERE user_id = ?", (user_id,)
        )
        return bool(getattr(cursor, "rowcount", 0))

    # ── 读取 ──────────────────────────────────────────────

    def _names(self, user_ids: list[str]) -> dict[str, list[str]]:
        """按 `user_id` 跨平台合并名字列表。

        `user_ledger` 按 (platform, user_id) 分行，同一个人在群聊与私聊里可能
        各占一行；画像只认 QQ 号，因此这里把各行的名字并起来去重。
        """
        ids = [str(uid) for uid in user_ids if str(uid or "").strip()]
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = self._db.query(
            "SELECT user_id, user_names FROM user_ledger"
            f" WHERE user_id IN ({placeholders}) ORDER BY last_seen ASC",
            tuple(ids),
        )
        names: dict[str, list[str]] = {}
        for row in rows:
            bucket = names.setdefault(str(row["user_id"] or ""), [])
            for name in _json_list(row["user_names"]):
                text = str(name)
                if text.strip() and text not in bucket:
                    bucket.append(text)
        return names

    def _to_profiles(self, rows) -> list[UserProfile]:
        names = self._names([str(row["user_id"] or "") for row in rows])
        return [
            _profile_from_row(row, names.get(str(row["user_id"] or ""), []))
            for row in rows
        ]

    def get(self, user_id: str) -> UserProfile | None:
        if not user_id:
            return None
        row = self._db.query_one(
            _SELECT_ALL + " WHERE user_id = ? ORDER BY last_seen DESC LIMIT 1",
            (user_id,),
        )
        if row is None:
            return None
        return self._to_profiles([row])[0]

    def list_page(self, limit: int, offset: int) -> tuple[list[UserProfile], int]:
        """按最近活跃分页，供页面展示。"""
        limit = max(int(limit), 1)
        offset = max(int(offset), 0)
        total_row = self._db.query_one(
            "SELECT COUNT(*) AS n FROM user_profiles"
        )
        total = int(total_row["n"] or 0) if total_row is not None else 0
        rows = self._db.query(
            _SELECT_ALL
            + " ORDER BY last_seen DESC, user_id ASC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        return self._to_profiles(rows), total

    def active_since(self, since: float, limit: int) -> list[UserProfile]:
        """最近活跃过的用户，按活跃时间倒序（整理时优先处理）。"""
        rows = self._db.query(
            _SELECT_ALL
            + " WHERE last_seen >= ? ORDER BY last_seen DESC LIMIT ?",
            (float(since), max(int(limit), 1)),
        )
        return self._to_profiles(rows)

    def count(self) -> int:
        row = self._db.query_one("SELECT COUNT(*) AS n FROM user_profiles")
        return int(row["n"] or 0) if row is not None else 0


class ProfileMaintainer:
    """整理记忆时逐步完善用户画像。"""

    def __init__(self, store: ProfileStore, memory_store, extractor, ledger, config):
        self._store = store
        self._memory_store = memory_store
        self._extractor = extractor
        self._ledger = ledger
        self._config = config
        # 白天维护与深睡整理可能同时触发：同一时刻只允许一轮画像更新
        self._running = False

    def update_config(self, config) -> None:
        self._config = config

    async def run(self) -> dict:
        """更新一批活跃用户的画像，返回统计信息。"""
        stats = {"updated": 0, "skipped": 0, "failed": 0}
        if not self._config.profile_enabled or self._running:
            return stats
        if not getattr(self._extractor, "available", False):
            return stats

        self._running = True
        try:
            window = max(int(self._config.profile_window_hours), 1) * 3600
            profiles = self._store.active_since(
                time.time() - window,
                max(int(self._config.profile_max_users_per_run), 1),
            )
            for profile in profiles:
                try:
                    if await self._update_one(profile):
                        stats["updated"] += 1
                    else:
                        stats["skipped"] += 1
                except Exception as error:
                    stats["failed"] += 1
                    logger.error(
                        "[什亭之枢] 更新用户画像失败 | %s: %s",
                        profile.user_id,
                        error,
                    )
        finally:
            self._running = False

        if stats["updated"]:
            logger.info(
                "[什亭之枢] 用户画像更新完成 更新%d 跳过%d 失败%d",
                stats["updated"],
                stats["skipped"],
                stats["failed"],
            )
        return stats

    async def _update_one(self, profile: UserProfile) -> bool:
        """用这个人的新增发言**与长期记忆**重写画像；返回是否真的更新了。

        素材有两路：短期账本里的新增发言，以及已经沉淀成长期 / 永久记忆的
        事实。后者很重要——很多认知早就不在短期账本里了（被压缩、被清空），
        只喂发言会让画像反复从头建立，还容易丢掉已经确认过的事。
        """
        rows = self._ledger.user_messages_since(
            profile.user_id,
            profile.profiled_at,
            limit=max(int(self._config.profile_max_messages), 1),
        )
        if len(rows) < max(int(self._config.profile_min_messages), 1):
            return False

        records = self._render_records(rows)
        if not records:
            return False

        memories = self._memories_of(profile.user_id)
        result = await self._extractor.build_profile(
            user_id=profile.user_id,
            display_name=profile.display_name,
            aliases=profile.aliases,
            summary=profile.summary,
            keywords=profile.keywords,
            records=records,
            memories=self._render_memories(memories),
            stance=profile.stance,
            dimensions=profile.dimensions,
        )
        if result is None:
            return False
        # build_profile 返回 ProfileDraft；态度为空串时传 None 以保留旧值
        updated = self._store.update(
            profile.user_id,
            summary=result.summary,
            keywords=result.keywords,
            stance=result.stance or None,
            dimensions=result.dimensions,
        )
        if not updated:
            return False
        removed = self._absorb(memories, result.absorbed)
        logger.info(
            "[什亭之枢] 用户画像已更新 | %s（%s）| 发言%d条 + 记忆%d条 → "
            "描述%d字 / 关键词%d个 / 态度%d字 / 关系%s / 吸收删除%d条",
            profile.label(),
            profile.user_id,
            len(rows),
            len(memories),
            len(result.summary),
            len(result.keywords),
            len(result.stance or ""),
            " ".join(f"{k}={v:.2f}" for k, v in (result.dimensions or {}).items())
            or "未更新",
            removed,
        )
        return True

    def _memories_of(self, user_id: str) -> list:
        """取这个人的长期 / 永久记忆；取不到时退化为只用发言。"""
        try:
            return self._memory_store.memories_of_user(
                user_id, _MAX_PROFILE_MEMORIES
            )
        except Exception as error:
            logger.debug("[什亭之枢] 读取画像素材记忆失败: %s", error)
            return []

    @staticmethod
    def _render_memories(memories: list) -> str:
        """把记忆渲染成「#编号 内容」，编号用于让模型指认可被吸收的条目。"""
        lines = [
            f"#{memory.id} {str(memory.content or '').strip()}"
            for memory in memories
            if str(memory.content or "").strip()
        ]
        return "\n".join(lines[:_MAX_PROFILE_MEMORIES])

    def _absorb(self, memories: list, absorbed_ids: list[int]) -> int:
        """删除已被画像完整吸收的记忆。

        只删这次真正提交给模型的条目，并跳过 `manual=1`——那是她或用户明确
        要求记住的，不能由整理隐式删除。模型拿不准时不会列出来，因此列出的
        条目可以放心删；失败只记日志，不影响画像已经写入的结果。
        """
        if not absorbed_ids:
            return 0
        candidates = {memory.id: memory for memory in memories if not memory.manual}
        removed = 0
        for memory_id in absorbed_ids:
            memory = candidates.get(int(memory_id))
            if memory is None:
                continue
            try:
                self._memory_store.delete(memory.id)
            except Exception as error:
                logger.error(
                    "[什亭之枢] 删除被画像吸收的记忆失败 | #%d: %s",
                    memory.id,
                    error,
                )
                continue
            removed += 1
            logger.info(
                "[什亭之枢] 记忆已并入用户画像并删除 | #%d | %s",
                memory.id,
                str(memory.content or "")[:40],
            )
        return removed

    @staticmethod
    def _render_records(rows: list[sqlite3.Row]) -> str:
        """把发言渲染成模型可读的文本，总量受字数上限约束。

        从**最新**往回取：画像要反映现在的样子，早期的旧发言应优先让位。
        每行做一次截断，避免一条长转发把整段预算吃掉。
        """
        lines: list[str] = []
        used = 0
        for row in reversed(rows):
            line = format_message_line(row).strip()
            if not line:
                continue
            line = line[:_MAX_LINE_CHARS]
            if used + len(line) + 1 > _MAX_RECORD_CHARS:
                break
            lines.append(line)
            used += len(line) + 1
        lines.reverse()
        return "\n".join(lines)
