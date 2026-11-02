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

模型拿到「已有画像 + 这个人最近的发言」，输出**完整的**新描述与关键词。
因此删除同样成立：新关键词列表里不再出现的词就是被删掉了。

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
        单独注入只是重复。
        """
        return bool(self.summary or self.keywords)

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
    ) -> bool:
        """用模型产出的完整结果替换描述与关键词。

        刻意做**整体替换**而非合并：删除关键词只能靠「新列表里没有它」来表达。
        """
        if not user_id:
            return False
        now = time.time()
        cursor = self._db.execute(
            """
            UPDATE user_profiles
               SET summary = ?,
                   keywords = ?,
                   profiled_at = ?,
                   updated_at = ?
             WHERE user_id = ?
            """,
            (
                (summary or "").strip(),
                json.dumps(
                    [k for k in keywords if str(k).strip()], ensure_ascii=False
                ),
                now,
                now,
                user_id,
            ),
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

    def __init__(self, store: ProfileStore, extractor, ledger, config):
        self._store = store
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
        """用这个人的新增发言重写画像；返回是否真的更新了。"""
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

        result = await self._extractor.build_profile(
            user_id=profile.user_id,
            display_name=profile.display_name,
            aliases=profile.aliases,
            summary=profile.summary,
            keywords=profile.keywords,
            records=records,
        )
        if result is None:
            return False
        summary, keywords = result
        if not self._store.update(
            profile.user_id,
            summary=summary,
            keywords=keywords,
        ):
            return False
        logger.info(
            "[什亭之枢] 用户画像已更新 | %s（%s）| 发言%d条 → 描述%d字 / 关键词%d个",
            profile.label(),
            profile.user_id,
            len(rows),
            len(summary),
            len(keywords),
        )
        return True

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
