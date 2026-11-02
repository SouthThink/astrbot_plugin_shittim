"""用户画像：以 QQ 号识别同一个人，跨私聊与群聊积累印象。

## 身份边界

同一个人在私聊、群 A、群 B 里可能有不同的昵称，但 `sender_id`（QQ 号）
始终相同。因此画像的主键是 `(platform, user_id)`，与 `chat_id` 无关：

- **名字**由 `user_ledger.user_names` 实时累积（每收一条消息就记一次），
  用于识别「小绿」和「绿」是同一个人；
- **印象**（描述 + 关键词）只在这里维护，由整理记忆时的轻量模型逐步完善。

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


def _profile_from_row(row: sqlite3.Row) -> UserProfile:
    aliases = [str(n) for n in _json_list(row["user_names"]) if str(n).strip()]
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


_SELECT_WITH_NAMES = """
SELECT p.*, l.user_names
  FROM user_profiles p
  LEFT JOIN user_ledger l
    ON l.platform = p.platform AND l.user_id = p.user_id
"""


class ProfileStore:
    """用户画像的读写。名字由 `user_ledger` 提供，这里只管印象。"""

    def __init__(self, db: MemoryDB, config):
        self._db = db
        self._config = config
        self._backfill()

    def update_config(self, config) -> None:
        self._config = config

    def _backfill(self) -> None:
        """为升级前已存在的用户补一行画像记录。

        老库只有 `user_ledger`，没有画像表。缺了这一步，老用户在页面里会
        全部消失，直到他们再次发言才重新出现。只补空行，不猜任何印象。
        """
        try:
            self._db.execute(
                """
                INSERT INTO user_profiles
                    (platform, user_id, message_count, first_seen, last_seen,
                     created_at, updated_at)
                SELECT l.platform, l.user_id, 0, l.first_seen, l.last_seen,
                       l.first_seen, l.last_seen
                  FROM user_ledger l
                  LEFT JOIN user_profiles p
                    ON p.platform = l.platform AND p.user_id = l.user_id
                 WHERE p.user_id IS NULL
                """
            )
        except Exception as error:
            logger.error("[什亭之枢] 用户画像回填失败: %s", error)

    # ── 写入 ──────────────────────────────────────────────

    def touch(self, platform: str, user_id: str, user_name: str = "") -> None:
        """登记一次发言：累加条数、刷新活跃时间与最新名字。

        画像内容不在这里生成——逐条调用模型成本过高，见模块开头的说明。
        """
        if not platform or not user_id:
            return
        now = time.time()
        row = self._db.query_one(
            "SELECT platform FROM user_profiles WHERE platform = ? AND user_id = ?",
            (platform, user_id),
        )
        if row is None:
            self._db.execute(
                """
                INSERT INTO user_profiles
                    (platform, user_id, display_name, message_count,
                     first_seen, last_seen, created_at, updated_at)
                VALUES (?, ?, ?, 1, ?, ?, ?, ?)
                """,
                (platform, user_id, user_name or "", now, now, now, now),
            )
            return
        self._db.execute(
            """
            UPDATE user_profiles
               SET display_name = COALESCE(NULLIF(?, ''), display_name),
                   message_count = message_count + 1,
                   last_seen = ?,
                   updated_at = ?
             WHERE platform = ? AND user_id = ?
            """,
            (user_name or "", now, now, platform, user_id),
        )

    def update(
        self,
        platform: str,
        user_id: str,
        *,
        summary: str,
        keywords: list[str],
    ) -> bool:
        """用模型产出的完整结果替换描述与关键词。

        刻意做**整体替换**而非合并：删除关键词只能靠「新列表里没有它」来表达。
        """
        if not platform or not user_id:
            return False
        now = time.time()
        cursor = self._db.execute(
            """
            UPDATE user_profiles
               SET summary = ?,
                   keywords = ?,
                   profiled_at = ?,
                   updated_at = ?
             WHERE platform = ? AND user_id = ?
            """,
            (
                (summary or "").strip(),
                json.dumps(
                    [k for k in keywords if str(k).strip()], ensure_ascii=False
                ),
                now,
                now,
                platform,
                user_id,
            ),
        )
        return bool(getattr(cursor, "rowcount", 0))

    # ── 读取 ──────────────────────────────────────────────

    def get(self, platform: str, user_id: str) -> UserProfile | None:
        if not platform or not user_id:
            return None
        row = self._db.query_one(
            _SELECT_WITH_NAMES + " WHERE p.platform = ? AND p.user_id = ?",
            (platform, user_id),
        )
        return _profile_from_row(row) if row is not None else None

    def list_page(self, limit: int, offset: int) -> tuple[list[UserProfile], int]:
        """按最近活跃分页，供页面展示。"""
        limit = max(int(limit), 1)
        offset = max(int(offset), 0)
        total_row = self._db.query_one(
            "SELECT COUNT(*) AS n FROM user_profiles"
        )
        total = int(total_row["n"] or 0) if total_row is not None else 0
        rows = self._db.query(
            _SELECT_WITH_NAMES
            + " ORDER BY p.last_seen DESC, p.user_id ASC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        return [_profile_from_row(row) for row in rows], total

    def active_since(self, since: float, limit: int) -> list[UserProfile]:
        """最近活跃过的用户，按活跃时间倒序（整理时优先处理）。"""
        rows = self._db.query(
            _SELECT_WITH_NAMES
            + " WHERE p.last_seen >= ? ORDER BY p.last_seen DESC LIMIT ?",
            (float(since), max(int(limit), 1)),
        )
        return [_profile_from_row(row) for row in rows]

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
            profile.platform,
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
