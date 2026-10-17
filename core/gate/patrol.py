"""群聊巡检：把「逐条判定」改成「攒一小段，再看一眼」。

## 为什么

原来的决断层对每条群聊消息都要问一次轻量模型，代价有三个：

- **成本**：群里闲聊越密，模型调用越多。
- **延迟闯祸**：一次调用撞上上游超时重试，判定可能几十秒后才回来。
  于是她拿着「很久以前那条消息」的判断去干预「现在」的对话——实测出现过
  95 秒后才放行一条早已翻篇的图片，她突然插一句驴唇不对马嘴的话。
- **抖动**：同类的消息时而拦时而放，看起来前后矛盾。

改成像人一样：听到一句话不急，等对方把这段说完，再决定要不要接。
判断为不回就安静一阵，过后重新开始听。

## 机制

防抖是目的，扣押是实现原理（沿用天使之心验证过的做法）：

- 群聊里的普通消息不在处理器里立即判定，而是**扣押**在窗口里。
- 窗口内又来消息 → 旧事件 KILL，只留最后一条事件持有窗口。
- 窗口安静满 window 秒（或自首次消息起满 max_wait 秒）→ 拿这段消息判一次。
- 判「不回」→ 该会话进入 cooldown 冷却，冷却期内连窗口都不开。
- 判「回」→ 放行最后一条事件。被 KILL 的那些消息由账本合并回
  `req.contexts`，所以主脑照样看得到整段对话，不会漏掉前文。

## 噪声不进模型

窗口内如果只有图片/表情、整段复读、或同一个人连刷短句，直接判不回，
而且**不进冷却**：它不值得花一次模型调用，也不该让她因此沉默一分钟。

## 放行的是最后一条事件

AstrBot 一条消息一个事件，`stop_event()` 只能决定这一条的命运。所以窗口里
只有最后一条能被放行——这正是「只放行最后边界事件」的含义。

## 私聊防抖（silent 窗口）

私聊不走模型判定（规则层直接定「要回」），但同样不该立刻回：对方常常把一句
话拆成好几条发，她抢在第二条前面回就成了打断。所以私聊也开窗，只是**静默
模式**——等最后一条消息之后安静满 `private_debounce_seconds` 秒就放行，
不问模型、不进冷却、不计入「上次判定时刻」。计时基准跟着最新那条走，
对方一直说就一直等，不设封顶。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from astrbot.api import logger

from ..common.event_utils import is_placeholder_only
from ..memory.ledger import format_message_line

PROCESS = "PROCESS"
KILL = "KILL"

# 判定占用该会话时，窗口到期后退让重试的间隔与次数上限。
# 上限存在的意义：宁可这一轮沉默，也不要无休止地等一个慢调用。
_RETRY_DELAY = 0.5
_MAX_TIMER_RETRY = 20

# 「同一个人连刷」的判定门槛：连续这么多条、且每条都这么短，才当噪声。
# 定得偏保守是刻意的——一个人连说几句短话也可能是正经内容，
# 误杀比漏判更伤人格。
_BURST_MIN_MESSAGES = 5
_BURST_MAX_CHARS = 6

# 窗口起点往前留的宽限：记账（priority 100）发生在扣押（priority 90）之前，
# 第一条消息的入库时间因此略早于窗口开启时刻，不给宽限就会被漏掉。
_WINDOW_GRACE_SECONDS = 1.0


@dataclass
class PatrolRecord:
    """一个会话当前扣押中的窗口。"""

    chat_id: str
    event: Any
    future: asyncio.Future
    version: int
    delay: float
    # 窗口首次开启的时刻。续窗保持不变，作为 max_wait 封顶的基准，
    # 否则群里一直有人说话就能无限推迟判定。
    opened_at: float
    timer: asyncio.Task | None = None
    # 静默窗口（私聊防抖）：到点直接放行，不问模型、不进冷却。
    silent: bool = False


def noise_reason(rows: list) -> str:
    """这段消息是否属于「不值得开一次模型」的噪声。

    只收高置信度的三类，宁可漏判也不要误杀正常对话：
    整段只有图片表情、整段复读、同一个人连刷极短句。
    """
    texts = [str(row["content"] or "").strip() for row in rows]
    texts = [text for text in texts if text]
    if not texts:
        return "整段没有内容"

    if all(is_placeholder_only(text) for text in texts):
        return "整段只有图片或表情"

    if len(texts) >= 2 and len(set(texts)) == 1:
        return "整段是复读"

    senders = [str(row["sender_id"] or "") for row in rows]
    if (
        len(texts) >= _BURST_MIN_MESSAGES
        and len(set(senders)) == 1
        and all(len(text) <= _BURST_MAX_CHARS for text in texts)
        and not any(text.endswith(("？", "?")) for text in texts)
    ):
        return "同一个人连刷短句"

    return ""


class PatrolManager:
    """群聊巡检窗口的调度与扣押。"""

    def __init__(self, config, short_term, decider) -> None:
        self._config = config
        self._short_term = short_term
        self._decider = decider
        self._lock = asyncio.Lock()
        self._records: dict[str, PatrolRecord] = {}
        # 正在判定中的会话：跨 await 的判定必须串行，否则两轮会互相踩
        self._judging: set[str] = set()
        self._cooldown_until: dict[str, float] = {}
        # 上次判定的时刻。下次判据从它算起，这样冷却期间攒下的对话
        # 不会凭白丢掉——她该一次看清这段时间里到底发生了什么。
        self._last_judged_at: dict[str, float] = {}
        self._version_seq = 0
        self._closed = False

    def update_config(self, config) -> None:
        self._config = config

    # ── 对外入口 ──────────────────────────────────────────

    async def hold(
        self, chat_id: str, event: Any, *, silent: bool = False
    ) -> asyncio.Future | None:
        """把这条消息扣押进窗口。

        Args:
            silent: 私聊防抖模式。窗口到期后不再问模型，直接放行最后一条事件，
                因此既不进冷却也不占用判定。计时基准是**最后一条消息**而不是
                首次开窗时刻——对方一直说，她就一直等。

        Returns:
            Future: 调用方 await，结果为 PROCESS（放行）或 KILL（拦下）。
            None:   冷却中，调用方应立即拦下这条事件，连窗口都不开。
        """
        if self._closed or not chat_id:
            return None

        now = time.time()
        async with self._lock:
            if not silent and self._cooldown_until.get(chat_id, 0.0) > now:
                logger.debug("[什亭之枢] 巡检冷却中，拦下 | %s", chat_id)
                return None

            existing = self._records.get(chat_id)
            if existing is not None:
                # 旧事件 KILL，只留最后一条持有窗口
                self._kill(existing, "被更新的消息取代")
            if silent:
                opened_at = now
                delay = self._silent_delay()
            else:
                opened_at = existing.opened_at if existing is not None else now
                delay = self._window_delay(opened_at, now)

            future: asyncio.Future = asyncio.get_running_loop().create_future()
            record = PatrolRecord(
                chat_id=chat_id,
                event=event,
                future=future,
                version=self._next_version(),
                delay=delay,
                opened_at=opened_at,
                silent=silent,
            )
            record.timer = asyncio.create_task(self._timer(record))
            self._records[chat_id] = record
            logger.debug(
                "[什亭之枢] %s %s | %s | 等待%.1fs",
                "私聊防抖" if silent else "巡检窗口",
                "续窗" if existing is not None else "开窗",
                chat_id,
                record.delay,
            )
            return future

    async def cleanup(self) -> None:
        """插件卸载：所有扣押中的事件一律按拦下收场。"""
        self._closed = True
        async with self._lock:
            records = list(self._records.values())
            self._records.clear()
            self._judging.clear()
            self._last_judged_at.clear()
        for record in records:
            self._kill(record, "插件卸载")
        timers = [record.timer for record in records if record.timer]
        if timers:
            await asyncio.gather(*timers, return_exceptions=True)

    # ── 内部 ──────────────────────────────────────────────

    def _next_version(self) -> int:
        self._version_seq += 1
        return self._version_seq

    def _window_delay(self, opened_at: float, now: float) -> float:
        """本次要等多久才判定：窗口时长与封顶剩余取小。"""
        try:
            window = float(self._config.patrol_window_seconds)
            max_wait = float(self._config.patrol_max_wait_seconds)
        except (TypeError, ValueError):
            window, max_wait = 5.0, 20.0
        window = max(window, 0.01)
        remaining = max_wait - (now - opened_at)
        if remaining <= 0:
            # 已经等够封顶时长：立刻判，不能再拖
            return 0.01
        return max(0.01, min(window, remaining))

    def _silent_delay(self) -> float:
        """私聊防抖的静默时长。不设封顶：对方一直说，她就一直等。"""
        try:
            seconds = float(self._config.private_debounce_seconds)
        except (TypeError, ValueError):
            seconds = 0.0
        return max(seconds, 0.01)

    def _kill(self, record: PatrolRecord, reason: str) -> None:
        """把一条窗口记录作废（取消定时、结束等待）。"""
        if record.timer and not record.timer.done():
            record.timer.cancel()
        if record.future and not record.future.done():
            record.future.set_result(KILL)
        logger.debug("[什亭之枢] 巡检窗口作废 | %s | %s", record.chat_id, reason)

    async def _silent_timer(self, record: PatrolRecord) -> None:
        """私聊防抖：静默满时长就直接放行最后一条，不问模型、不进冷却。"""
        try:
            await asyncio.sleep(record.delay)
            async with self._lock:
                current = self._records.get(record.chat_id)
                if current is None or current.version != record.version:
                    # 已被更新的消息取代，或已被清理
                    return
                self._records.pop(record.chat_id, None)
            self._finish(record, PROCESS)
        except asyncio.CancelledError:
            return
        except Exception as error:
            logger.error("[什亭之枢] 私聊防抖异常，按放行处理: %s", error, exc_info=True)
            self._finish(record, PROCESS)

    async def _timer(self, record: PatrolRecord) -> None:
        """窗口计时；到点后占用会话做一次判定。"""
        if record.silent:
            await self._silent_timer(record)
            return
        try:
            for attempt in range(_MAX_TIMER_RETRY + 1):
                delay = record.delay if attempt == 0 else _RETRY_DELAY
                await asyncio.sleep(delay)

                async with self._lock:
                    current = self._records.get(record.chat_id)
                    if current is None or current.version != record.version:
                        # 已被更新的消息取代，或已被清理
                        return
                    if record.chat_id in self._judging:
                        # 上一轮判定还没结束：让开，等它跑完再判，
                        # 否则两轮判定会同时读写同一段账本
                        if attempt < _MAX_TIMER_RETRY:
                            continue
                        self._records.pop(record.chat_id, None)
                        self._finish(record, KILL)
                        return
                    self._records.pop(record.chat_id, None)
                    self._judging.add(record.chat_id)

                should_reply, reason, enter_cooldown = await self._decide(record)

                async with self._lock:
                    self._judging.discard(record.chat_id)
                    if enter_cooldown and not should_reply:
                        self._cooldown_until[record.chat_id] = (
                            time.time() + self._cooldown_seconds()
                        )
                logger.info(
                    "[什亭之枢] 巡检判定 | %s | %s | %s",
                    record.chat_id,
                    "开口" if should_reply else "沉默",
                    reason,
                )
                # 先落日志再唤醒等待方：调用方的「拦/放」日志要排在理由之后
                self._finish(record, PROCESS if should_reply else KILL)
                return
        except asyncio.CancelledError:
            return
        except Exception as error:
            logger.error(
                "[什亭之枢] 巡检判定异常，按不回复处理: %s", error, exc_info=True
            )
            async with self._lock:
                self._judging.discard(record.chat_id)
            self._finish(record, KILL)

    def _finish(self, record: PatrolRecord, verdict: str) -> None:
        if record.future and not record.future.done():
            record.future.set_result(verdict)

    def _cooldown_seconds(self) -> float:
        try:
            return max(float(self._config.patrol_cooldown_seconds), 0.0)
        except (TypeError, ValueError):
            return 60.0

    async def _decide(self, record: PatrolRecord) -> tuple[bool, str, bool]:
        """判定这一窗消息。返回 (要不要开口, 理由, 是否计入冷却)。"""
        # 起点取「上次判定之后」与「本次窗口开启」中更早的那个：冷却期间
        # 攒下的对话同样要给她看，否则她会莫名其妙地缺失一段记忆。
        since = record.opened_at - _WINDOW_GRACE_SECONDS
        anchor = self._last_judged_at.get(record.chat_id)
        if anchor is not None:
            since = min(since, anchor)
        self._last_judged_at[record.chat_id] = time.time()

        rows = self._short_term.messages_since(record.chat_id, since)
        if not rows:
            return False, "窗口里没有新消息", False

        if bool(getattr(self._config, "patrol_noise_filter", True)):
            noise = noise_reason(rows)
            if noise:
                return False, noise, False

        lines = [format_message_line(row) for row in rows]
        chat_label = str(rows[-1]["chat_label"] or "")
        before = self._history_before(record.chat_id, rows[0])
        decision = await self._decider.judge_window(
            record.chat_id, chat_label, lines, before=before
        )
        return bool(decision.should_reply), decision.reason, True

    def _history_before(self, chat_id: str, first_row) -> list[str]:
        """窗口开始前这个会话的几句原文。

        判定对象是「上次判定之后的增量」，安静一段时间后它可能只剩
        「不懂」这样一句。把紧邻的前文一并交给模型，它才看得出这句在接什么。
        """
        try:
            limit = int(self._config.context_messages)
        except (TypeError, ValueError):
            limit = 6
        if limit <= 0:
            return []
        try:
            rows = self._short_term.messages_before(
                chat_id,
                float(first_row["created_at"]),
                int(first_row["id"]),
                limit,
            )
        except Exception as error:
            logger.debug("[什亭之枢] 读取判定前文失败: %s", error)
            return []
        return [format_message_line(row) for row in rows]
