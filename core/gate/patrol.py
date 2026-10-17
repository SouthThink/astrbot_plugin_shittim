"""群聊巡检：将逐条判定改为窗口聚合判定。

## 设计动机

逐条调用轻量模型存在以下问题：

- **成本**：群消息越密集，模型调用次数越多。
- **延迟风险**：调用遇到上游超时重试时，判定结果可能延迟数十秒返回，
  导致对已经过时的消息执行放行。
- **一致性**：同类消息的判定结果不稳定，表现为拦截策略前后矛盾。

窗口聚合的做法是：等待一段对话结束（或达到封顶时长）后统一判定；
判定为不回复时进入冷却期，冷却结束后重新收集。

## 机制

防抖是目的，事件扣押是实现方式：

- 群聊普通消息不在处理器中立即判定，而是**扣押**在窗口内。
- 窗口内出现新消息时，旧事件以 KILL 结束，仅最后一条事件持有窗口。
- 窗口安静满 `patrol_window_seconds`（或自开启起满 `patrol_max_wait_seconds`）后判定一次。
- 判定为不回复 → 该会话进入 `patrol_cooldown_seconds` 冷却期，冷却期内不开启窗口。
- 判定为回复 → 放行最后一条事件。被 KILL 的消息仍记录在账本中，主脑通过全局时间线读取完整对话。

## 噪声不进入模型

窗口内若仅含图片/表情、整段重复内容，或同一发送者连续发送短句，直接判定为不回复，
且**不进入冷却**：这类内容不值得消耗模型调用，也不应导致会话长时间静默。

## 放行最后一条事件

AstrBot 每条消息对应一个事件，`stop_event()` 只能作用于当前事件。
因此窗口内只有最后一条事件可以被放行，即「仅放行最后边界事件」。

## 私聊防抖（静默窗口）

私聊不走模型判定（规则层直接判定为需要回复），但同样不应立即回复：用户常将一句话
拆成多条发送，立即回复会造成打断。因此私聊也开启窗口，但为**静默模式**：最后一条
消息之后安静满 `private_debounce_seconds` 秒即放行，不询问模型、不进入冷却、
不更新判定锚点。计时基准跟随最新消息，不设封顶。
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

# 判定占用该会话时，窗口到期后的退让重试间隔与次数上限。
# 设置上限用于避免因单次慢调用无限期推迟判定：超过上限后按不回复处理。
_RETRY_DELAY = 0.5
_MAX_TIMER_RETRY = 20

# 连续短句噪声的判定门槛：连续条数与单条最大长度。
# 阈值刻意保守——连续短句也可能是正常内容，误判正常对话的代价更高。
_BURST_MIN_MESSAGES = 5
_BURST_MAX_CHARS = 6

# 窗口起点向前保留的宽限：记账（priority 100）先于扣押（priority 90）执行，
# 第一条消息的入库时间略早于窗口开启时刻，无宽限时会遗漏该条消息。
_WINDOW_GRACE_SECONDS = 1.0


@dataclass
class PatrolRecord:
    """一个会话当前扣押中的窗口。"""

    chat_id: str
    event: Any
    future: asyncio.Future
    version: int
    delay: float
    # 窗口首次开启时刻。续窗时保持不变，作为 max_wait 封顶的基准，
    # 避免持续发言导致判定被无限期推迟。
    opened_at: float
    timer: asyncio.Task | None = None
    # 静默窗口（私聊防抖）：到期直接放行，不调用模型、不进入冷却。
    silent: bool = False


def noise_reason(rows: list) -> str:
    """判断这段消息是否属于无需调用模型的噪声。

    仅匹配三类高置信度噪声，优先避免误判正常对话：
    整段仅含图片/表情、整段内容重复、同一发送者连续发送极短句。
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
        # 正在判定中的会话：跨 await 的判定必须串行，避免两轮判定并发读写同一段账本
        self._judging: set[str] = set()
        self._cooldown_until: dict[str, float] = {}
        # 上次判定时刻。后续判定的起点由它计算，使冷却期间产生的对话
        # 仍会进入下一次判定上下文。
        self._last_judged_at: dict[str, float] = {}
        self._version_seq = 0
        self._closed = False

    def update_config(self, config) -> None:
        self._config = config

    # ── 对外入口 ──────────────────────────────────────────

    async def hold(
        self, chat_id: str, event: Any, *, silent: bool = False
    ) -> asyncio.Future | None:
        """将该消息扣押进窗口。

        Args:
            silent: 私聊防抖模式。窗口到期后不调用模型，直接放行最后一条事件，
                因此不进入冷却，也不占用判定流程。计时基准为**最后一条消息**，
                而非首次开窗时刻。

        Returns:
            Future: 调用方 await，结果为 PROCESS（放行）或 KILL（拦下）。
            None:   处于冷却期，调用方应立即拦下该事件，不开启窗口。
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
        """插件卸载：所有扣押中的事件统一按拦下处理。"""
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
        """本次判定的等待时长：窗口时长与封顶剩余时间取较小值。"""
        try:
            window = float(self._config.patrol_window_seconds)
            max_wait = float(self._config.patrol_max_wait_seconds)
        except (TypeError, ValueError):
            window, max_wait = 5.0, 20.0
        window = max(window, 0.01)
        remaining = max_wait - (now - opened_at)
        if remaining <= 0:
            # 已达到封顶时长：立即判定，不再延迟
            return 0.01
        return max(0.01, min(window, remaining))

    def _silent_delay(self) -> float:
        """私聊防抖的静默时长。不设封顶，计时基准始终跟随最新消息。"""
        try:
            seconds = float(self._config.private_debounce_seconds)
        except (TypeError, ValueError):
            seconds = 0.0
        return max(seconds, 0.01)

    def _kill(self, record: PatrolRecord, reason: str) -> None:
        """作废一条窗口记录（取消定时器并结束等待）。"""
        if record.timer and not record.timer.done():
            record.timer.cancel()
        if record.future and not record.future.done():
            record.future.set_result(KILL)
        logger.debug("[什亭之枢] 巡检窗口作废 | %s | %s", record.chat_id, reason)

    async def _silent_timer(self, record: PatrolRecord) -> None:
        """私聊防抖：静默满时长后直接放行最后一条事件，不调用模型、不进入冷却。"""
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
                        # 上一轮判定尚未结束：等待其完成后再判定，
                        # 避免两轮判定并发读写同一段账本
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
                # 先写日志再唤醒等待方，使调用方的拦截/放行日志排在判定理由之后
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
        """判定当前窗口内的消息。返回 (是否回复, 判定理由, 是否计入冷却)。"""
        # 起点取「上次判定时刻」与「本次窗口开启时刻」中较早者：
        # 冷却期间产生的对话也必须进入本次判定上下文，否则会缺失一段记忆。
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
        """窗口开始前该会话的若干条消息。

        判定对象是「上次判定之后的增量」，静默一段时间后可能只剩一条
        缺少上下文的短句。补充紧邻的前文后，模型才能判断该短句的指代对象。
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
