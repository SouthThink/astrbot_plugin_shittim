"""群聊巡检：将逐条判定改为窗口聚合判定。

## 设计动机

逐条调用轻量模型存在以下问题：

- **成本**：群消息越密集，模型调用次数越多。
- **延迟风险**：调用遇到上游超时重试时，判定结果可能延迟数十秒返回，
  导致对已经过时的消息执行放行。
- **一致性**：同类消息的判定结果不稳定，表现为拦截策略前后矛盾。

窗口聚合的做法是：等这段对话真正安静下来再统一判定；判定为不回复时进入
冷却期，冷却结束后重新等待新消息。

## 机制

防抖是目的，事件扣押是实现方式：

- 群聊普通消息不在处理器中立即判定，而是**扣押**在窗口内。
- 窗口内出现新消息时，旧事件以 KILL 结束，仅最后一条事件持有窗口，
  同时把静默计时重置为完整窗口。
- 窗口安静满 `patrol_window_seconds` 后判定一次。持续发言时不封顶判定，
  只在真正无人说话时才询问模型。**开口之后同样要重新等满一个窗口**：
  她说完一句，下一句仍要等对话再次安静，不会因为「刚说过话」就立刻判定。
- 判定为不回复 → 该会话进入 `patrol_cooldown_seconds` 冷却期，冷却期内不开启窗口。
- 判定为回复 → 放行最后一条事件。被 KILL 的消息仍记录在账本中，主脑通过全局时间线读取完整对话。
- **开口本身不写冷却**：判定只说「值得接」，真正决定说不说的是主脑。主脑随后
  调用 `stay_silent` 收回发言时，才由 `enter_cooldown()` 让该会话进入冷却——
  「巡检判了沉默」与「主脑决定不说」是同一件事的两个环节，都算沉默。
- 冷却只挡**普通群消息**：@、私聊与文本里提到名字都能穿透，仍会开窗。
- 窗口正文包含**她紧邻的上一句话**：判定看到的是「她说 → 对方回」的完整来回，
  按发生顺序排列，顺序本身说明关系，不额外加提示词。

## 噪声不进入模型

窗口内若仅含图片/表情、整段重复内容，或同一发送者连续发送短句，直接判定为不回复，
且**不进入冷却**：这类内容不值得消耗模型调用，也不应导致会话长时间静默。

## 放行最后一条事件

AstrBot 每条消息对应一个事件，`stop_event()` 只能作用于当前事件。
因此窗口内只有最后一条事件可以被放行，即「仅放行最后边界事件」。

## 静默窗口（私聊防抖与 @）

私聊与群内 @ 不走模型判定（规则层直接判定为需要回复），但同样不应立即回复：
对方常将一句话拆成多条发送，立即回复会造成打断。因此这些消息也开启窗口，
但为**静默模式**：最后一条消息之后安静满 `private_debounce_seconds` 秒即放行，
不询问模型、不进入冷却。计时基准跟随最新消息，并受
`private_debounce_max_seconds` 封顶，持续有人说话也会到点回复。
静默窗口开启后，普通消息只会顺延计时，不会顶掉持有窗口的 @ 事件。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from astrbot.api import logger

from ..common.event_utils import (
    EXTRA_JUDGE_REASON,
    is_placeholder_only,
    is_private_chat,
    set_extra,
)
from ..memory.ledger import format_message_line

PROCESS = "PROCESS"
KILL = "KILL"

# 连续短句噪声的判定门槛：连续条数与单条最大长度。
# 阈值刻意保守——连续短句也可能是正常内容，误判正常对话的代价更高。
_BURST_MIN_MESSAGES = 5
_BURST_MAX_CHARS = 6

# 窗口起点向前保留的宽限：记账（priority 100）先于扣押（priority 90）执行，
# 第一条消息的入库时间略早于窗口开启时刻，无宽限时会遗漏该条消息。
_WINDOW_GRACE_SECONDS = 1.0

# 最近判定记录的保留条数：每个会话只留最后一条，页面用来回答
# 「上一次为什么没接话」。只存内存，重启即清空。
_MAX_LAST_DECISIONS = 40


@dataclass
class PatrolRecord:
    """一个会话当前扣押中的窗口。"""

    chat_id: str
    event: Any
    future: asyncio.Future
    version: int
    delay: float
    # 窗口首次开启时刻。续窗时保持不变，作为本次判定取样起点的基准：
    # 连续发言被拆成多个窗口时，判定仍能看到整段对话。
    opened_at: float
    # 计时轮次。顺延窗口时递增，旧计时任务据此识别自己已过期：
    # 只靠 `cancel()` 不够——已睡醒、正在等锁的旧任务取消不掉。
    epoch: int = 0
    timer: asyncio.Task | None = None
    # 静默窗口（私聊防抖 / 群内 @）：到期直接放行，不调用模型、不进入冷却。
    silent: bool = False
    # 窗口里只是提到了名字：判定时要求模型先确认是否真的在叫她。
    mention: bool = False
    # 当前这一轮计时的到期时刻（新消息会顺延）。仅供页面快照展示，
    # 真正的调度仍由 timer 负责，不依赖该字段。
    deadline: float = 0.0


def _row_text(row) -> str:
    """取一行的可判定文本：正文 + 合并转发内容。

    合并转发在账本里只是「[转发消息]」占位，真正的内容记录在
    `forward_content`；只看正文会把「转发了一段聊天」误判成纯噪声。
    """
    text = str(row["content"] or "").strip()
    try:
        forwarded = str(row["forward_content"] or "").strip()
    except (KeyError, IndexError, TypeError):
        forwarded = ""
    return f"{text} {forwarded}".strip()


def noise_reason(rows: list) -> str:
    """判断这段消息是否属于无需调用模型的噪声。

    仅匹配三类高置信度噪声，优先避免误判正常对话：
    整段仅含图片/表情、整段内容重复、同一发送者连续发送极短句。
    """
    texts = [_row_text(row) for row in rows]
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
        # 判为不回复后的冷却：完全关掉该会话的判定。开口不写冷却——
        # 她说完一句之后，新消息照常开窗，只是要重新等满一个窗口
        # （见 `note_spoke`）。
        self._cooldown_until: dict[str, float] = {}
        # 最近一次判定（每个会话一条），供页面排障
        self._last_decisions: dict[str, dict] = {}
        self._version_seq = 0
        self._closed = False

    def update_config(self, config) -> None:
        self._config = config

    def snapshot(self) -> dict:
        """当前巡检运行时状态（供页面展示）。

        只读取内存中的记录，不做任何 IO，也不改变调度：页面看到的是
        「谁在等、等多久、谁在判定、谁在冷却」，与 `hold()` 的判据完全同源。
        """
        now = time.time()
        windows = [
            {
                "chat_id": record.chat_id,
                "silent": bool(record.silent),
                "mention": bool(record.mention),
                "opened_at": float(record.opened_at),
                "delay": float(record.delay),
                "remaining": max(float(record.deadline or 0.0) - now, 0.0),
            }
            for record in self._records.values()
        ]
        windows.sort(key=lambda item: item["remaining"])
        cooldowns = [
            {
                "chat_id": chat_id,
                "until": float(until),
                "remaining": max(float(until) - now, 0.0),
            }
            for chat_id, until in self._cooldown_until.items()
            if float(until) > now
        ]
        cooldowns.sort(key=lambda item: item["remaining"])
        last_decisions = sorted(
            self._last_decisions.values(), key=lambda item: item["at"], reverse=True
        )
        return {
            "windows": windows,
            "judging": sorted(self._judging),
            "cooldowns": cooldowns,
            "last_decisions": last_decisions,
        }

    # ── 对外入口 ──────────────────────────────────────────

    async def hold(
        self,
        chat_id: str,
        event: Any,
        *,
        silent: bool = False,
        mention: bool = False,
    ) -> asyncio.Future | None:
        """将该消息扣押进窗口。

        Args:
            silent: 静默模式（私聊防抖）。窗口到期后不调用模型，直接放行最后
                一条事件，因此不进入冷却，也不占用判定流程。计时基准为
                **最后一条消息**，而非首次开窗时刻。
            mention: 这条消息在文本里提到了她（不是 @）。窗口内只要出现过
                这种提及，判定时就会附带「先确认是否真的在叫她」的要求。

        Returns:
            Future: 调用方 await，结果为 PROCESS（放行）或 KILL（拦下）。
            None:   处于冷却期，调用方应立即拦下该事件，不开启窗口。
        """
        if self._closed or not chat_id:
            return None

        now = time.time()
        async with self._lock:
            existing = self._records.get(chat_id)

            # 静默窗口（@ / 私聊）进行中：普通消息只记账并顺延静默计时，
            # 不顶掉持有窗口的 @ 事件——避免一次 @ 把后续整段闲聊都变成必回。
            if existing is not None and existing.silent and not silent:
                self._extend_silent(existing, now)
                blocked: asyncio.Future = (
                    asyncio.get_running_loop().create_future()
                )
                blocked.set_result(KILL)
                return blocked

            # 冷却期内不开普通巡检窗口。能穿透冷却的是「有人明确在叫她」：
            # 静默窗口（@ / 私聊）与文本里提到了名字。@ 与私聊本来就要回，
            # 提及则至少值得再问一次模型——把人叫了名字却完全不理，比多说一句更糟。
            # 冷却只挡没人叫她的普通群消息。冷却只有一种来源：**这一轮最终沉默**
            # （巡检判了不回复，或主脑判定开口后用 `stay_silent` 收回）。
            if not silent and not mention and existing is None:
                if self._cooldown_until.get(chat_id, 0.0) > now:
                    logger.debug("[什亭之枢] 巡检冷却中，拦下 | %s", chat_id)
                    self._remember(
                        chat_id,
                        "cooldown",
                        "冷却中，普通消息不判定、直接拦下",
                        judged=False,
                    )
                    return None

            # 走到这里就是要开新窗口：上一轮留下的冷却到此为止。
            # 穿透冷却的那一句已经开过窗（她可能已经回话），残留的冷却不能再挡
            # 紧跟着的闲聊，否则表现为「叫了她、她答了，下一句又不理」。
            # 顺带清掉过期记录，避免计时字典随会话数无限增长。
            self._cooldown_until.pop(chat_id, None)

            if existing is not None:
                # 旧事件 KILL，只留最后一条持有窗口
                self._kill(existing, "被更新的消息取代")
            # 窗口内只要出现过名字提及，后续消息也按「含提及」判定：
            # 提及句往往不是最后一条，语义需要结合整段对话确认。
            keep_mention = mention or (existing is not None and existing.mention)
            if silent:
                opened_at = now
                delay = self._silent_delay(opened_at, now)
            else:
                opened_at = existing.opened_at if existing is not None else now
                delay = self._window_delay()

            future: asyncio.Future = asyncio.get_running_loop().create_future()
            record = PatrolRecord(
                chat_id=chat_id,
                event=event,
                future=future,
                version=self._next_version(),
                delay=delay,
                opened_at=opened_at,
                silent=silent,
                mention=keep_mention,
                deadline=now + delay,
            )
            record.timer = asyncio.create_task(self._timer(record, record.epoch))
            self._records[chat_id] = record
            logger.debug(
                "[什亭之枢] %s %s | %s | 等待%.1fs",
                "静默防抖" if silent else "巡检窗口",
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
        for record in records:
            self._kill(record, "插件卸载")
        timers = [record.timer for record in records if record.timer]
        if timers:
            await asyncio.gather(*timers, return_exceptions=True)

    async def enter_cooldown(
        self,
        chat_id: str,
        reason: str = "主脑决定本轮不发言",
        *,
        mention: bool = False,
    ) -> None:
        """让该会话进入巡检冷却：这一轮最终没有开口。

        巡检判「开口」只说明值得接，真正说不说由主脑决定。主脑调用
        `stay_silent` 收回发言时，从群友的视角看就是「叫了她、她没理」，
        与巡检自己判沉默没有区别，因此同样要进入冷却——否则会被立刻
        重新开窗，下一段闲聊又把她拽进判定。

        Args:
            reason: 写进最近判定快照的理由，页面据此回答「为什么静默」。
            mention: 本轮是文本提到名字，仅用于快照展示。
        """
        if self._closed or not chat_id:
            return
        # 冷却只由群聊的「说还是不说」产生：私聊的静默窗口不参与。
        if is_private_chat(chat_id):
            return
        until = time.time() + self._cooldown_seconds()
        async with self._lock:
            self._cooldown_until[chat_id] = until
            # 她组织语言期间进来的消息可能已经开过窗：那批消息属于「这一轮」，
            # 冷却就要当场把它们一起挡掉，否则它们到点仍会判定一次。
            # 静默窗口例外——那里面是 @ 或私聊，本来就必回，不能被这一轮沉默掐掉。
            pending = self._records.get(chat_id)
            if pending is not None and not pending.silent:
                self._records.pop(chat_id, None)
            else:
                pending = None
        if pending is not None:
            self._kill(pending, "本轮最终没有开口，一起进入冷却")
        self._remember(chat_id, "silent", reason, judged=False)
        logger.info("[什亭之枢] 巡检判定 | %s | 沉默 | %s", chat_id, reason)
        logger.debug(
            "[什亭之枢] 进入巡检冷却 | %s | 冷却%.0fs | mention=%s",
            chat_id,
            self._cooldown_seconds(),
            bool(mention),
        )

    async def note_spoke(self, chat_id: str) -> None:
        """记下她刚在这个会话说过话，并把仍在等的新消息推后一个完整窗口。

        开口不写冷却：她说完一句，下一句话照常判定。但她组织语言、分段发送
        期间进来的消息往往已经开过窗，计时可能在她说完之前就到了——那会表现
        成「她刚说完立刻又接一句」。这些窗口已经开过、改判据来不及，只能把
        计时从此刻重新起算：判定仍要等对话重新安静满一个窗口。

        只动计时，不动取样起点（`opened_at`）：她说话期间进来的消息还没被
        判定过，把它们挪出窗口等于直接丢掉。
        """
        if self._closed or not chat_id:
            return
        now = time.time()
        rescheduled = 0.0
        async with self._lock:
            record = self._records.get(chat_id)
            # 静默窗口（@ / 私聊）等的是「对方说完」，与她说没说无关，不动。
            if record is not None and not record.silent:
                delay = self._window_delay()
                deadline = now + delay
                if float(record.deadline or 0.0) < deadline:
                    if record.timer and not record.timer.done():
                        record.timer.cancel()
                    record.delay = delay
                    record.deadline = deadline
                    record.epoch += 1
                    # 取消旧 timer 与新建任务之间没有 await，旧的不会插进来；
                    # 已睡醒的旧任务靠 epoch 认出自己过期。
                    record.timer = asyncio.create_task(
                        self._timer(record, record.epoch)
                    )
                    rescheduled = delay
        if rescheduled > 0:
            logger.debug(
                "[什亭之枢] 巡检窗口顺延 | %s | 她刚说完，重新等%.1fs",
                chat_id,
                rescheduled,
            )

    # ── 内部 ──────────────────────────────────────────────

    def _next_version(self) -> int:
        self._version_seq += 1
        return self._version_seq

    def _window_delay(self) -> float:
        """群聊巡检的静默时长。每条新消息都会重置计时，不设封顶。"""
        try:
            window = float(self._config.patrol_window_seconds)
        except (TypeError, ValueError):
            window = 10.0
        return max(window, 0.01)

    def _silent_delay(self, opened_at: float, now: float) -> float:
        """静默防抖的等待时长：跟随最新消息重置，但受最长等待时间约束。

        `private_debounce_max_seconds <= 0` 表示不封顶；否则持续有人说话时，
        最晚也会在开窗后这么久放行，避免回复被无限期推迟。
        """
        try:
            seconds = float(self._config.private_debounce_seconds)
        except (TypeError, ValueError):
            seconds = 0.0
        delay = max(seconds, 0.01)
        try:
            max_wait = float(
                getattr(self._config, "private_debounce_max_seconds", 60.0)
            )
        except (TypeError, ValueError):
            max_wait = 60.0
        if max_wait > 0:
            remaining = max_wait - (now - opened_at)
            if remaining <= 0:
                return 0.01
            delay = min(delay, remaining)
        return max(delay, 0.01)

    def _extend_silent(self, record: PatrolRecord, now: float) -> None:
        """顺延静默窗口的计时，但不更换持有窗口的事件。

        静默窗口由 @ 或私聊打开时，窗口里可能继续涌入普通消息。这些消息
        只应顺延计时，不能让它们顶掉原来那条「一定要回」的事件。
        """
        if record.timer and not record.timer.done():
            record.timer.cancel()
        record.delay = self._silent_delay(record.opened_at, now)
        record.deadline = now + record.delay
        # 只有 cancel() 不够：已睡醒、正在等锁的旧任务取消不掉，epoch 让它自己认出过期。
        record.epoch += 1
        record.timer = asyncio.create_task(self._timer(record, record.epoch))
        logger.debug(
            "[什亭之枢] 静默窗口顺延 | %s | 等待%.1fs",
            record.chat_id,
            record.delay,
        )

    def _kill(self, record: PatrolRecord, reason: str) -> None:
        """作废一条窗口记录（取消定时器并结束等待）。"""
        if record.timer and not record.timer.done():
            record.timer.cancel()
        if record.future and not record.future.done():
            record.future.set_result(KILL)
        logger.debug("[什亭之枢] 巡检窗口作废 | %s | %s", record.chat_id, reason)

    def _remember(
        self, chat_id: str, verdict: str, reason: str, *, judged: bool
    ) -> None:
        """记下该会话最近一次判定/拦放事实，供页面排障。

        `judged=False` 表示没有真正询问判定模型（冷却拦下、静默放行、
        窗口作废），这类记录不该被误读成「模型决定不接」。
        """
        if not chat_id:
            return
        self._last_decisions.pop(chat_id, None)
        self._last_decisions[chat_id] = {
            "chat_id": chat_id,
            "at": time.time(),
            "verdict": verdict,
            "reason": reason,
            "judged": bool(judged),
        }
        while len(self._last_decisions) > _MAX_LAST_DECISIONS:
            oldest = next(iter(self._last_decisions))
            self._last_decisions.pop(oldest, None)

    async def _silent_timer(self, record: PatrolRecord, epoch: int) -> None:
        """静默防抖：静默满时长后直接放行最后一条事件，不调用模型、不进入冷却。"""
        try:
            await asyncio.sleep(record.delay)
            async with self._lock:
                current = self._records.get(record.chat_id)
                if current is None or current.version != record.version:
                    # 已被更新的消息取代，或已被清理
                    return
                if record.epoch != epoch:
                    # 计时已被顺延，这一轮作废
                    return
                self._records.pop(record.chat_id, None)
            self._remember(
                record.chat_id,
                "release",
                "静默防抖到期，直接放行",
                judged=False,
            )
            self._finish(record, PROCESS)
        except asyncio.CancelledError:
            return
        except Exception as error:
            logger.error("[什亭之枢] 静默防抖异常，按放行处理: %s", error, exc_info=True)
            self._finish(record, PROCESS)

    async def _timer(self, record: PatrolRecord, epoch: int) -> None:
        """窗口计时；到点后占用会话做一次判定。"""
        if record.silent:
            await self._silent_timer(record, epoch)
            return
        try:
            await asyncio.sleep(record.delay)

            async with self._lock:
                current = self._records.get(record.chat_id)
                if current is None or current.version != record.version:
                    # 已被更新的消息取代，或已被清理
                    return
                if record.epoch != epoch:
                    # 计时已被顺延（她刚说完话，或静默窗口续时）：这一轮作废，
                    # 新任务负责下一次到期。
                    return
                if record.chat_id in self._judging:
                    # 上一轮判定尚未结束：本次窗口直接作废，不排队、不补偿。
                    # 冷却或占用结束后也不会补判，只有新消息才重新开窗。
                    self._records.pop(record.chat_id, None)
                    self._remember(
                        record.chat_id,
                        "skipped",
                        "上一轮判定尚未结束，本次窗口作废",
                        judged=False,
                    )
                    self._finish(record, KILL)
                    return
                self._records.pop(record.chat_id, None)
                self._judging.add(record.chat_id)

            should_reply, reason, enter_cooldown = await self._decide(record)
            decided_reply = should_reply

            async with self._lock:
                self._judging.discard(record.chat_id)
                # 判定期间若有新消息开窗，旧事件已经过时：即使模型说「开口」
                # 也不能释放它，否则会在很久之后接一句早已翻篇的话。
                newer = self._records.get(record.chat_id)
                superseded = newer is not None and newer.version > record.version
                if enter_cooldown and not should_reply:
                    self._cooldown_until[record.chat_id] = (
                        time.time() + self._cooldown_seconds()
                    )
            if should_reply and superseded:
                should_reply = False
                reason = f"{reason}（判定期间已出现新消息，交由下一窗口处理）"
                verdict = "superseded"
            elif decided_reply:
                verdict = "reply"
            elif enter_cooldown:
                verdict = "silent"
            else:
                verdict = "skipped"
            # 开口不写冷却：她说完这一句，下一句话照常开窗，只是要重新等满
            # 一个窗口（`note_spoke`）。只有「判为不回复」才写 `_cooldown_until`。
            # 未调用模型就跳过（噪声预过滤 / 窗口没有新消息）只记 DEBUG，
            # 避免刷屏；真正由模型判定的沉默才记 INFO。
            log = logger.info if enter_cooldown else logger.debug
            log(
                "[什亭之枢] 巡检判定 | %s | %s | %s",
                record.chat_id,
                "开口" if should_reply else "沉默",
                reason,
            )
            self._remember(
                record.chat_id,
                verdict,
                reason,
                judged=enter_cooldown,
            )
            if should_reply:
                # 把「为什么值得接」交给主脑，注入时当作它自己刚冒出的念头。
                # 只传这一条理由，不传任何判定标准——标准属于人格提示词。
                set_extra(record.event, EXTRA_JUDGE_REASON, reason)
            # 先写日志再唤醒等待方，使调用方的拦截/放行日志排在判定理由之后
            self._finish(record, PROCESS if should_reply else KILL)
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

    def _previous_line(self, chat_id: str, first_row) -> str:
        """窗口第一条之前紧邻的那一行，若她说的则返回格式化后的文本。

        把这段对话看作一个窗口：她刚说过的话也在里面。补上这一行，判定
        看到的就是「她说 → 对方回」的完整来回，顺序本身已经说明了关系，
        不需要在提示词里额外解释「你刚刚开口了」。

        只在紧邻的一行是 `assistant` 时补，中间夹了别人的话就不补——那时
        顺序已经由前文的其他人发言交代清楚，硬塞一句反而会让人误以为
        这条回应是接在她的那句话后面。
        """
        try:
            rows = self._short_term.messages_before(
                chat_id,
                float(first_row["created_at"]),
                int(first_row["id"]),
                1,
            )
        except Exception as error:
            logger.debug("[什亭之枢] 读取窗口前一行失败: %s", error)
            return ""
        if not rows:
            return ""
        try:
            row = rows[-1]
            if str(row["role"] or "") != "assistant":
                return ""
            return format_message_line(row)
        except (KeyError, IndexError, TypeError):
            return ""

    def _cooldown_seconds(self) -> float:
        try:
            return max(float(self._config.patrol_cooldown_seconds), 0.0)
        except (TypeError, ValueError):
            return 60.0

    async def _decide(self, record: PatrolRecord) -> tuple[bool, str, bool]:
        """判定当前窗口内的消息。返回 (是否回复, 判定理由, 是否计入冷却)。"""
        # 只看本次窗口开启之后的消息：冷却期或被上一轮判定占用期间的消息
        # 不进入本次判定——错过就错过，绝不补判。
        since = record.opened_at - _WINDOW_GRACE_SECONDS

        rows = self._short_term.messages_since(record.chat_id, since)
        if not rows:
            return False, "窗口里没有新消息", False

        # 窗口里只有她自己的发言（例如平台把机器人发出的消息也回灌成事件）：
        # 没有可接的内容，也不能据此让会话进入冷却，否则她刚说完一句，
        # 下一句话就会被自己触发的冷却挡掉。
        if all(str(row["role"] or "") == "assistant" for row in rows):
            return False, "窗口里只有我自己的发言", False

        if bool(getattr(self._config, "patrol_noise_filter", True)):
            noise = noise_reason(rows)
            if noise:
                return False, noise, False

        # 她刚说的话也属于这段对话：补进正文最前面，让模型看到的是一段
        # 按发生顺序排列的完整来回，而不是孤零零的一句回应。判断依据只有
        # 顺序本身，不再额外加提示词。
        previous = self._previous_line(record.chat_id, rows[0])
        lines = [format_message_line(row) for row in rows]
        if previous:
            lines.insert(0, previous)
        chat_label = str(rows[-1]["chat_label"] or "")
        before = self._history_before(record.chat_id, rows[0])
        decision = await self._decider.judge_window(
            record.chat_id,
            chat_label,
            lines,
            before=before,
            mention=record.mention,
        )
        # 技术性失败（超时 / 异常 / 解析失败）不进入冷却：上游抖动不应让群聊静默。
        return (
            bool(decision.should_reply),
            decision.reason,
            not bool(getattr(decision, "technical", False)),
        )

    def _history_before(self, chat_id: str, first_row) -> list[str]:
        """窗口开始前该会话的若干条消息。

        判定对象是「本次窗口内的增量」，静默一段时间后可能只剩一条
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
