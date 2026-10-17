"""全局作息状态机：阶段推导与状态持久化。

## 状态由纯函数推导而非累积

需求中的两条规则在时间上重叠：浅睡满 2 小时转深睡、起床前 2 小时转浅睡。
若各自维护独立计时器，会出现「深睡中却显示可被叫醒」这类自相矛盾的中间态。

因此这里将阶段定义为 `now` 的**纯函数**：相同输入在任何时刻得到相同结果。
重启后无需恢复任何中间态，计算一次即可确定当前阶段，只需持久化几个「事实」
（本轮窗口起止、被叫醒时刻等）。

## 三个状态

- `AWAKE`：清醒，一切照旧
- `LIGHT`：浅睡，可被 @ / 私聊叫醒
- `DEEP`：深睡，不可叫醒

「被叫醒后的清醒宽限」是一个**覆盖层**，不进入 `phase_at()`——它由调用方单独判断。
这样可以避免污染 tick 的阶段转换：被叫醒不会被误判为正常起床。
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

from ..common.db import MemoryDB

AWAKE = "awake"
LIGHT = "light"
DEEP = "deep"

# 睡眠窗口内的两个阶段
SLEEP_PHASES = (LIGHT, DEEP)

_SECONDS_PER_DAY = 86400.0

# 时刻解析失败时的兜底（比"整天不睡"更安全）
_FALLBACK_SLEEP = "23:30"
_FALLBACK_WAKE = "07:00"


def _start_of_day(ts: float) -> float:
    """本地时区当天的 00:00 时间戳。"""
    tm = time.localtime(ts)
    return time.mktime((tm.tm_year, tm.tm_mon, tm.tm_mday, 0, 0, 0, 0, 0, -1))


def _date_str(ts: float) -> str:
    tm = time.localtime(ts)
    return f"{tm.tm_year:04d}-{tm.tm_mon:02d}-{tm.tm_mday:02d}"


def parse_hhmm(text: str) -> int | None:
    """把 "HH:MM" 解析成当天的秒偏移；格式不对返回 None。"""
    raw = (text or "").strip()
    if not raw:
        return None
    parts = raw.split(":")
    if len(parts) != 2:
        return None
    try:
        hour = int(parts[0])
        minute = int(parts[1])
    except (TypeError, ValueError):
        return None
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour * 3600 + minute * 60


@dataclass(frozen=True)
class SleepWindow:
    """一轮作息：从入睡到起床。plan_date 以**入睡日**为基准。"""

    sleep_at: float
    wake_at: float
    plan_date: str


def _window_for_day(
    now: float, config, postponed_hours: float, day_offset: int
) -> SleepWindow:
    """构造「以 now 所在日期 + day_offset 那天为入睡日」的那一轮作息窗口。"""
    sleep_sec = parse_hhmm(getattr(config, "sleep_at", ""))
    if sleep_sec is None:
        sleep_sec = parse_hhmm(_FALLBACK_SLEEP) or 0
    wake_sec = parse_hhmm(getattr(config, "wake_at", ""))
    if wake_sec is None:
        wake_sec = parse_hhmm(_FALLBACK_WAKE) or 0
    latest_sec = parse_hhmm(getattr(config, "wake_latest", ""))

    postpone = max(float(postponed_hours or 0.0), 0.0) * 3600.0
    day_start = _start_of_day(now) + day_offset * _SECONDS_PER_DAY

    sleep_at = day_start + sleep_sec + postpone
    # 起床时刻早于入睡时刻 → 起床落在次日
    wake_day = day_start + _SECONDS_PER_DAY if wake_sec <= sleep_sec else day_start
    wake_at = wake_day + wake_sec + postpone
    if latest_sec is not None:
        wake_at = min(wake_at, _start_of_day(wake_at) + latest_sec)
    return SleepWindow(
        sleep_at=sleep_at, wake_at=wake_at, plan_date=_date_str(day_start)
    )


def plan_window(now: float, config, postponed_hours: float = 0.0) -> SleepWindow:
    """给出 `now` 所在（或即将到来）的作息窗口。

    逐日回看一天：先试「昨天入睡 → 今天起床」，再试「今天入睡 → 明天起床」，
    取第一个起床时刻还没到的窗口。这样跨零点的作息（23:30 入睡、次日 07:00 起床）
    与同日凌晨的作息（01:00 入睡、07:00 起床）都不需要特判。

    推迟入睡会让入睡与起床一起顺延，但起床受 `wake_latest` 上限约束。
    """
    for offset in (-1, 0):
        window = _window_for_day(now, config, postponed_hours, offset)
        if now < window.wake_at:
            return window

    # 兜底：今天的窗口已经整体过去（作息设在当日凌晨且已过起床点），取次日
    return _window_for_day(now, config, postponed_hours, 1)


def last_sleep_start(now: float, config, postponed_hours: float = 0.0) -> float:
    """最近一次**已经开始**的入睡时刻；无法确定时返回 0。

    起床提示以它作为「睡眠期间」的起点（入睡之后收到、且尚未归档的消息才算
    睡眠期消息）。不能直接使用 `plan_window()`：起床时该函数的窗口已切换到
    下一轮。
    """
    best = 0.0
    for offset in (0, -1, -2):
        sleep_at = _window_for_day(now, config, postponed_hours, offset).sleep_at
        if sleep_at <= now and sleep_at > best:
            best = sleep_at
    return best


def phase_at(now: float, window: SleepWindow, config) -> str:
    """由时间轴推导阶段（不含清醒宽限）。"""
    if not (window.sleep_at <= now < window.wake_at):
        return AWAKE
    lead = max(int(getattr(config, "wake_notice_lead_minutes", 120) or 0), 0) * 60
    if now >= window.wake_at - lead:
        # 起床前的一段时间强制浅睡：这段时间可被叫醒
        return LIGHT
    light_until_deep = (
        max(int(getattr(config, "light_until_deep_minutes", 120) or 0), 0) * 60
    )
    if now - window.sleep_at >= light_until_deep:
        return DEEP
    return LIGHT


# 可写入 sleep_state 的字段白名单：避免把外部传入的任意键拼进 SQL
_WRITABLE_FIELDS = frozenset(
    {
        "plan_date",
        "sleep_at",
        "wake_at",
        "phase",
        "postponed_hours",
        "grace_until",
        "awakened_at",
        "wake_notice_pending",
        "wake_notice_fired_at",
        "last_curation_at",
        "last_cleared_at",
    }
)


class SleepStateStore:
    """sleep_state 单行表的读写。"""

    def __init__(self, db: MemoryDB):
        self._db = db

    def load(self) -> dict:
        row = self._db.query_one("SELECT * FROM sleep_state WHERE id = 1")
        return dict(row) if row is not None else {}

    def save(self, **fields) -> None:
        payload = {k: v for k, v in fields.items() if k in _WRITABLE_FIELDS}
        if not payload:
            return
        self._db.execute("INSERT OR IGNORE INTO sleep_state (id, updated_at) VALUES (1, 0)")
        assignments = ", ".join(f"{name} = ?" for name in payload)
        self._db.execute(
            f"UPDATE sleep_state SET {assignments}, updated_at = ? WHERE id = 1",
            (*payload.values(), time.time()),
        )

    def reset_round(self, window: SleepWindow, postponed_hours: float = 0.0) -> None:
        """开启新一轮作息：写入窗口并清掉上一轮的叫醒/宽限/提示痕迹。"""
        self.save(
            plan_date=window.plan_date,
            sleep_at=window.sleep_at,
            wake_at=window.wake_at,
            phase="",
            postponed_hours=float(postponed_hours or 0.0),
            grace_until=0.0,
            awakened_at=0.0,
            wake_notice_pending=0,
        )


class WakeSignalCounter:
    """唤醒信号计数：同一会话在滑动窗口内累计 @ / 私聊次数。

    仅存于内存——窗口只有几分钟，重启后重新计数是合理且更安全的行为
    （重启不会使重启前累计的未达标信号被视为达标）。
    """

    def __init__(self, window_seconds: float, threshold: int):
        self._window = max(float(window_seconds), 0.0)
        self._threshold = max(int(threshold), 1)
        self._hits: dict[str, deque[float]] = {}

    def record(self, chat_id: str, now: float) -> bool:
        """记一次信号，返回是否已达到叫醒阈值。"""
        if not chat_id:
            return False
        hits = self._hits.setdefault(chat_id, deque())
        cutoff = now - self._window
        while hits and hits[0] < cutoff:
            hits.popleft()
        hits.append(now)
        return len(hits) >= self._threshold

    def clear(self, chat_id: str | None = None) -> None:
        """清空信号。不传 chat_id 时清空全部会话。"""
        if chat_id is None:
            self._hits.clear()
        else:
            self._hits.pop(chat_id, None)
