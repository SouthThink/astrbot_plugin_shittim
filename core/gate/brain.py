"""主脑占用闸门：同一时刻只允许一轮主脑在说话。

AstrBot 的 LLM 互斥只覆盖**同一个会话**：

    async with session_lock_manager.acquire_lock(event.unified_msg_origin):

私聊与群聊是两个事件、两个任务（`event_bus.dispatch()` 对每条消息
`asyncio.create_task(...)`），因此可以并发跑两轮主脑。本插件接管了放行权，
就必须自己补上这一层：统一人格只有一份记忆，两轮同时生成的结果不只是
「各回各的」，还会互相读对方刚写进去的内容，表现为「在群里回答私聊的话题」。

用法：**放行之前**先 `acquire()`，拿到就把事件唤醒（`is_at_or_wake_command = True`）；
拿不到就有界等待，等到仍不空则放弃这一条（拦下，日志留痕）。

占用按**事件对象身份**记录：`on_agent_done` / `after_message_sent` /
`on_decorating_result` / `stay_silent` 四个收口点都会调 `release()`，
谁先到谁释放，重复释放是无害的空操作——任何一条路径漏了都不会把闸门锁死。

四个收口点是必要的：主脑用 `stay_silent` 收回发言时消息链为空，AstrBot 的
respond 阶段遇到空链直接返回（`respond/stage.py`：「空消息链没有任何可发内容，
直接返回、也不触发 after_message_sent」），只挂 `after_message_sent` 会漏。

无锁实现：`acquire()` 里检查与置位之间没有 `await`，在单线程事件循环中是原子的。
不持有 `asyncio.Lock` 是为了避免「占用者异常消失把锁永久带走」。

两道兜底，防止占用被卡住：

- **拿到了却没开始跑**（`_UNSTARTED_SECONDS`）：放行之后仍可能有后续阶段把事件
  拦下来（白名单、会话停用、限流……），主脑根本没被调用，`on_agent_begin`
  不会来。这种占用只等一个很短的时间就作废，否则一次没跑起来的放行会冻结
  其它所有会话好几分钟。
- **开始了却没结束**（`_STALE_SECONDS`）：主脑异常退出、插件被热重载打断。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from astrbot.api import logger

# 等待占用释放的上限（秒）。私聊与群聊几乎同时到达时，后一条最多等前一轮生成完。
# 这个量级与判定超时（engine._JUDGE_TIMEOUT_SECONDS = 30）对齐：主脑一轮正常
# 不会超过它，等不到就说明对方卡住了，再等只会让用户白等。
_WAIT_SECONDS = 30.0

# 「已占用但主脑迟迟没开始」的作废时长（秒）。放行之后事件仍可能被后续阶段
# 拦下（白名单 / 会话停用 / 限流），此时 on_agent_begin 不会触发。
_UNSTARTED_SECONDS = 15.0

# 「主脑跑起来了却没结束」的作废时长（秒）。四个收口点理论上都会释放，
# 这里只是兜底：主脑异常退出、插件被热重载打断时，占用不会永久占着。
_STALE_SECONDS = 300.0

# 等待期间的轮询间隔（秒）。事件循环里让出控制权即可，不必高频。
_POLL_SECONDS = 0.2


class BrainGate:
    """全局「主脑是否正在说话」的唯一事实源。"""

    def __init__(
        self,
        wait_seconds: float = _WAIT_SECONDS,
        stale_seconds: float = _STALE_SECONDS,
        unstarted_seconds: float = _UNSTARTED_SECONDS,
    ):
        self._wait_seconds = max(float(wait_seconds), 0.0)
        self._stale_seconds = max(float(stale_seconds), 1.0)
        self._unstarted_seconds = max(float(unstarted_seconds), 1.0)
        self._holder: Any = None
        self._holder_chat: str = ""
        self._since: float = 0.0
        self._begun = False
        self._closed = False
        # 被让过路的次数，仅供排障页面看
        self._yields: int = 0

    # ── 查询 ─────────────────────────────────────────────────

    def _stale(self, now: float) -> bool:
        """占用是否已经该作废：开始前用短时限，开始后用长时限。"""
        if self._since <= 0.0:
            return False
        limit = self._stale_seconds if self._begun else self._unstarted_seconds
        return (now - self._since) >= limit

    @property
    def busy(self) -> bool:
        if self._holder is None:
            return False
        return not self._stale(time.time())

    @property
    def holder_chat(self) -> str:
        return self._holder_chat if self.busy else ""

    def owns(self, event: Any) -> bool:
        """这个事件是不是当前的占用者。"""
        return self._holder is not None and self._holder is event

    def mark_begun(self, event: Any) -> None:
        """主脑真的开始跑了：从此改用长时限兜底。"""
        if self._holder is not None and self._holder is event:
            self._begun = True

    def snapshot(self) -> dict:
        """供 `web_api.runtime()` 展示。"""
        now = time.time()
        return {
            "busy": self.busy,
            "chat": self.holder_chat,
            "held_seconds": round(now - self._since, 1) if self.busy else 0.0,
            "begun": bool(self._begun),
            "wait_seconds": self._wait_seconds,
            "stale_seconds": self._stale_seconds,
            "unstarted_seconds": self._unstarted_seconds,
            "yields": self._yields,
        }

    # ── 占用 ─────────────────────────────────────────────────

    async def acquire(self, event: Any, chat_id: str = "") -> bool:
        """申请占用；拿到返回 True，超时返回 False。

        已经持有占用的事件再次申请（同一条事件重复走到放行分支）直接放行，
        不做自我等待。
        """
        if self._closed:
            return False
        deadline = time.time() + self._wait_seconds
        while True:
            now = time.time()
            if self._holder is None or self._holder is event or self._stale(now):
                if self._holder is not None and self._holder is not event:
                    logger.warning(
                        "[什亭之枢] 主脑占用未按时释放，已强制作废 | %s | 已占用 %.0fs | 是否已开始：%s",
                        self._holder_chat or "未知",
                        now - self._since,
                        "是" if self._begun else "否",
                    )
                self._holder = event
                self._holder_chat = chat_id
                self._since = now
                self._begun = False
                return True
            if now >= deadline:
                return False
            # 让出控制权：等待期间其它事件仍要能跑（尤其是正在说话的那一轮）
            await asyncio.sleep(_POLL_SECONDS)

    def release(self, event: Any = None, reason: str = "") -> bool:
        """释放占用；只有占用者本人能释放，重复调用是空操作。"""
        if self._holder is None:
            return False
        if event is not None and self._holder is not event:
            return False
        holder_chat = self._holder_chat
        self._holder = None
        self._holder_chat = ""
        self._since = 0.0
        self._begun = False
        if reason:
            logger.debug("[什亭之枢] 主脑占用释放 | %s | %s", holder_chat, reason)
        return True

    def yield_once(self) -> None:
        """记录一次「被让路」（等待超时后放弃这条事件）。"""
        self._yields += 1

    def clear(self) -> None:
        """插件卸载时调用：作废占用并拒绝新的申请。"""
        self._holder = None
        self._holder_chat = ""
        self._since = 0.0
        self._begun = False
        self._closed = True
