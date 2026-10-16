"""什亭之枢 —— 让智能体拥有统一人格、跨会话记忆与主动行动能力。

## 四条链路

**拦放链路（接管官方 LLM）**
  所有消息 → 记入短期账本 + 刷新该会话焦点 → 决断层规则层
    ├─ 上游指令 / 私聊 / 被点名 / 无文本 → 当场定夺
    └─ 群聊里的普通消息 → 巡检窗口：攒一小段对话再判一次
         ├─ 不回 → stop_event()，官方 LLM 完全不触发，该会话安静一阵
         └─ 要回 → 置位 is_at_or_wake_command，放行主脑生成回复

  不逐条判定是有原因的：单条消息看不出「这轮问答已经闭环」，而一次慢调用
  的判定若迟到几十秒，她会拿着过期判断去干预已经在聊别的话题的群。

**焦点链路（她在哪）**
  她可以同时在多个会话里"在场"，像一个人同时开着几个聊天窗口。
  收到某会话的消息 → 该会话获得焦点；安静超过 N 分钟 → 焦点失效（她休息了）。
  焦点**只约束主动发言**：被动接话永远不受限，任何消息都能唤醒她。

**记忆链路**
  短期：跨会话原始消息总账，带来源标记，保证她不会把 A 群的话回到 B 群；
        条目超阈值时压成摘要，深睡期收口成「当天摘要」并清空
  长期：压缩时顺带产出的条目，排在一条「最后一次被想起」的位置榜上，
        尾部逐级变模糊、最终被遗忘
  永久：被反复用到或被明确要求记住的记忆，不排榜、不降级、不遗忘

**行动链路（跨会话穿梭）**
  list_sessions 看有哪些会话 → say 过去说一句 → leave_session / rest 离开
"""

from __future__ import annotations

import asyncio
import os
import time
from collections import deque

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.message_components import Plain
from astrbot.api.star import Context, Star, register
from astrbot.core.star.star_tools import StarTools

from .core.common.config import (
    DecisionConfig,
    FocusConfig,
    MemoryConfig,
    ScheduleConfig,
)
from .core.common.db import MemoryDB
from .core.common.event_utils import (
    EXTRA_LEFT_TO,
    EXTRA_MESSAGE_ID,
    call_str,
    describe_chat,
    describe_non_text,
    extract_reply_text,
    get_extra,
    is_private_chat,
    is_upstream_command_event,
    platform_of,
    set_extra,
)
from .core.common.time_utils import format_age, format_clock
from .core.focus.manager import FocusManager
from .core.gate.engine import DecisionEngine
from .core.gate.patrol import (
    KILL as PATROL_KILL,
    PROCESS as PATROL_PROCESS,
    PatrolManager,
)
from .core.memory.extractor import MemoryExtractor
from .core.memory.history import (
    strip_invalid_image_urls,
    strip_invalid_images,
)
from .core.memory.injector import (
    NOTICE_AWAKENED,
    NOTICE_WAKE,
    build_injected_entry,
    build_memory_blocks,
    build_schedule_notice,
)
from .core.memory.ledger import (
    SOURCE_LIVE,
    SOURCE_SLEEP,
    ShortTermLedger,
    format_message_line,
)
from .core.memory.maintenance import MemoryMaintainer
from .core.memory.store import MemoryStore
from .core.memory.timeline import build_timeline
from .core.memory.vision import ImageCaptioner
from .core.schedule.state import (
    AWAKE,
    DEEP,
    LIGHT as SLEEP_LIGHT,
    SLEEP_PHASES,
    SleepStateStore,
    WakeSignalCounter,
    last_sleep_start,
    phase_at,
    plan_window,
)
from .web_api import register_all_routes

# 决断层在事件链上的优先级：必须早于 AstrBot 的默认 LLM 阶段。
# 记账用 100、判定用 90，保证判定时当前消息已在账本里。
#
# 注意 90 是个偏「霸道」的取值：它会先于其它插件（priority < 90）的消息处理器
# 执行，被拦下的消息其它插件也看不到。这是「彻底接管」的代价。
# 若发现与其它插件冲突，把 _PRIORITY_GATE 调低（例如 -10，天使之心用的值）
# 即可让出优先级，只要仍高于默认 LLM 阶段就依然能拦住官方 LLM。
_PRIORITY_RECORD = 100
_PRIORITY_GATE = 90
# 睡眠门：必须排在决断层**之前**。决断层的规则层对私聊是「直接放行」，
# 睡眠判定若晚于它，私聊会直接穿透给主脑，"睡觉时不进模型"就成了空话。
# 同时它必须晚于记账（100），这样睡着时收到的消息照常入账。
_PRIORITY_SLEEP = 95
# 工具清理与上下文注入
_PRIORITY_TOOLS = 60
# 上下文注入必须**晚于**所有会重写 req.contexts 的插件。
# AstrBot 的 on_llm_request 按 priority 从大到小执行（数值越大越早），
# 而天使之心（50）会用自己重建的历史整体覆盖 req.contexts——若我们排在
# 它前面，注入块和它看不见的回合会被直接抹掉。天使记忆（40）走 append、
# tool_use_cleaner（0）会重建 contexts，所以取负值排到它们之后，
# 做上下文成型前的最后一手。
_PRIORITY_INJECT = -10
# 坏图清理：与记忆开关无关，且要看得到所有插件处理后的最终上下文，
# 因此排在注入之后，作为发往模型前的最后一道保险。
_PRIORITY_IMAGE_GUARD = -20
# 结果装饰：放在最后，让其它插件先跑完，我们最后决定要不要发
_PRIORITY_DECORATE = -100
# 发送后收口
_PRIORITY_AFTER_SENT = 100

# AstrBot 内置的「给用户发消息」工具。它参数写错时会静默发到当前会话，
# 由我们逐请求摘除，跨会话发言统一走 say。
_BUILTIN_SEND_TOOL = "send_message_to_user"

# 休息状态机落地前的临时兜底：距上次维护超过该时长就在回复后补跑一次
_MAINTENANCE_FALLBACK_SECONDS = 24 * 3600

# 作息检查节拍。60 秒足够捕捉入睡/起床跨越，且这个 tick 只做纯算术，
# 真正耗时的整理由 P3 的深睡调度单独触发。
_SLEEP_TICK_SECONDS = 60.0


def _plugin_version() -> str:
    """从 metadata.yaml 读版本号，避免与 @register 双处维护。"""
    try:
        meta_path = os.path.join(os.path.dirname(__file__), "metadata.yaml")
        with open(meta_path, encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("version:"):
                    return line.split(":", 1)[1].strip()
    except (OSError, ValueError):
        pass
    return "0.0.0"


def _build_plain_chain(text: str) -> MessageChain:
    """构造一条纯文本消息链。

    两种写法都试：`Context.send_message` 要求第二个参数是 MessageChain
    （内部会取 `.chain`），传组件列表会报「'list' object has no attribute
    'chain'」。但不同 AstrBot 版本对 MessageChain 的构造方式不一致，
    所以退一步再试列表构造。
    """
    try:
        return MessageChain().message(text)
    except Exception:
        return MessageChain([Plain(text)])


@register(
    "astrbot_plugin_shittim",
    "SouthThink",
    "什亭之枢：让智能体拥有统一人格、跨会话记忆与主动行动能力",
    _plugin_version(),
    "https://github.com/SouthThink/astrbot_plugin_shittim",
)
class ShittimPlugin(Star):
    """什亭之枢主插件类。"""

    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context)
        self.context = context
        self.config = config if isinstance(config, dict) else {}
        self.enabled = bool(self.config.get("enabled", True))
        self.memory_config = MemoryConfig.from_dict(self.config)
        self.decision_config = DecisionConfig.from_dict(self.config)
        self.focus_config = FocusConfig.from_dict(self.config)
        self.schedule_config = ScheduleConfig.from_dict(self.config)

        # ── 存储与各层组件 ────────────────────────────────────
        data_dir = StarTools.get_data_dir("astrbot_plugin_shittim")
        self.db = MemoryDB(data_dir)
        self.short_term = ShortTermLedger(self.db, self.memory_config)
        self.memory_store = MemoryStore(self.db, self.memory_config)
        # 图片转文字：AstrBot 只给「[图片]」占位，账本里的描述得自己补
        self.captioner = ImageCaptioner(self.db, self.memory_config, self.context)
        self.extractor = MemoryExtractor(self.context, self.memory_config)
        self.maintainer = MemoryMaintainer(
            self.memory_store,
            self.extractor,
            self.memory_config,
            short_term=self.short_term,
        )
        self.decider = DecisionEngine(
            self.context,
            self.decision_config,
            self.memory_config,
            self.short_term,
            self.memory_store,
        )
        self.focus = FocusManager(self.db, self.focus_config)
        # 全局作息：状态由时间轴推导，这里只存「事实」+ 内存快照
        self.sleep_state = SleepStateStore(self.db)
        self._wake_signals = WakeSignalCounter(
            self.schedule_config.wake_signal_window_minutes * 60,
            self.schedule_config.wake_signal_threshold,
        )
        # 群聊巡检：把「每条消息问一次模型」改成「攒一小段再判一次」
        self.patrol = PatrolManager(
            self.decision_config, self.short_term, self.decider
        )

        # ── 运行时状态 ────────────────────────────────────────
        self._background_tasks: set[asyncio.Task] = set()
        self._is_terminating = False
        self._last_maintenance_at = 0.0
        self._send_timestamps: deque[float] = deque()
        self._allowed_targets = self._load_allowed_targets()
        # 摘除内置工具失败只告警一次，避免每条请求刷屏
        self._tool_removal_warned = False
        self._tool_removal_logged = False
        # 作息内存快照：睡眠门是热路径，不允许做 IO，所以窗口与宽限都读内存，
        # 由 tick / 配置保存 / 叫醒时刷新。
        self._sleep_window = None
        self._grace_until = 0.0
        self._sleep_tick_started = False
        # 整理是重活（多次模型调用 + 重写历史），同一时刻只允许一次
        self._curating = False

        tiers = self.memory_store.count_by_tier()
        # 记忆页面（pages/memory/）的只读接口
        register_all_routes(self)
        logger.info(
            "[什亭之枢] 初始化完成 | 接管=%s 决策模型=%s | 记忆=%s 压缩模型=%s | "
            "短期%d条 长期%d条 永久%d条 | 关注中%d个会话",
            "开" if self.decision_config.enabled else "关",
            self._decision_provider_label(),
            "启用" if self.memory_config.enabled else "停用",
            self.memory_config.extract_provider_id or "未配置（压缩关闭）",
            self.short_term.count(),
            tiers.get("long", 0),
            tiers.get("permanent", 0),
            len(self.focus.list_focused(200)),
        )

        # 先把窗口算出来，让睡眠门在第一个 tick 之前就能工作；再启动周期检查。
        self._refresh_sleep_window()
        self._ensure_sleep_tick()
        self._init_curation_clock()
        if not self.schedule_config.enabled:
            # 作息关掉后 tick 不再运行，暂存区从此没人负责放行；这里兜一下，
            # 否则睡眠期间收到的消息会永远卡在 staged=1（既不整理也不清空）。
            self.short_term.release_staged()

    # ══════════════════════════════════════════════════════════
    #  全局作息：状态推导入口
    # ══════════════════════════════════════════════════════════

    def _schedule_active(self) -> bool:
        return (
            self.enabled
            and self.schedule_config.enabled
            and not self._is_terminating
        )

    def _sleep_phase(self, now: float | None = None) -> str:
        """当前作息阶段。只读内存快照，不做任何 IO。"""
        if not self._schedule_active():
            return AWAKE
        window = self._sleep_window
        if window is None:
            window = plan_window(time.time(), self.schedule_config)
        return phase_at(
            time.time() if now is None else now, window, self.schedule_config
        )

    def _in_grace(self, now: float) -> bool:
        return self._grace_until > now

    def _refresh_sleep_window(self) -> None:
        """重算内存中的作息窗口快照（tick / 配置保存时调用）。"""
        if not self.schedule_config.enabled:
            self._sleep_window = None
            return
        postponed = 0.0
        try:
            posted = self.sleep_state.load().get("postponed_hours")
            postponed = float(posted or 0.0)
        except Exception as error:
            logger.error("[什亭之枢] 读取作息状态失败: %s", error)
        self._sleep_window = plan_window(
            time.time(), self.schedule_config, postponed
        )

    def _ensure_sleep_tick(self) -> None:
        """启动作息 tick。

        正常路径在 __init__ 里启动；若那次没拿到运行中的事件循环，
        退到首次事件处理时补启动。
        """
        if self._sleep_tick_started or self._is_terminating:
            return
        try:
            self._spawn(self._sleep_tick_loop())
        except RuntimeError:
            return
        self._sleep_tick_started = True

    def _init_curation_clock(self) -> None:
        """首次运行时把「上次整理」设为现在。

        否则 `last_curation_at = 0` 会被算成「已经很久没整理」，插件一装上就
        立刻跑一次整理——表现为装完就见她"沉默了一会儿"还动了记忆。
        """
        try:
            if not float(self.sleep_state.load().get("last_curation_at") or 0.0):
                self.sleep_state.save(last_curation_at=time.time())
        except Exception as error:
            logger.error("[什亭之枢] 初始化整理节拍失败: %s", error)

    # ══════════════════════════════════════════════════════════
    #  事件：记账（先于一切判定）
    # ══════════════════════════════════════════════════════════

    @filter.event_message_type(filter.EventMessageType.ALL, priority=_PRIORITY_RECORD)
    async def record_incoming_message(self, event: AstrMessageEvent):
        """把收到的消息记入短期账本，并刷新该会话的焦点。

        不回复、不干预，只记账。必须先于决断层执行：无论后面这条消息被放行
        还是被拦下，它都要进入她的记忆。被拦下的消息会被标记为 gated。
        """
        if not self.enabled or self._is_terminating:
            return
        try:
            sender_id = call_str(event, "get_sender_id")
            # 只有确认 sender_id 非空且等于自己时才跳过，避免老版本缺少
            # get_self_id 时两边都取到空串而把正常消息全部丢掉
            if sender_id and sender_id == call_str(event, "get_self_id"):
                return  # 自己的消息由 after_message_sent 记录，避免重复入账
            content = event.get_message_outline() or ""
            if not content.strip():
                # 图片/表情这类无文本消息：用占位描述入账，让她的经历不留空白。
                # 决断层那边拿到的仍是空文本，所以照样不会触发回复。
                content = describe_non_text(event)
            if not content.strip():
                return
            chat_kind, chat_label = describe_chat(event)
            chat_id = event.unified_msg_origin or ""

            # 睡眠期间照常记账，但**不刷新焦点**：否则每个说话的会话都会重新
            # 拿到主动发言资格，与入睡收口的 focus.rest() 直接相抵。
            # 宽限期内她是醒着的（正在回话），那段消息按实时消息算，不当睡眠期。
            now = time.time()
            sleeping = (
                self._sleep_phase(now) in SLEEP_PHASES and not self._in_grace(now)
            )
            if not sleeping:
                # 有人在这个会说话，她就会看过去：刷新焦点。
                # 焦点只约束主动发言，所以这里对「接不接话」没有任何影响。
                self.focus.touch(chat_id, chat_label)

            if not self._memory_active():
                return
            sender_name = call_str(event, "get_sender_name")
            message_id = self.short_term.add(
                chat_id=chat_id,
                chat_label=chat_label,
                chat_kind=chat_kind,
                sender_id=sender_id,
                sender_name=sender_name,
                role="user",
                content=content,
                # 睡眠期消息：标 source='sleep' 以示来源、staged=1 留在暂存区，
                # 起床时统一放行（见 core/schedule/state.py 与 core/memory/ledger.py）。
                source=SOURCE_SLEEP if sleeping else SOURCE_LIVE,
                staged=sleeping,
            )
            if message_id:
                # 供决断层在拦下这条消息时回填标记
                set_extra(event, EXTRA_MESSAGE_ID, message_id)
                # 图里是什么，得自己问一次视觉模型：AstrBot 只给「[图片]」占位
                self._schedule_image_caption(event, message_id)
            self.memory_store.upsert_user(
                platform_of(event) or "unknown",
                sender_id,
                sender_name,
            )
        except Exception as error:
            logger.error("[什亭之枢] 记录消息失败: %s", error)

    # ══════════════════════════════════════════════════════════
    #  事件：睡眠门（先于决断层）
    # ══════════════════════════════════════════════════════════

    @filter.event_message_type(filter.EventMessageType.ALL, priority=_PRIORITY_SLEEP)
    async def sleep_gate(self, event: AstrMessageEvent):
        """睡眠期间拦下一切**非命令**消息，且不让它们进任何模型判定。

        为什么必须排在决断层（90）之前：决断层的规则层对私聊是直接放行的
        （`always_reply_private`），睡眠判定若晚于它，私聊会直接穿透给主脑，
        "睡觉时不进模型"就成了空话。

        判定顺序：

        1. 上游指令 —— 让路。需求是「忽略一切**非命令**消息」，命令必须穿透，
           否则睡眠期间连 `/` 指令都用不了。放行不等于解除睡眠、也不开宽限。
        2. 清醒 —— 交回原有链路。
        3. 宽限期内（刚被叫醒）—— 放行。
        4. 浅睡 —— 累计 @/私聊信号，达标就叫醒，并放行触发叫醒的这一条。
        5. 深睡 —— 一律拦下，不计数、不响应。

        整条链路零 token：只看时间轴与计数，不调用任何模型。
        """
        if not self._schedule_active():
            return
        self._ensure_sleep_tick()
        try:
            if is_upstream_command_event(event):
                return
            now = time.time()
            phase = self._sleep_phase(now)
            if phase == AWAKE or self._in_grace(now):
                return
            chat_id = event.unified_msg_origin or ""
            if (
                phase == SLEEP_LIGHT
                and self._is_wake_signal(event, chat_id)
                and self._wake_signals.record(chat_id, now)
            ):
                self._on_awakened(chat_id, now)
                return
            self._block(event)
        except Exception as error:
            # 睡眠门自己出问题不该让机器人陪葬：放行，交给原生链路
            logger.error("[什亭之枢] 睡眠门异常，已放行: %s", error)

    def _is_wake_signal(self, event, chat_id: str) -> bool:
        """叫醒信号只认 @ 她 与私聊；群聊里的普通消息不算。"""
        if is_private_chat(chat_id):
            return True
        return bool(getattr(event, "is_at_or_wake_command", False))

    # ── 作息收口 ──────────────────────────────────────────

    def _on_fall_asleep(self) -> None:
        """入睡收口：收回全部焦点、清零叫醒信号。

        静默执行，不做睡前告别——主动广播"我睡了"会把这件事变成公开信息，
        而她只该表现为"没回应"。
        """
        try:
            count = self.focus.rest()
            self._wake_signals.clear()
        except Exception as error:
            logger.error("[什亭之枢] 入睡收口失败: %s", error)
            return
        logger.info("[什亭之枢] 入睡收口 | 收回 %d 个会话的焦点", count)

    def _on_wake_up(self) -> None:
        """起床收口：放行睡眠期消息、恢复那些会话的焦点、置起床提示。"""
        try:
            chat_ids = self.short_term.staged_chat_ids()
            labels = [
                (chat_id, self.short_term.label_for_chat(chat_id))
                for chat_id in chat_ids
            ]
            released = self.short_term.release_staged()
            for chat_id, label in labels:
                self.focus.touch(chat_id, label)
            self._wake_signals.clear()
            self._grace_until = 0.0
            # 起床提示待注入：由 P2 的注入通道消费，注入后清标记。
            self.sleep_state.save(wake_notice_pending=1, grace_until=0.0)
        except Exception as error:
            logger.error("[什亭之枢] 起床收口失败: %s", error)
            return
        logger.info(
            "[什亭之枢] 起床收口 | 放行睡眠期消息 %d 条，涉及 %d 个会话",
            released,
            len(labels),
        )

    def _on_awakened(self, chat_id: str, now: float) -> None:
        """被叫醒：开启清醒宽限，并让她在那个会话有资格说话。"""
        grace_seconds = max(int(self.schedule_config.wake_grace_minutes), 0) * 60
        self._grace_until = now + grace_seconds
        self._wake_signals.clear(chat_id)
        try:
            self.focus.touch(chat_id, self.short_term.label_for_chat(chat_id))
            self.sleep_state.save(grace_until=self._grace_until, awakened_at=now)
        except Exception as error:
            logger.error("[什亭之枢] 叫醒收口失败: %s", error)
        logger.info(
            "[什亭之枢] 被叫醒 | %s | 清醒宽限 %d 分钟",
            chat_id,
            grace_seconds // 60,
        )

    # ── 周期检查 ──────────────────────────────────────────

    async def _sleep_tick_loop(self) -> None:
        """按 `_SLEEP_TICK_SECONDS` 周期检查作息跨越。只做纯算术。"""
        while not self._is_terminating:
            try:
                self._sleep_tick()
            except Exception as error:
                logger.error("[什亭之枢] 作息检查失败: %s", error)
            await asyncio.sleep(_SLEEP_TICK_SECONDS)

    def _sleep_tick(self) -> None:
        """把持久化的阶段与时间轴推导出的阶段对齐，并执行转换收口。"""
        if not self._schedule_active():
            return
        now = time.time()
        state = self.sleep_state.load()
        postponed = float(state.get("postponed_hours") or 0.0)
        window = plan_window(now, self.schedule_config, postponed)
        self._sleep_window = window
        self._grace_until = float(state.get("grace_until") or 0.0)

        if str(state.get("plan_date") or "") != window.plan_date:
            # 跨入新一轮作息：先结算上一轮（若已过起床点），再开启这一轮。
            # 结算必须放在 reset 之后，否则 wake_notice_pending 会被重置掉。
            previous_wake = float(state.get("wake_at") or 0.0)
            self.sleep_state.reset_round(window, postponed)
            state = self.sleep_state.load()
            if previous_wake and now >= previous_wake:
                self._on_wake_up()

        phase = phase_at(now, window, self.schedule_config)
        previous_phase = str(state.get("phase") or "")
        if phase != previous_phase:
            logger.info(
                "[什亭之枢] 作息转换 | %s → %s", previous_phase or "未知", phase
            )
            if phase in SLEEP_PHASES and previous_phase in ("", AWAKE):
                self._on_fall_asleep()
            elif phase == AWAKE and previous_phase in SLEEP_PHASES:
                self._on_wake_up()
            self.sleep_state.save(phase=phase)

        # 整理节拍与阶段转换无关：即使这一 tick 没有跨阶段，也要看一眼到没到点。
        self._maybe_curate(phase, now)

    def _maybe_curate(self, phase: str, now: float) -> None:
        """按节拍触发记忆整理（§9.1）。只在 tick 里调用。

        主触发：处于深睡且距上次整理满 `curation_min_interval_hours`。
        兜底触发：不在深睡但距上次已超过 `curation_force_after_hours`——
        睡得太短时深睡阶段可能整段不出现，不兜底就永远不整理。
        """
        cfg = self.schedule_config
        if not cfg.curation_enabled or self._curating or not self._memory_active():
            return
        try:
            last = float(self.sleep_state.load().get("last_curation_at") or 0.0)
        except Exception as error:
            logger.error("[什亭之枢] 读取整理节拍失败: %s", error)
            return
        if not last:
            return
        hours = (now - last) / 3600.0
        if phase == DEEP and hours >= cfg.curation_min_interval_hours:
            self._spawn(self._run_curation(full_reset=True))
            return
        if phase != DEEP and hours >= cfg.curation_force_after_hours:
            logger.warning(
                "[什亭之枢] 已 %.0f 小时未整理记忆且当前不在深睡，"
                "执行兜底整理（不清空账本、不重建历史）",
                hours,
            )
            self._spawn(self._run_curation(full_reset=False))

    # ══════════════════════════════════════════════════════════
    #  事件：拦放（接管官方 LLM）
    # ══════════════════════════════════════════════════════════

    @filter.event_message_type(filter.EventMessageType.ALL, priority=_PRIORITY_GATE)
    async def gate_official_llm(self, event: AstrMessageEvent):
        """在 AstrBot 官方 LLM 被触发之前做拦放判定。

        **规则层**能定的（上游指令、私聊、被点名、空文本）当场定夺；剩下的
        群聊普通消息交给**巡检器**：把这条事件扣押进窗口，等这段对话说完再
        判一次要不要接话——而不是每来一句就问一次模型。

        私聊是个中间态：规则层已经定了「要回」，但回之前同样要攒一攒
        （`_gate_private_debounce`），免得抢在对方第二条消息前面插嘴。

        要回 → 置位 is_at_or_wake_command 强行唤醒主脑，让它生成回复；
        不回 → stop_event()，官方 LLM 完全不触发。
        """
        if not self._gate_active():
            return

        try:
            ruled = self.decider.rule_decide(event)
        except Exception as error:
            # 决断层自己出问题不该让机器人陪葬：放行，交给原生链路
            logger.error("[什亭之枢] 决断层异常，已放行: %s", error)
            return

        if ruled is not None:
            if ruled.debounce:
                await self._gate_private_debounce(event, ruled)
                return
            self._apply_decision(event, ruled)
            return

        await self._gate_by_patrol(event)

    async def _gate_private_debounce(self, event: AstrMessageEvent, ruled) -> None:
        """私聊防抖：静默满 `private_debounce_seconds` 秒没有新消息才唤醒主脑。

        私聊本身不需要模型判定要不要回，但需要等：对方常把一句话拆成几条发，
        立刻回就成了打断。窗口沿用巡检器（静默模式），窗口内旧事件 KILL，
        只留最后一条持有窗口；被 KILL 的那些由账本合并回 `req.contexts`，
        所以主脑读到的仍是完整一段，只是等这段说完再一起回。
        """
        seconds = self._private_debounce_seconds()
        chat_id = event.unified_msg_origin or ""
        if seconds <= 0 or not chat_id:
            # 未配置防抖（或取不到会话标识）：退回立即放行
            self._apply_decision(event, ruled)
            return

        future = await self.patrol.hold(chat_id, event, silent=True)
        if future is None:
            # 只有插件正在卸载时才会走到这里，交给卸载流程收场
            return

        verdict = await future
        if self._is_terminating:
            logger.debug("[什亭之枢] 插件已卸载，放弃本次私聊防抖结果")
            return
        if verdict == PATROL_PROCESS:
            self._release(event)
            logger.info(
                "[什亭之枢] 放行主脑 [私聊防抖] 静默 %.1fs 无新消息", seconds
            )
        else:
            self._block(event)
            logger.info("[什亭之枢] 拦截官方 LLM [私聊防抖] 被更新的消息取代")

    def _private_debounce_seconds(self) -> float:
        """私聊防抖时长；配置取不到时按「不防抖」处理，宁可立刻回也不无声拖延。"""
        try:
            return max(float(self.decision_config.private_debounce_seconds), 0.0)
        except (TypeError, ValueError):
            return 0.0

    async def _gate_by_patrol(self, event: AstrMessageEvent) -> None:
        """群聊里的普通消息：扣押进窗口，攒一段再判。"""
        chat_id = event.unified_msg_origin or ""

        if not self.decision_config.patrol_enabled:
            # 巡检关闭：退回逐条判定，用账本里最新这条当判据
            chat_label, lines = self._latest_window(chat_id)
            decision = await self.decider.judge_window(chat_id, chat_label, lines)
            if self._is_terminating:
                return
            self._apply_decision(event, decision)
            return

        future = await self.patrol.hold(chat_id, event)
        if future is None:
            # 冷却中：安静地拦下，连窗口都不开
            if not self._is_terminating:
                self._block(event)
            return

        verdict = await future
        # 等窗口的这几秒里插件可能已被重载或卸载，此时不该再动这个事件：
        # 新实例的巡检会重新判定，我们插手只会两边打架。
        if self._is_terminating:
            logger.debug("[什亭之枢] 插件已卸载，放弃本次巡检结果")
            return
        if verdict == PATROL_PROCESS:
            self._release(event)
            logger.info("[什亭之枢] 放行主脑 [patrol] 窗口判定：开口")
        else:
            self._block(event)
            logger.info("[什亭之枢] 拦截官方 LLM [patrol] 窗口判定：沉默")

    def _apply_decision(self, event: AstrMessageEvent, decision) -> None:
        """按判定结果放行或拦下这条事件。"""
        if decision.passthrough:
            logger.debug("[什亭之枢] 原样让路 [%s] %s", decision.source, decision.reason)
            return
        if decision.should_reply:
            self._release(event)
            logger.info("[什亭之枢] 放行主脑 [%s] %s", decision.source, decision.reason)
            return
        self._block(event)
        logger.info("[什亭之枢] 拦截官方 LLM [%s] %s", decision.source, decision.reason)

    def _release(self, event: AstrMessageEvent) -> None:
        """放行主脑。

        群聊里 AstrBot 可能判定未唤醒而不调用主脑，这里强行唤醒。
        """
        try:
            event.is_at_or_wake_command = True
        except Exception:
            pass

    def _block(self, event: AstrMessageEvent) -> None:
        """拦下这条事件：官方 LLM 完全不触发。"""
        event.stop_event()
        self._mark_gated(event)

    def _latest_window(self, chat_id: str) -> tuple[str, list[str]]:
        """逐条判定模式下的判据：该会话账本里最新的一条。"""
        rows = self.short_term.recent_of_chat(chat_id, 1)
        if not rows:
            return "", []
        return str(rows[-1]["chat_label"] or ""), [format_message_line(rows[-1])]

    # ══════════════════════════════════════════════════════════
    #  事件：召回并注入记忆
    # ══════════════════════════════════════════════════════════

    @filter.on_llm_request(priority=_PRIORITY_TOOLS)
    async def hide_builtin_send_tool(self, event: AstrMessageEvent, req):
        """把 AstrBot 内置的 send_message_to_user 从本次请求的可用工具里摘掉。

        为什么必须在这一层：全局的 `provider_manager.llm_tools.remove_func()` 对
        这个内置工具无效——调用不报错，日志显示「已移除」，但模型照样能调用
        （实测过）。它显然不在插件工具注册表里。逐请求摘除才是可靠的拦截点。

        为什么必须摘掉：该工具参数写错时**不报错，而是静默把消息发到当前会话**。
        已经造成两次错发——本该发给别人的话发给了正在对话的人。它还绕过我们的
        会话解析（平台前缀修正）与焦点校验。
        """
        if not self.decision_config.disable_builtin_send_tool:
            return
        try:
            func_tool = getattr(req, "func_tool", None)
            if func_tool is None:
                # 本次请求压根没有任何工具，无从摘除，也不算异常
                return
            remover = getattr(func_tool, "remove_tool", None)
            if not callable(remover):
                if not self._tool_removal_warned:
                    self._tool_removal_warned = True
                    logger.warning(
                        "[什亭之枢] 当前 AstrBot 版本不支持 req.func_tool.remove_tool，"
                        "无法摘除内置 send_message_to_user；请留意模型误用它导致错发"
                    )
                return
            remover(_BUILTIN_SEND_TOOL)
            if not self._tool_removal_logged:
                # 只报一次，既给排障一个明确信号，又不刷屏
                self._tool_removal_logged = True
                logger.info(
                    "[什亭之枢] 已从请求工具列表中摘除内置 %s，跨会话发言统一走 say",
                    _BUILTIN_SEND_TOOL,
                )
        except Exception as error:
            if not self._tool_removal_warned:
                self._tool_removal_warned = True
                logger.warning("[什亭之枢] 摘除内置发送工具失败: %s", error)

    @filter.on_llm_request(priority=_PRIORITY_INJECT)
    async def inject_memory_context(self, event: AstrMessageEvent, req):
        """在请求发给主模型前，把上下文换成一整条全局时间线，并注入记忆。

        必须排在会重写 req.contexts 的插件（天使之心，50）**之后**：它用自己
        重建的历史整体覆盖 contexts，我们后跑才能把上下文换成自己的时间线，
        而不是被它抹掉。排序语义见 _PRIORITY_INJECT。

        结果顺序就是模型的读取顺序：

            ① 永久记忆 → ② 长期记忆 →（全局时间线）→ 当前消息 → ③ 作息事件提示

        ①② 是**长期背景**，插在时间线之前；③ 是**时效事件**（刚被叫醒、刚起床），
        追加在最新处。放反了就会把"刚刚发生的事"讲成很久以前的事。
        """
        if not self._memory_active() and not self._schedule_active():
            return
        try:
            contexts = getattr(req, "contexts", None)
            if not isinstance(contexts, list):
                return

            if self._memory_active():
                chat_id = event.unified_msg_origin or ""
                self._inject_memory_blocks(event, req, chat_id)

            # 作息事件提示：独立于记忆系统——她睡着时发生了什么，与记忆是否
            # 启用无关，只要作息开着就该告诉她。
            self._inject_schedule_notice(req)
        except Exception as error:
            logger.error("[什亭之枢] 记忆注入失败: %s", error)

    def _inject_memory_blocks(
        self, event: AstrMessageEvent, req, chat_id: str
    ) -> None:
        """用全局时间线重建上下文，再把记忆块插到最前。"""
        # 1) 时间线：丢弃 AstrBot 的按会话历史，整体换成她的单一记忆流。
        #    必须在插入记忆块之前做，否则下一个请求会把我们注入的块也当历史。
        self._rebuild_timeline(req, chat_id)

        prefix: list[dict] = []

        # 2) 记忆块：① 永久记忆全量 → ② 榜前 auto_top 条正文 → ③ 其后 auto_index
        #    条只给关键词的索引。**自动插入不算「被想起」**：只有她主动 recall 才
        #    算，否则榜前那几条会互相把对方永远按在最前面，榜就冻住了。
        permanent = self.memory_store.permanents()
        bodies, index = self.memory_store.auto_injection()
        for block in build_memory_blocks(
            permanent=permanent,
            bodies=bodies,
            index=index,
            token_budget=self.memory_config.recall_token_budget,
        ):
            prefix.append(build_injected_entry(block))

        if prefix:
            req.contexts = prefix + req.contexts

    def _rebuild_timeline(self, req, chat_id: str) -> None:
        """把整条时间线写回 req.contexts。

        AstrBot 的历史是**按会话隔离**的，而她是同一个人：切会话不该换记忆。
        所以这里不再叠加，而是整体替换——她读到的永远是自己一天的连续经历，
        只有最后那条消息换了个人。

        出任何问题都退回 AstrBot 原生历史（不清空 contexts），宁可上下文
        不符合预期，也不要让她面对一片空白。
        """
        try:
            entries = self.short_term.timeline_entries(
                self.memory_config.timeline_max_messages
            )
            rendered = build_timeline(
                entries,
                current_prompt=str(getattr(req, "prompt", "") or ""),
                token_budget=self.memory_config.timeline_token_budget,
            )
            req.contexts = rendered
            logger.debug(
                "[什亭之枢] 时间线重建 | %s | %d 条",
                chat_id,
                len(rendered),
            )
        except Exception as error:
            logger.error("[什亭之枢] 时间线重建失败，保留原生上下文: %s", error)

    def _notice_kind(self, state: dict) -> str:
        """待注入的作息事件类型；没有则返回空串。

        起床优先于叫醒：起床发生在窗口末尾，若那一刻还挂着一条没来得及注入的
        叫醒提示，该说的也是「你睡醒了」而不是「你被叫醒了」。

        两种提示共用 `wake_notice_fired_at` 作为「已注入」的水位线：叫醒只在
        `awakened_at` 高于它时才算待注入，注入后水位线抬高，自然就不重复了。
        """
        if int(state.get("wake_notice_pending") or 0) == 1:
            return NOTICE_WAKE
        awakened = float(state.get("awakened_at") or 0.0)
        fired = float(state.get("wake_notice_fired_at") or 0.0)
        if awakened > fired:
            return NOTICE_AWAKENED
        return ""

    def _inject_schedule_notice(self, req) -> None:
        """把作息事件提示追加到上下文最新处，并消费待注入标记。

        必须持久化「是否已注入」：注入块每轮请求都会重建，不记标记的话这条
        提示会在她醒来后的每一轮对话里重复出现。
        """
        if not self._schedule_active():
            return
        try:
            state = self.sleep_state.load()
            kind = self._notice_kind(state)
            if not kind:
                return
            contexts = getattr(req, "contexts", None)
            if not isinstance(contexts, list):
                return

            now = time.time()
            sleep_at = 0.0
            total = 0
            digest: list[tuple[str, int, str]] = []
            if kind == NOTICE_WAKE:
                # 睡眠期消息的起点必须是**本轮**入睡时刻：plan_window() 此刻
                # 已经翻到下一轮了，只有 last_sleep_start() 找得回上一次入睡。
                sleep_at = last_sleep_start(
                    now,
                    self.schedule_config,
                    float(state.get("postponed_hours") or 0.0),
                )
                total, digest = self.short_term.sleep_digest(sleep_at)

            notice = build_schedule_notice(
                kind=kind,
                now=now,
                sleep_at=sleep_at,
                total=total,
                digest=digest,
            )
            contexts.append(build_injected_entry(notice))
            self.sleep_state.save(wake_notice_pending=0, wake_notice_fired_at=now)
            logger.info(
                "[什亭之枢] 注入作息提示 | %s | 睡眠期消息 %d 条",
                kind,
                total,
            )
        except Exception as error:
            logger.error("[什亭之枢] 作息提示注入失败: %s", error)

    @filter.on_llm_request(priority=_PRIORITY_IMAGE_GUARD)
    async def guard_broken_images(self, event: AstrMessageEvent, req):
        """发往模型前摘掉读不到的图片引用。

        与记忆系统解耦：一张坏图会让**整条请求**被上游 400 拒绝，AstrBot 随即
        把这轮切给备用模型——人格和工具全变，这与记忆是否启用毫无关系。
        排在所有上下文构建插件之后，才看得到它们放进来的图片。
        """
        self._strip_broken_images(req)

    # ══════════════════════════════════════════════════════════
    #  事件：发送后收口
    # ══════════════════════════════════════════════════════════

    @filter.on_decorating_result(priority=_PRIORITY_DECORATE)
    async def drop_origin_reply_after_leaving(self, event: AstrMessageEvent):
        """她本轮已前往别的会话时，丢掉原会话的收尾发言。

        没有这一步，她就会「走过去说了一句，再跑回来汇报一句」——像隔空喊话
        加复命，而不是一个真的走过去说话的人。她应该说在那边，不是说在这边。

        注意这不影响她在调用 say **之前**说的话：那部分在工具调用时就已发出，
        所以「好，我这就去」这类过渡语照常送达，被丢掉的只是走之后的收尾汇报。

        放在极低优先级，让其它插件的结果装饰先跑完，我们最后再决定要不要发。
        """
        if not self._proactive_config().get("silence_origin_after_say", True):
            return
        if not get_extra(event, EXTRA_LEFT_TO):
            return
        try:
            result = event.get_result() if hasattr(event, "get_result") else None
            if result is None:
                return
            chain = getattr(result, "chain", None)
            if chain:
                result.chain = []
                logger.info(
                    "[什亭之枢] 她已前往 %s，原会话本轮发言已省略",
                    get_extra(event, EXTRA_LEFT_TO, ""),
                )
        except Exception as error:
            logger.error("[什亭之枢] 抑制原会话发言失败: %s", error)

    @filter.after_message_sent(priority=_PRIORITY_AFTER_SENT)
    async def after_message_sent(self, event: AstrMessageEvent):
        """记录她自己的回复，并在后台触发压缩与维护。"""
        if self._is_terminating or not self._memory_active():
            return
        try:
            content = extract_reply_text(event)
            if content:
                chat_kind, chat_label = describe_chat(event)
                self.short_term.add(
                    chat_id=event.unified_msg_origin or "",
                    chat_label=chat_label,
                    chat_kind=chat_kind,
                    sender_id=call_str(event, "get_self_id"),
                    sender_name="",
                    role="assistant",
                    content=content,
                )
        except Exception as error:
            logger.error("[什亭之枢] 记录回复失败: %s", error)

        # 维护放后台，绝不阻塞消息链路
        self._schedule_maintenance_if_due()

    # ══════════════════════════════════════════════════════════
    #  LLM 工具：跨会话穿梭
    # ══════════════════════════════════════════════════════════

    @filter.llm_tool(name="list_sessions")
    async def list_sessions(self, event: AstrMessageEvent, limit: int = 15) -> str:
        """
        列出你最近活跃过的会话。想知道「有哪些地方可以去」时调用。

        Args:
            limit(number): 最多列出多少个会话，默认15
        """
        if not self._memory_active():
            return "记忆系统未启用。"
        try:
            rows = self.short_term.list_known_chats(int(limit))
            focused_ids = {str(r["chat_id"]) for r in self.focus.list_focused(200)}
        except Exception as error:
            logger.error("[什亭之枢] 列出会话失败: %s", error)
            return f"列出会话失败：{error}"
        if not rows:
            return "你还没有见过任何会话。"
        now = time.time()
        lines = [
            f"- {row['chat_label']}｜会话ID {row['chat_id']}｜"
            f"最近 {format_age(now - float(row['last_at'] or now))}｜"
            f"{row['message_count']}条"
            f"{'｜关注中' if str(row['chat_id']) in focused_ids else '｜已离开'}"
            for row in rows
        ]
        return (
            "你见过的会话（发消息时把「会话ID」或「标签」填进 session 即可）：\n"
            + "\n".join(lines)
        )

    @filter.llm_tool(name="peek_session")
    async def peek_session(
        self, event: AstrMessageEvent, session: str, limit: int = 12
    ) -> str:
        """
        查看某个会话最近的对话内容。想知道某个群或某个人最近在聊什么时调用。

        Args:
            session(string): 会话标识，用 list_sessions 得到，也可直接填群号、QQ号或会话标签
            limit(number): 查看多少条，默认12，最多40
        """
        if not self._memory_active():
            return "记忆系统未启用。"
        try:
            limit = max(1, min(int(limit), 40))
            chat_id = self.short_term.resolve_chat(session)
            if not chat_id:
                return f"找不到会话「{session}」，可以先用 list_sessions 看看有哪些。"
            rows = self.short_term.recent_of_chat(chat_id, limit)
        except Exception as error:
            logger.error("[什亭之枢] 查看会话失败: %s", error)
            return f"查看会话失败：{error}"
        if not rows:
            return f"会话「{session}」还没有可看的记录。"
        return f"【{rows[0]['chat_label']}】最近的对话：\n" + "\n".join(
            format_message_line(row) for row in rows
        )

    @filter.llm_tool(name="say")
    async def say(
        self,
        event: AstrMessageEvent,
        session: str,
        text: str,
        target_type: str = "auto",
        force: bool = False,
    ) -> str:
        """
        前往某个会所说一句话。调用后你就「在」那里了，那边之后有人说话你会正常参与。

        主动开口前你需要了解那个地方的语境，否则会显得突兀。如果你最近没有关注过
        目标会话，本次调用不会真的发送，而是先把那个会话的最近对话给你看；
        读完后确认要说的话合适，带上 force=true 再调用一次即可发出。

        Args:
            session(string): 要去哪个会话，填群号、QQ号或会话标签（用 list_sessions 查看）
            text(string): 要说的话
            target_type(string): 可选，private（私聊）或 group（群聊）；不确定填 auto
            force(bool): 目标会话已不再关注时是否仍坚持发言，默认 false
        """
        proactive = self._proactive_config()
        if not proactive.get("enabled", True):
            return "错误：主动发言功能已关闭。"

        message = (text or "").strip()
        if not message:
            return "错误：text 不能为空。"

        resolved = self._resolve_target(event, target_type, session)
        if resolved is None:
            return (
                f"错误：无法确定会话「{session}」。"
                "可以先用 list_sessions 查看会话ID，或明确指定 target_type。"
            )
        umo, chat_kind, numeric_id = resolved
        target_key = f"{chat_kind}:{numeric_id}"

        if self._allowed_targets and target_key not in self._allowed_targets:
            return f"错误：目标 {target_key} 不在允许的白名单中"

        # 先看后发：没关注过那个地方就先别开口，把语境递给她再说
        gate = self._check_say_gate(event, umo, force)
        if gate is not None:
            return gate

        if not self._check_rate_limit(proactive):
            return "错误：主动发言过于频繁，请稍后再试"

        try:
            sent = await self.context.send_message(umo, _build_plain_chain(message))
        except Exception as error:
            logger.error("[什亭之枢] 发言失败 | %s | %s", umo, error)
            return f"发送失败：{error}"

        # AstrBot 找不到对应平台实例时只打警告并返回 False，不会抛异常。
        # 不检查返回值就会把失败当成功报给她，她会以为话已经带到了。
        if sent is False:
            logger.warning("[什亭之枢] 发言未被平台接受 | %s", umo)
            return (
                f"发送失败：平台未接受会话标识 {umo}。"
                "请确认目标会话是否正确，或先用 list_sessions 查看已知会话的会话ID。"
            )

        self._send_timestamps.append(time.time())
        # 她「去」了那里：刷新焦点，之后那边有人说话她会正常参与
        label = self.short_term.label_for_chat(umo)
        self.focus.touch(umo, label)
        # 标记「她本轮已经离开原会话」：收尾时据此不再回原会话汇报
        if umo != (event.unified_msg_origin or ""):
            set_extra(event, EXTRA_LEFT_TO, umo)
        # 写回短期账本。主动说出的话不在 AstrBot 的历史里，标 gated 让
        # 上下文档知道「这条主脑看不到」，避免她忘记自己说过。
        self.short_term.add(
            chat_id=umo,
            chat_label=label,
            chat_kind=chat_kind,
            sender_id="",
            sender_name="",
            role="assistant",
            content=message,
            gated=True,
        )
        logger.info(
            "[什亭之枢] 前往会话发言 | %s (%s) | %s", target_key, umo, message[:50]
        )
        return f"你已经在「{label or target_key}」说了一句：{message}"

    @filter.llm_tool(name="leave_session")
    async def leave_session(self, event: AstrMessageEvent, session: str) -> str:
        """
        离开某个会话。离开后你不再主动关注它、也不会主动在那里说话，
        直到那边又有人说话，或你再次前往。某个话题已经聊完、你不想再介入时调用。

        Args:
            session(string): 要离开的会话，填群号、QQ号或会话标签（用 list_sessions 查看）
        """
        if not self.enabled or self._is_terminating:
            return "插件未启用。"
        try:
            umo = self.short_term.resolve_chat(session)
            if not umo:
                return f"找不到会话「{session}」，可以先用 list_sessions 看看有哪些。"
            if not self.focus.leave(umo):
                return f"你本来就没有在关注「{session}」。"
            label = self.short_term.label_for_chat(umo)
        except Exception as error:
            logger.error("[什亭之枢] 离开会话失败: %s", error)
            return f"离开失败：{error}"
        logger.info("[什亭之枢] 主动离开会话 | %s", umo)
        return f"你已经离开「{label}」，不再主动关注那边的动静。"

    @filter.llm_tool(name="rest")
    async def rest(self, event: AstrMessageEvent) -> str:
        """
        暂时离开所有会话去休息。之后你不再主动关注任何地方，
        直到某个会话又有人说话把你唤醒。想安静一会儿时调用。
        """
        if not self.enabled or self._is_terminating:
            return "插件未启用。"
        try:
            count = self.focus.rest()
        except Exception as error:
            logger.error("[什亭之枢] 进入休息失败: %s", error)
            return f"休息失败：{error}"
        logger.info("[什亭之枢] 主动进入休息，离开 %d 个会话", count)
        return f"你去休息了，暂时离开了 {count} 个会话。有人再找你时你会醒过来。"

    # ══════════════════════════════════════════════════════════
    #  LLM 工具：主动记忆
    # ══════════════════════════════════════════════════════════

    @filter.llm_tool(name="remember")
    async def remember_memory(
        self,
        event: AstrMessageEvent,
        content: str,
        user_name: str = "",
        tags: str = "",
    ) -> str:
        """
        记住一条值得长期保存的信息。当对方明确要求你记住某事，
        或你判断这条信息长期有用时调用。写入的是永久记忆，不会遗忘。

        Args:
            content(string): 要记住的内容。写成客观陈述句并写明主语，不超过60字，例如「小明不喜欢猫」
            user_name(string): 这条信息属于谁，无法判断则留空
            tags(string): 分类标签，多个用逗号分隔，可留空
        """
        if not self._memory_active():
            return "记忆系统未启用，无法记录。"
        text = (content or "").strip()
        if not text:
            return "错误：content 不能为空。"
        if len(text) > 120:
            return "错误：content 过长，请压缩到 120 字以内。"
        try:
            name = (user_name or "").strip()
            user_id = (call_str(event, "get_sender_id") or None) if name else None
            tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]
            memory_id = self.memory_store.add(
                content=text,
                tags=tag_list,
                chat_id=event.unified_msg_origin or None,
                user_id=user_id,
                user_name=name or None,
                manual=True,
            )
        except Exception as error:
            logger.error("[什亭之枢] 主动记忆失败: %s", error)
            return f"记录失败：{error}"
        logger.info("[什亭之枢] 主动记忆 #%d: %s", memory_id, text[:40])
        return f"已永久记住（#{memory_id}）：{text}"

    @filter.llm_tool(name="recall")
    async def recall_memory(self, event: AstrMessageEvent, query: str) -> str:
        """
        回想一件你已经记不清细节的旧事。
        当你隐约觉得「好像有这么回事」、记忆里只剩个大概，
        或当前话题碰到某个熟悉但模糊的人、事、地点时调用，先把细节想清楚再开口。

        Args:
            query(string): 要回想的内容关键词
        """
        if not self._memory_active():
            return "记忆系统未启用，无法回忆。"
        text = (query or "").strip()
        if not text:
            return "错误：query 不能为空。"
        try:
            recalled = self.memory_store.recall(
                query_text=text,
                user_id=call_str(event, "get_sender_id") or None,
                top_k=self.memory_config.recall_top_k,
                user_top_k=self.memory_config.recall_user_top_k,
                min_score=0.0,  # 主动回忆不设阈值，由模型自己判断是否可用
            )
        except Exception as error:
            logger.error("[什亭之枢] 主动回忆失败: %s", error)
            return f"回忆失败：{error}"
        if not recalled:
            return "想了半天，没想起来相关的。"
        return "你想起来了一些事：\n" + "\n".join(
            f"- {item.memory.content}" for item in recalled
        )

    @filter.llm_tool(name="forget")
    async def forget_memory(self, event: AstrMessageEvent, memory: str) -> str:
        """
        忘掉一条记忆。当对方明确要求你别再记得某件事时调用。
        永久记忆同样会被忘掉；忘掉的内容不会被重新记住。

        Args:
            memory(string): 要忘掉的那件事，用关键词或一句话描述，例如「小明的生日」
        """
        if not self._memory_active():
            return "记忆系统未启用，无法遗忘。"
        text = (memory or "").strip()
        if not text:
            return "错误：memory 不能为空。"
        try:
            candidates = self.memory_store.search_by_keywords(text, limit=5)
        except Exception as error:
            logger.error("[什亭之枢] 查找待遗忘记忆失败: %s", error)
            return f"遗忘失败：{error}"
        if not candidates:
            return f"记忆里没有与「{text}」相关的内容。"

        target = candidates[0]
        try:
            self.memory_store.request_forget(target.id)
        except Exception as error:
            logger.error("[什亭之枢] 标记遗忘失败: %s", error)
            return f"遗忘失败：{error}"
        logger.info(
            "[什亭之枢] 按要求遗忘 #%d（%s）: %s",
            target.id,
            target.tier,
            target.content[:40],
        )
        others = "".join(f"\n- {m.content}" for m in candidates[1:3])
        tail = f"\n如果还指这几条，也请告诉我：{others}" if others else ""
        tier_label = "永久" if target.tier == "permanent" else "长期"
        return (
            f"已经把它从{tier_label}记忆里划掉了：{target.content}。"
            f"它不会再被你想起，下一次整理时会彻底消失。{tail}"
        )

    # ══════════════════════════════════════════════════════════
    #  官方历史回填
    # ══════════════════════════════════════════════════════════

    def _schedule_image_caption(self, event: AstrMessageEvent, message_id: int) -> None:
        """把图片转述丢到后台。

        不能在这里等视觉模型：记账是事件链的第一环，卡住它会把后面所有插件
        的处理一起拖慢。引用必须**当场**取出来——图片的临时文件、链接都可能
        在我们睡一觉之后失效。
        """
        if not self.memory_config.vision_enabled or not message_id:
            return
        refs = ImageCaptioner.collect_refs(
            event, self.memory_config.vision_max_images
        )
        if not refs:
            return
        self._spawn(self._caption_message(message_id, refs))

    async def _caption_message(self, message_id: int, refs: list[str]) -> None:
        """转述回来就回填账本，她与记忆页面便都能看到图里是什么了。"""
        try:
            caption = await self.captioner.describe(refs)
        except Exception as error:
            logger.error("[什亭之枢] 图片转述异常: %s", error)
            return
        if not caption:
            return
        self.short_term.set_image_caption(message_id, caption)
        logger.debug("[什亭之枢] 图片转述完成 | %s", caption[:60])

    # ══════════════════════════════════════════════════════════
    #  记忆维护
    # ══════════════════════════════════════════════════════════

    async def run_maintenance(self) -> dict:
        """执行一次白天的惰性维护（短期压缩 / 降级 / 遗忘 / 晋升 / 合并）。

        **不含收口与归档**：这条路径是「发送消息后」的周期性兜底，宁可少做也
        不能在清醒状态下清账本（那等于当场失忆）。完整整理走 `_run_curation()`。

        短期压缩放在这里，是为了白天短期层涨过阈值时能就地压一轮，
        别让上下文一直涨到晚上。
        """
        try:
            stats = await self.maintainer.run(
                curate_budget=self.schedule_config.curate_max_per_run
            )
            condense_stats = await self.maintainer.condense_short_term()
            stats["condensed_folded"] = condense_stats.get("folded", 0)
        except Exception as error:
            logger.error("[什亭之枢] 记忆维护失败: %s", error)
            stats = {}
        # 焦点记录也在这里顺手清理，避免表长期膨胀
        try:
            self.focus.prune()
        except Exception as error:
            logger.error("[什亭之枢] 焦点记录清理失败: %s", error)
        return stats

    async def _run_curation(self, *, full_reset: bool) -> dict:
        """深睡期的一次完整整理（§6，顺序固定）。

            1 降级 → 遗忘 → 晋升 → 合并（maintainer.run，作用于原有那张榜）
            2 短期收口：把短期层压成一条「当天摘要」+ 若干长期记忆
            3 清空短期层（归档原始消息 + 收起摘要）
            4 保留期清理

        **收口必须在清空之前**：短期层一清空，没压过的内容就永远进不了长期记忆。
        所以 `collapse.failed` 时一律跳过第 3 步并告警。

        收口放在维护之后，是为了让当天摘要**确定性地落在榜首**：它最后写入，
        同刻并列时 id 最大。榜序即「她记得最牢的」，当天摘要必须是第一条。

        收口是「短期层 → 长期层」唯一的通道（逐批提炼已并入它），所以白天
        攒下的、睡觉时暂存刚放行的，全都在这一步被处理一遍。
        """
        if self._curating or self._is_terminating:
            return {}
        self._curating = True
        started = time.time()
        try:
            cfg = self.schedule_config
            stats = await self.maintainer.run(curate_budget=cfg.curate_max_per_run)
            if self._is_terminating:
                return {}
            collapse = await self.maintainer.collapse_short_term(
                max_rounds=cfg.collapse_max_rounds
            )
            stats["condensed_entries"] = collapse.get("entries", 0)
            stats["condensed_rounds"] = collapse.get("rounds", 0)
            stats["condensed_folded"] = collapse.get("folded", 0)
            stats["day_summary"] = 1 if collapse.get("day_summary") else 0

            # 清空的硬保护：收口没成功就**不清**。否则那一天的内容会静默消失，
            # 而且不报任何错——这正是「没整理完就不许清账本」的由来。
            reason = ""
            if not full_reset:
                reason = "本次是兜底整理（清醒状态下不清账本）"
            elif collapse.get("failed"):
                reason = "短期收口未完成，原始条目还没进长期记忆"
            elif not self.extractor.available:
                reason = "未配置压缩模型"
            elif not cfg.daily_reset_enabled:
                reason = "每日清空已关闭"
            if reason:
                stats["reset_skipped"] = reason
                logger.warning("[什亭之枢] 跳过清空短期层：%s", reason)
            else:
                archived, retired = self.short_term.clear_short_term(time.time())
                stats["archived_messages"] = archived
                stats["retired_summaries"] = retired
                stats["purged_expired"] = self.short_term.purge_expired(
                    cfg.archive_retention_days
                )

            self.sleep_state.save(last_curation_at=time.time())
            logger.info(
                "[什亭之枢] 记忆整理完成 用时%.0fms | 收口%d条→%d轮 当天摘要%d "
                "降级%d(重写%d) 遗忘%d 晋升%d 合并%d簇 | 归档%d条 收起摘要%d条%s",
                (time.time() - started) * 1000,
                stats.get("condensed_entries", 0),
                stats.get("condensed_rounds", 0),
                stats.get("day_summary", 0),
                stats.get("demoted", 0),
                stats.get("compressed", 0),
                stats.get("forgotten", 0),
                stats.get("promoted", 0),
                stats.get("merge_clusters", 0),
                stats.get("archived_messages", 0),
                stats.get("retired_summaries", 0),
                f" | 保留期清理{stats['purged_expired']}条"
                if stats.get("purged_expired")
                else "",
            )
            return stats
        except Exception as error:
            logger.error("[什亭之枢] 记忆整理失败: %s", error)
            return {}
        finally:
            self._curating = False

    def _schedule_maintenance_if_due(self) -> None:
        if not self.memory_config.maintenance_enabled:
            return
        now = time.time()
        if now - self._last_maintenance_at < _MAINTENANCE_FALLBACK_SECONDS:
            return
        self._last_maintenance_at = now
        self._spawn(self.run_maintenance())

    # ══════════════════════════════════════════════════════════
    #  生命周期
    # ══════════════════════════════════════════════════════════

    def on_config_saved(self):
        """WebUI 保存配置后热生效。"""
        self.enabled = bool(self.config.get("enabled", True))
        self.memory_config = MemoryConfig.from_dict(self.config)
        self.decision_config = DecisionConfig.from_dict(self.config)
        self.focus_config = FocusConfig.from_dict(self.config)
        # 作息配置也必须重读，否则改了入睡/起床时刻要重载插件才生效
        self.schedule_config = ScheduleConfig.from_dict(self.config)
        self.short_term.update_config(self.memory_config)
        self.memory_store.update_config(self.memory_config)
        self.extractor.update_config(self.memory_config)
        self.maintainer.update_config(self.memory_config)
        self.decider.update_config(self.decision_config, self.memory_config)
        self.captioner.update_config(self.memory_config)
        self.patrol.update_config(self.decision_config)
        self.focus.update_config(self.focus_config)
        self._allowed_targets = self._load_allowed_targets()
        # 作息：换新配置后重建计数窗口并立即重算快照，不必等下一个 tick
        self._wake_signals = WakeSignalCounter(
            self.schedule_config.wake_signal_window_minutes * 60,
            self.schedule_config.wake_signal_threshold,
        )
        self._refresh_sleep_window()
        self._ensure_sleep_tick()
        logger.info(
            "[什亭之枢] 配置已热更新 | 接管=%s 决策模型=%s | 作息=%s 当前=%s",
            "开" if self.decision_config.enabled else "关",
            self._decision_provider_label(),
            "启用" if self.schedule_config.enabled else "停用",
            self._sleep_phase(),
        )

    async def terminate(self):
        """插件卸载：先收口后台任务，再关记忆库。"""
        self._is_terminating = True
        pending = [task for task in self._background_tasks if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._background_tasks.clear()
        # 扣押中的窗口事件一律按拦下收场，否则它们会一直挂在等待里
        await self.patrol.cleanup()
        self.db.close()
        logger.info("[什亭之枢] 已卸载，记忆库已关闭")

    # ══════════════════════════════════════════════════════════
    #  内部工具
    # ══════════════════════════════════════════════════════════

    def _memory_active(self) -> bool:
        return self.enabled and self.memory_config.enabled and not self._is_terminating

    def _gate_active(self) -> bool:
        return self.enabled and self.decision_config.enabled and not self._is_terminating

    def _decision_provider_label(self) -> str:
        provider_id = (
            self.decision_config.provider_id or self.memory_config.extract_provider_id
        )
        return provider_id or "未配置（走规则兜底）"

    def _mark_gated(self, event: AstrMessageEvent) -> None:
        """把被拦下的这条消息标记为主脑看不到，供跨会话语境补齐。"""
        try:
            message_id = int(get_extra(event, EXTRA_MESSAGE_ID, 0) or 0)
        except (TypeError, ValueError):
            return
        if message_id:
            self.short_term.mark_gated([message_id])

    def _build_query_text(self, chat_id: str) -> str:
        """召回查询词 = 当前会话最近的几条消息。"""
        rows = self.short_term.recent_of_chat(
            chat_id, self.memory_config.short_term_context_per_chat * 2
        )
        return "\n".join(
            str(row["content"]) for row in rows[-8:] if str(row["content"] or "").strip()
        )

    def _strip_broken_images(self, req) -> None:
        """请求发出前摘掉读不到的图片引用。

        上游只接受 base64 / http(s) 的图片 URL。本地图片由 AstrBot 在 provider
        层转 base64，一旦转换失败（文件被清理、路径失效、内容为空）它会**原样
        透传**，严格的上游直接 400 —— 而 AstrBot 随后会把这轮切给备用模型，
        语气突变、工具调用多半失效。一张坏图的代价不该是整个回合。
        """
        if not self.memory_config.strip_invalid_images:
            return
        dropped: list[str] = []
        try:
            contexts = getattr(req, "contexts", None)
            if isinstance(contexts, list):
                cleaned, gone = strip_invalid_images(contexts)
                if gone:
                    req.contexts = cleaned
                    dropped.extend(gone)
            urls = getattr(req, "image_urls", None)
            if isinstance(urls, list):
                cleaned_urls, gone_urls = strip_invalid_image_urls(urls)
                if gone_urls:
                    req.image_urls = cleaned_urls
                    dropped.extend(gone_urls)
        except Exception as error:
            logger.error("[什亭之枢] 清理失效图片引用失败: %s", error)
            return
        if dropped:
            logger.warning(
                "[什亭之枢] 已摘除 %d 个读不到的图片引用，避免整条请求被上游拒绝: %s",
                len(dropped),
                " | ".join(ref[:120] for ref in dropped[:3]),
            )

    def _resolve_target(
        self, event: AstrMessageEvent, target_type: str, target_id: str
    ) -> tuple[str, str, str] | None:
        """把 LLM 给出的目标解析成 (完整 umo, 会话类型, 数字 ID)。"""
        token = (target_id or "").strip()
        if not token:
            return None
        target_type = (target_type or "auto").strip().lower()

        # 非纯数字或显式 auto：先按已知会话解析（群号/QQ号/标签/完整 umo 都能命中）
        if target_type == "auto" or not token.isdigit():
            chat_id = self.short_term.resolve_chat(token)
            if chat_id:
                kind = "private" if is_private_chat(chat_id) else "group"
                return chat_id, kind, chat_id.rsplit(":", 1)[-1]

        # 纯数字且类型明确：用当前事件的平台实例 ID 拼 umo。
        # 注意两点：平台前缀必须取自本事件的 unified_msg_origin（见 platform_of），
        # MessageType 必须是 FriendMessage / GroupMessage，不是 private / group。
        if token.isdigit() and target_type in ("private", "group"):
            platform = platform_of(event)
            if not platform:
                return None
            message_type = "FriendMessage" if target_type == "private" else "GroupMessage"
            return f"{platform}:{message_type}:{token}", target_type, token

        # auto 且解析不出已知会话：不做猜测，让模型明确指定类型。
        # 猜错会把消息发到错误的会话，代价远高于多问一句。
        return None

    def _check_say_gate(
        self, event: AstrMessageEvent, umo: str, force: bool
    ) -> str | None:
        """主动发言前的闸：确认她在关注这个会话。返回 None 表示放行。

        不直接硬拒，而是把那边的最近对话递给她：她读完后带 force=true 重发即可。
        这样既保住了「先看语境再开口」，又不会让工具循环在「没资格」和
        「不知道该说什么」之间空转。
        """
        if force or not self.focus_config.require_focus_for_say:
            return None
        # 目标就是当前会话，她显然就在这儿
        if umo == (event.unified_msg_origin or ""):
            return None
        if self.focus.is_focused(umo):
            return None

        rows = self.short_term.recent_of_chat(umo, 12)
        label = self.short_term.label_for_chat(umo)
        if not rows:
            # 从没见过的会话，没有语境可读，不额外设障
            return None

        idle = self.focus.idle_seconds(umo)
        idle_desc = "很久之前" if idle == float("inf") else format_age(idle)
        preview = "\n".join(format_message_line(row) for row in rows)
        return (
            f"你已经有 {idle_desc} 没关注「{label}」了，现在并不在那里。\n"
            f"它最近的对话是：\n{preview}\n\n"
            "请先读懂这里的气氛，确认你要说的话合适，"
            "然后带上 force=true 重新调用 say 发出。"
        )

    def _proactive_config(self) -> dict:
        value = self.config.get("proactive")
        return value if isinstance(value, dict) else {}

    def _load_allowed_targets(self) -> set[str]:
        targets = self._proactive_config().get("allowed_targets")
        if not isinstance(targets, list):
            return set()
        return {str(target).strip() for target in targets if str(target).strip()}

    def _check_rate_limit(self, proactive: dict) -> bool:
        """滑动窗口限流：分别检查每分钟与每小时。"""
        try:
            per_minute = int(proactive.get("rate_limit_per_minute", 3) or 3)
            per_hour = int(proactive.get("rate_limit_per_hour", 20) or 20)
        except (TypeError, ValueError):
            per_minute, per_hour = 3, 20

        now = time.time()
        while self._send_timestamps and now - self._send_timestamps[0] > 3600:
            self._send_timestamps.popleft()
        if len(self._send_timestamps) >= per_hour:
            logger.warning("[什亭之枢] 触发每小时主动消息限制")
            return False
        if sum(1 for ts in self._send_timestamps if ts > now - 60) >= per_minute:
            logger.warning("[什亭之枢] 触发每分钟主动消息限制")
            return False
        return True

    def _spawn(self, coro) -> None:
        """登记后台任务：统一追踪，便于 terminate 时取消并等待。"""
        if self._is_terminating:
            return
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._on_background_task_done)

    def _on_background_task_done(self, task: asyncio.Task) -> None:
        self._background_tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error("[什亭之枢] 后台任务异常退出: %s", error)
