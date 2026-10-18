"""什亭之枢 —— 为智能体提供统一人格、跨会话记忆与主动行动能力。

## 五条链路

**拦放链路（接管官方 LLM）**
  所有消息 → 记入短期账本 + 刷新该会话焦点 → 决断层规则层
    ├─ 上游指令 / 私聊 / 被点名 / 无文本 → 立即判定
    └─ 群聊普通消息 → 巡检窗口：累积一段对话后统一判定
         ├─ 不回复 → stop_event()，官方 LLM 不触发，该会话进入静默
         └─ 回复 → 置位 is_at_or_wake_command，放行主脑生成回复

  不逐条判定的原因：单条消息无法反映「本轮问答是否已结束」；且单次耗时较长的
  判定若延迟数十秒，结果可能已与当前话题不符。

**作息链路（当前是否清醒）**
  阶段由时间与配置实时推导，仅持久化事实（窗口起止、叫醒时刻等）。
  深睡拦截全部消息；浅睡需累积唤醒信号；命令消息始终放行。

**焦点链路（当前在哪些会话）**
  可同时在多个会话中「在场」。收到某会话消息 → 该会话获得焦点；
  安静超过 N 分钟 → 焦点失效。
  焦点**只约束主动发言**，被动接话不受限制，任何消息均可触发唤醒。

**记忆链路**
  短期：跨会话原始消息总账，带来源标记，避免 A 群的消息回复到 B 群；
        条目超阈值时压成摘要，深睡期收口为「当天摘要」并清空
  长期：压缩时一并产出的条目，排列在「最后一次被想起」的位置榜上，
        尾部逐级降低精度，最终被遗忘
  永久：被反复使用或被明确要求记住的记忆，不参与榜序、不降级、不遗忘

**行动链路（跨会话操作）**
  list_sessions 查看会话 → say 发送消息 → leave_session / rest 离开
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

try:  # AstrBot 4.27+ 的临时内容块：仅用于单轮请求，不写入持久历史
    from astrbot.core.agent.message import TextPart
except ImportError:  # pragma: no cover - 旧版本退化为 _no_save 的 dict
    TextPart = None

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
from .core.memory.timeline import build_timeline, build_timeline_block
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
# 结果装饰：置于最后执行，待其它插件完成后由本插件决定是否发送
_PRIORITY_DECORATE = -100
# 发送后收口
_PRIORITY_AFTER_SENT = 100

# AstrBot 内置的消息发送工具。参数错误时会静默发送到当前会话，
# 因此由本插件逐请求移除；跨会话发言统一使用 say。
_BUILTIN_SEND_TOOL = "send_message_to_user"

# 休息状态机落地前的临时兜底：距上次维护超过该时长时，在回复后补跑一次
_MAINTENANCE_FALLBACK_SECONDS = 24 * 3600

# 短期压缩的检查节流：触发条件为条目总数，计数需要一次带索引查询。
# 记账属于热路径，两次检查之间至少间隔该时长；压缩本身可容忍该延迟。
_CONDENSE_CHECK_SECONDS = 30.0

# 作息检查节拍。60 秒足以覆盖入睡/起床的边界变化，且该 tick 仅执行纯算术；
# 耗时整理由深睡调度单独触发。
_SLEEP_TICK_SECONDS = 60.0

# 阶段的中文说法：日志、工具返回、记忆页面共用同一套，避免三处各写一份
_PHASE_LABELS = {AWAKE: "清醒", SLEEP_LIGHT: "浅睡", DEEP: "深睡"}


def _persona_dialogs(contexts) -> list[dict]:
    """取出 AstrBot 插入在上下文最前的人格**预设对话**。

    AstrBot 在 `_ensure_persona_and_skills()` 中将人格的 `begin_dialogs` 通过
    `req.contexts[:0] = ...` 插入最前，且每条都带 `_no_save`；随后才执行
    本插件的 `on_llm_request`。整体替换上下文时必须原样保留这些条目，
    否则人格预设对话会被静默移除。

    **人格说明本身不在 contexts 中**——它位于 `req.system_prompt`
    （`# Persona Instructions`），本插件不改写该字段。

    仅取**头部连续**的 `_no_save` 段：AstrBot 插入的是连续一段，而插件在
    其他位置追加的系统条目（如文件解析结果）不带该标记，不会被误判。
    """
    kept: list[dict] = []
    for item in contexts or []:
        if isinstance(item, dict) and item.get("_no_save"):
            kept.append(item)
            continue
        break
    return kept


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
        # 群聊巡检：将「每条消息调用一次模型」改为「累积一小段后统一判定」
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
        # 临时内容块不受支持时只告警一次
        self._temp_part_warned = False
        # 作息内存快照：睡眠门是热路径，不允许做 IO，所以窗口与宽限都读内存，
        # 由 tick / 配置保存 / 叫醒时刷新。
        self._sleep_window = None
        self._grace_until = 0.0
        self._sleep_tick_started = False
        # 整理是重活（多次模型调用 + 重写历史），同一时刻只允许一次
        self._curating = False
        # 短期压缩也要调模型，同样防重入；与 _curating 互斥，避免两边同时动账本
        self._condensing = False
        self._last_condense_check = 0.0

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

        # 先计算窗口，使睡眠门在首个 tick 之前即可工作；随后启动周期检查。
        self._refresh_sleep_window()
        self._ensure_sleep_tick()
        self._init_curation_clock()
        if not self.schedule_config.enabled:
            # 作息关闭后 tick 不再运行，暂存区将无人负责放行；此处兜底放行，
            # 否则睡眠期消息会永久停留在 staged=1（既不整理也不清理）。
            self.short_term.release_staged()

    # ══════════════════════════════════════════════════════════
    #  全局作息：状态推导入口
    # ══════════════════════════════════════════════════════════

    def _schedule_active(self) -> bool:
        return self.schedule_config.enabled and not self._is_terminating

    def _sleep_phase(self, now: float | None = None) -> str:
        """当前作息阶段。仅读取内存快照，不执行 IO。"""
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

    def sleep_snapshot(self) -> dict | None:
        """当前作息快照（供记忆页面展示）。作息关闭时返回 None。

        现算而不存中间态：阶段是 `phase_at()` 的纯函数结果，落库只会多一份
        可能过期的真值。公开（无下划线）是因为 `web_api` 只读接口要调它。
        """
        if not self.schedule_config.enabled or self._is_terminating:
            return None
        now = time.time()
        try:
            state = self.sleep_state.load()
            postponed = float(state.get("postponed_hours") or 0.0)
            window = plan_window(now, self.schedule_config, postponed)
            phase = phase_at(now, window, self.schedule_config)
            last = float(state.get("last_curation_at") or 0.0)
        except Exception as error:
            logger.error("[什亭之枢] 读取作息快照失败: %s", error)
            return None
        return {
            "phase": phase,
            "phase_label": _PHASE_LABELS.get(phase, phase),
            "sleep_at": window.sleep_at,
            "wake_at": window.wake_at,
            "postponed_hours": postponed,
            "last_curation_at": last,
        }

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
        """首次运行时将「上次整理」初始化为当前时刻。

        否则 `last_curation_at = 0` 会被判定为长期未整理，插件首次启动时
        会立即执行一次整理。
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
        """将收到的消息记入短期账本，并刷新该会话的焦点。

        不回复、不干预，仅执行记账。必须先于决断层执行：无论消息被放行
        还是被拦下，都需要进入记忆。被拦下的消息会被标记为 gated。
        """
        if self._is_terminating:
            return
        try:
            sender_id = call_str(event, "get_sender_id")
            # 只有确认 sender_id 非空且等于自己时才跳过，避免老版本缺少
            # get_self_id 时两边都取到空串而把正常消息全部丢掉
            if sender_id and sender_id == call_str(event, "get_self_id"):
                return  # 自己的消息由 after_message_sent 记录，避免重复入账
            content = event.get_message_outline() or ""
            if not content.strip():
                # 图片 / 表情等无文本消息：以占位描述记账，避免对话流出现空缺。
                # 决断层读取的仍是空文本，因此不会触发回复。
                content = describe_non_text(event)
            if not content.strip():
                return
            chat_kind, chat_label = describe_chat(event)
            chat_id = event.unified_msg_origin or ""

            # 睡眠期照常记账，但**不刷新焦点**：否则每个发言会话都会重新获得
            # 主动发言资格，抵消入睡时的 focus.rest()。
            # 宽限期内按实时消息处理，不计入睡眠期。
            now = time.time()
            sleeping = (
                self._sleep_phase(now) in SLEEP_PHASES and not self._in_grace(now)
            )
            if not sleeping:
                # 有消息的会话获得焦点。焦点仅约束主动发言，不影响是否接话。
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
                # 图片内容需要单独调用视觉模型：AstrBot 仅提供「[图片]」占位
                self._schedule_image_caption(event, message_id)
            self.memory_store.upsert_user(
                platform_of(event) or "unknown",
                sender_id,
                sender_name,
            )
            # 账本长度已变化，检查是否需要压缩（节流在函数内部，热路径仅增加一次比较）
            self._schedule_condense_if_due()
        except Exception as error:
            logger.error("[什亭之枢] 记录消息失败: %s", error)

    # ══════════════════════════════════════════════════════════
    #  事件：睡眠门（先于决断层）
    # ══════════════════════════════════════════════════════════

    @filter.event_message_type(filter.EventMessageType.ALL, priority=_PRIORITY_SLEEP)
    async def sleep_gate(self, event: AstrMessageEvent):
        """睡眠期间拦下所有**非命令**消息，且不进入任何模型判定。

        必须早于决断层（90）：决断层规则层会对私聊直接放行
        （`always_reply_private`），若睡眠判定晚于它，私聊消息会直接进入主脑，
        「睡眠期不调用模型」的约束将失效。

        判定顺序：

        1. 上游指令 —— 放行。需求是「忽略所有**非命令**消息」，命令必须穿透，
           否则睡眠期间无法执行 `/` 指令。放行不代表解除睡眠，也不开启宽限。
        2. 清醒 —— 交回原有链路。
        3. 宽限期内（刚被叫醒）—— 放行。
        4. 浅睡 —— 累计 @/私聊信号，达到阈值即叫醒，并放行触发叫醒的消息。
        5. 深睡 —— 一律拦下，不计数、不响应。

        整条链路零 token：仅读取时间轴与计数，不调用任何模型。
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
            # 睡眠门自身异常不应导致机器人整体失效：放行，交由原生链路处理
            logger.error("[什亭之枢] 睡眠门异常，已放行: %s", error)

    def _is_wake_signal(self, event, chat_id: str) -> bool:
        """唤醒信号仅认 @ 与私聊；群聊中的普通消息不算。"""
        if is_private_chat(chat_id):
            return True
        return bool(getattr(event, "is_at_or_wake_command", False))

    # ── 作息收口 ──────────────────────────────────────────

    def _on_fall_asleep(self) -> None:
        """入睡收口：收回全部焦点、清零唤醒信号。

        静默执行，不做睡前告别——主动广播「我睡了」会使该状态成为公开信息，
        正确表现应为「没有回应」。
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
            # 起床提示待注入：由作息事件注入通道消费，注入后清除标记。
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
        """被叫醒：开启清醒宽限，并使该会话恢复主动发言资格。"""
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
            # 跨入新一轮作息：先结算上一轮（若已过起床时间），再开启本轮。
            # 结算必须在 reset 之后执行，否则 wake_notice_pending 会被重置。
            previous_wake = float(state.get("wake_at") or 0.0)
            # 新一轮恢复基础时间表：推迟仅对当次作息生效。必须使用归零后的
            # 推迟量重新计算窗口；若直接写回已计算的 window，会将本次顺延
            # 带入次日，导致推迟量逐日累积。
            window = plan_window(now, self.schedule_config, 0.0)
            self._sleep_window = window
            self.sleep_state.reset_round(window, 0.0)
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

        # 整理节拍与阶段转换无关：即使本次 tick 未发生阶段变化，也需检查是否满足整理条件。
        self._maybe_curate(phase, window, now)

    def _maybe_curate(self, phase: str, window, now: float) -> None:
        """按节拍触发记忆整理（§9.1）。仅由 tick 调用。

        **主触发：本轮深睡窗口内尚未整理**（`last_curation_at < window.sleep_at`）。
        该判据直接对应当前睡眠轮次，不依赖「距上次满 24 小时」的绝对时间间隔。

        不使用 24 小时间隔的原因：该间隔以上次整理**完成**时刻为锚点。
        若插件在启动当晚 22:56 加载、01:30 进入深睡，间隔仅 2.5 小时，
        整夜都会被判据阻止；此外每次整理耗时还会使锚点逐日偏移。
        按「本轮窗口」判断只需确认本轮是否已整理，不会产生累积偏差。

        兜底触发：不在深睡且距上次整理超过 `curation_force_after_hours`。
        睡眠时长过短时深睡窗口可能不出现，缺少兜底将导致永不整理。

        所有守卫仅推迟执行，不会丢失：本次 tick 条件不满足时，下个 tick 重新检查。
        """
        cfg = self.schedule_config
        if (
            not cfg.curation_enabled
            or self._curating
            or self._condensing
            or not self._memory_active()
        ):
            return
        # 处于清醒宽限（刚被叫醒、正在回复）时不执行整理：收口会压缩当前
        # 正在使用的上下文，与兜底整理跳过收口的原因一致。宽限结束后再次
        # 检查；主判据基于本轮窗口，不会因此遗漏。
        if self._in_grace(now):
            return
        try:
            last = float(self.sleep_state.load().get("last_curation_at") or 0.0)
        except Exception as error:
            logger.error("[什亭之枢] 读取整理节拍失败: %s", error)
            return
        if phase == DEEP and last < float(window.sleep_at):
            logger.info(
                "[什亭之枢] 本轮深睡窗口尚未整理，开始整理（上次 %s）",
                time.strftime("%m-%d %H:%M", time.localtime(last)) if last else "从未",
            )
            self._spawn(self._run_curation(full_reset=True))
            return
        force_seconds = max(int(cfg.curation_force_after_hours), 0) * 3600
        if phase != DEEP and force_seconds and now - last >= force_seconds:
            logger.warning(
                "[什亭之枢] 已 %.0f 小时未整理记忆且当前不在深睡，"
                "执行兜底整理（不收口、不清空账本）",
                (now - last) / 3600.0,
            )
            self._spawn(self._run_curation(full_reset=False))

    # ══════════════════════════════════════════════════════════
    #  事件：拦放（接管官方 LLM）
    # ══════════════════════════════════════════════════════════

    @filter.event_message_type(filter.EventMessageType.ALL, priority=_PRIORITY_GATE)
    async def gate_official_llm(self, event: AstrMessageEvent):
        """在 AstrBot 官方 LLM 触发之前执行拦放判定。

        **规则层**可确定的（上游指令、私聊、被点名、空文本）在此直接处理；
        其余群聊普通消息交由**巡检器**：扣押事件，等待一段对话结束后统一判定
        是否需要接话，而非逐条调用模型。

        私聊为中间态：规则层仅判定需要回复，但回复前同样进入静默窗口
        （`_gate_private_debounce`），避免打断用户连续发言。

        需要回复 → 设置 `is_at_or_wake_command` 以唤醒主脑生成回复；
        不回复 → `stop_event()`，官方 LLM 不触发。
        """
        if not self._gate_active():
            return

        try:
            ruled = self.decider.rule_decide(event)
        except Exception as error:
            # 决断层异常不应影响机器人运行：放行并交由原生链路处理
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
        """私聊防抖：静默满 `private_debounce_seconds` 秒且无新消息后唤醒主脑。

        私聊无需模型判定是否回复，但需要等待：用户常将一句话拆成多条发送，
        立即回复会造成打断。窗口复用巡检器的静默模式：窗口内旧事件以 KILL
        结束，仅最后一条持有窗口；被 KILL 的消息仍记录在账本中，主脑通过
        全局时间线读取完整对话，只是等到该段结束后统一回复。
        """
        seconds = self._private_debounce_seconds()
        chat_id = event.unified_msg_origin or ""
        if seconds <= 0 or not chat_id:
            # 未配置防抖或取不到会话标识：立即放行
            self._apply_decision(event, ruled)
            return

        future = await self.patrol.hold(chat_id, event, silent=True)
        if future is None:
            # 仅在插件卸载时进入此分支，交由卸载流程处理
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
        """私聊防抖时长；配置不可用时按「不防抖」处理，优先立即回复而非静默延迟。"""
        try:
            return max(float(self.decision_config.private_debounce_seconds), 0.0)
        except (TypeError, ValueError):
            return 0.0

    async def _gate_by_patrol(self, event: AstrMessageEvent) -> None:
        """群聊普通消息：扣押进窗口，聚合一段后判定。"""
        chat_id = event.unified_msg_origin or ""

        if not self.decision_config.patrol_enabled:
            # 巡检关闭：退化为逐条判定，以账本中最新消息作为判据
            chat_label, lines, before = self._latest_window(chat_id)
            decision = await self.decider.judge_window(
                chat_id, chat_label, lines, before=before
            )
            if self._is_terminating:
                return
            self._apply_decision(event, decision)
            return

        future = await self.patrol.hold(chat_id, event)
        if future is None:
            # 处于冷却期：直接拦下，不开启窗口
            if not self._is_terminating:
                self._block(event)
            return

        verdict = await future
        # 等待窗口期间插件可能已被重载或卸载，此时不应继续处理该事件：
        # 新实例的巡检会重新判定，继续处理会导致两个实例冲突。
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

    def _latest_window(self, chat_id: str) -> tuple[str, list[str], list[str]]:
        """逐条判定模式下的判据：最新一条 + 它前面那几句。

        只看最新一条等于逼模型猜「不懂」「那他呢」指的是什么，
        所以把紧邻的前文一并带上，判定对象仍是最新这条。
        """
        try:
            limit = max(int(self.decision_config.context_messages), 0)
        except (TypeError, ValueError):
            limit = 6
        rows = self.short_term.recent_of_chat(chat_id, limit + 1)
        if not rows:
            return "", [], []
        latest = rows[-1]
        before = [format_message_line(row) for row in rows[:-1]]
        return str(latest["chat_label"] or ""), [format_message_line(latest)], before

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
        """在主模型请求前注入记忆与作息事件。

        默认（`memory.context_takeover=False`）**不修改 `req.contexts`**：
        沿用 AstrBot 原生按会话历史，记忆块与跨会话背景通过
        `req.extra_user_content_parts` 追加为临时内容块（`mark_as_temp`），
        只作用于本轮请求，不写入持久历史，也不会破坏原生前缀缓存。

        跨会话背景会补充其它会话的消息与摘要，以及当前会话中被拦下
        （未进入原生历史）的消息；原生历史为空时还会补上当前会话的全部条目。

        仅在显式开启接管时，才用本插件的全局时间线整体替换 `req.contexts`。
        该模式会丢弃原生按会话历史，并改变请求前缀，可能降低缓存命中率；
        时间线中的消息行已改为绝对时间戳，减轻前缀逐轮变化的问题。

        结果顺序即模型读取顺序：

            ① 永久记忆 → ② 长期记忆 → ③ 跨会话背景 →（原生历史/时间线）→ 作息事件提示
        """
        if not self._memory_active() and not self._schedule_active():
            return
        try:
            if self._memory_active():
                chat_id = event.unified_msg_origin or ""
                self._inject_memory_blocks(event, req, chat_id)

            # 作息事件提示独立于记忆系统：睡眠期间发生的事件与记忆是否启用无关，
            # 只要作息功能开启就应注入。
            self._inject_schedule_notice(req)
        except Exception as error:
            logger.error("[什亭之枢] 记忆注入失败: %s", error)

    def _inject_memory_blocks(
        self, event: AstrMessageEvent, req, chat_id: str
    ) -> None:
        """注入记忆块；接管模式下重建时间线，否则追加临时内容块。"""
        blocks: list[str] = []

        # ① 永久记忆全量 → ② 榜前 auto_top 条正文 → ③ 其后 auto_index
        #    条关键词索引。**自动插入不计入「被想起」**：仅主动 recall 才计入，
        #    否则榜前条目会相互提升排名，导致榜单停止更新。
        permanent = self.memory_store.permanents()
        bodies, index = self.memory_store.auto_injection()
        for block in build_memory_blocks(
            permanent=permanent,
            bodies=bodies,
            index=index,
            token_budget=self.memory_config.recall_token_budget,
        ):
            blocks.append(block)

        if self.memory_config.context_takeover:
            # 接管模式：先重建时间线，再把记忆块插到最前。
            # 必须在插入记忆块之前执行，否则下次请求会将注入块视为历史消息。
            self._rebuild_timeline(req, chat_id)
            if not isinstance(getattr(req, "contexts", None), list):
                return
            if blocks:
                req.contexts = [
                    build_injected_entry(block) for block in blocks
                ] + req.contexts
            return

        # 官方模式：不改动 req.contexts。原生历史已按会话维护，
        # 这里只追加长期记忆与跨会话连续性块。
        blocks.append(self._cross_session_block(event, req, chat_id))
        self._append_temp_parts(req, blocks)

    def _cross_session_block(
        self, event: AstrMessageEvent, req, chat_id: str
    ) -> str:
        """生成跨会话背景文本块（默认模式）。

        统一人格要求她记得别处发生的事，但原生历史只覆盖当前会话，因此这里补上
        其它会话的消息、摘要，以及当前会话中被拦下（未进入原生历史）的消息。

        取样按**会话配额**而非全局最新：一个刷屏的群聊不该把别的会话从背景里
        挤掉，否则切回那些会话时记忆就断了。
        """
        try:
            entries = self.short_term.timeline_entries_balanced(
                per_chat=self.memory_config.short_term_context_per_chat,
                max_chats=self.memory_config.short_term_context_chats,
                limit=self.memory_config.timeline_max_messages,
                current_chat_id=chat_id,
            )
        except Exception as error:
            logger.error("[什亭之枢] 跨会话背景读取失败: %s", error)
            return ""
        if not entries:
            return ""

        # 当前消息由 req.prompt 承载，若同时出现在背景块里会重复一次。
        current_id = int(get_extra(event, EXTRA_MESSAGE_ID, 0) or 0)
        if current_id:
            entries = [
                entry
                for entry in entries
                if entry.get("kind") != "message"
                or int(entry["row"]["id"] or 0) != current_id
            ]

        try:
            chat_label = describe_chat(event)[1]
        except Exception:
            chat_label = ""
        return build_timeline_block(
            entries,
            current_chat_id=chat_id,
            chat_label=chat_label,
            include_current=not self._has_native_history(req),
            token_budget=self.memory_config.timeline_token_budget,
        )

    @staticmethod
    def _has_native_history(req) -> bool:
        """原生历史里是否已有真实对话。

        人格预设对话由 AstrBot 以 `_no_save` 插在 contexts 最前，它们属于人格，
        不算经历。历史为空（新会话或被重置）时需要把当前会话的条目也补进背景块，
        否则主脑会完全失忆。
        """
        contexts = getattr(req, "contexts", None)
        if not isinstance(contexts, list):
            return False
        for item in contexts:
            if isinstance(item, dict) and not item.get("_no_save"):
                return True
        return False

    def _append_temp_parts(self, req, texts: list[str]) -> None:
        """把文本追加为仅本轮可见的临时内容块。

        AstrBot 会把这些内容拼接到当前用户消息之后；标记为临时后不会写入
        持久历史。若运行版本没有 `extra_user_content_parts`，退化为在
        `req.contexts` 末尾追加 `_no_save` 条目——只追加、不替换，
        已有前缀保持稳定，记忆能力与缓存命中都不受影响。
        """
        blocks = [str(text or "").strip() for text in texts]
        blocks = [block for block in blocks if block]
        if not blocks:
            return

        parts = getattr(req, "extra_user_content_parts", None)
        if isinstance(parts, list):
            for block in blocks:
                if TextPart is not None:
                    parts.append(TextPart(text=block).mark_as_temp())
                else:
                    parts.append({"type": "text", "text": block, "_no_save": True})
            return

        contexts = getattr(req, "contexts", None)
        if not isinstance(contexts, list):
            return
        if not self._temp_part_warned:
            self._temp_part_warned = True
            logger.warning(
                "[什亭之枢] 当前 AstrBot 不支持 extra_user_content_parts，"
                "已退化为在上下文末尾追加临时条目"
            )
        for block in blocks:
            contexts.append(
                {"role": "user", "content": block, "_no_save": True}
            )

    def _rebuild_timeline(self, req, chat_id: str) -> None:
        """将整条时间线写回 `req.contexts`（仅接管模式）。

        AstrBot 的历史按会话隔离，而接管模式采用统一人格：切换会话不应切换
        记忆。因此不做叠加，而是整体替换——主脑始终读取连续的经历，仅最后
        一条消息来自不同会话。消息行使用绝对时间戳，使已有前缀保持稳定。

        人格预设对话需要保留（见 `_persona_dialogs`）：它们属于稳定人格，
        而非经历，排在时间线之前。

        发生异常时保留 AstrBot 原生历史（不清空 contexts），优先保证上下文可用。
        """
        # 必须在替换前读取：这些条目随后会被整体替换
        persona = _persona_dialogs(getattr(req, "contexts", None))
        try:
            entries = self.short_term.timeline_entries(
                self.memory_config.timeline_max_messages
            )
            rendered = build_timeline(
                entries,
                current_prompt=str(getattr(req, "prompt", "") or ""),
                token_budget=self.memory_config.timeline_token_budget,
            )
            req.contexts = persona + rendered
            logger.debug(
                "[什亭之枢] 时间线重建 | %s | %d 条（人格预设 %d 条）",
                chat_id,
                len(rendered),
                len(persona),
            )
        except Exception as error:
            logger.error("[什亭之枢] 时间线重建失败，保留原生上下文: %s", error)

    def _notice_kind(self, state: dict) -> str:
        """待注入的作息事件类型；没有则返回空串。

        起床优先于叫醒：起床发生在窗口末尾，若该时刻仍存在未注入的叫醒提示，
        应提示的内容是「你睡醒了」而非「你被叫醒了」。

        两种提示共用 `wake_notice_fired_at` 作为「已注入」水位线：叫醒仅在
        `awakened_at` 高于该水位线时才算待注入，注入后抬高水位线即可去重。
        """
        if int(state.get("wake_notice_pending") or 0) == 1:
            return NOTICE_WAKE
        awakened = float(state.get("awakened_at") or 0.0)
        fired = float(state.get("wake_notice_fired_at") or 0.0)
        if awakened > fired:
            return NOTICE_AWAKENED
        return ""

    def _inject_schedule_notice(self, req) -> None:
        """将作息事件提示追加到上下文末尾，并消费待注入标记。

        必须持久化「是否已注入」：注入块每轮请求都会重建，若不记录标记，
        该提示会在醒来后的每一轮对话中重复出现。
        """
        if not self._schedule_active():
            return
        try:
            state = self.sleep_state.load()
            kind = self._notice_kind(state)
            if not kind:
                return

            now = time.time()
            sleep_at = 0.0
            total = 0
            digest: list[tuple[str, int, str]] = []
            if kind == NOTICE_WAKE:
                # 睡眠期消息的起点必须是**本轮**入睡时刻：plan_window() 此时
                # 已切换到下一轮，只有 last_sleep_start() 能取到上一次入睡时刻。
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
            if self.memory_config.context_takeover:
                contexts = getattr(req, "contexts", None)
                if not isinstance(contexts, list):
                    return
                contexts.append(build_injected_entry(notice))
            else:
                # 官方模式：作为临时内容块追加到当前用户消息之后，
                # 不改动原生按会话历史。
                self._append_temp_parts(req, [notice])
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
        """请求发往模型前摘除不可用的图片引用。

        与记忆系统解耦：单个失效图片会导致**整条请求**被上游以 400 拒绝，
        AstrBot 随即切换到备用模型，导致人格与工具全部改变，这与记忆是否启用
        无关。该处理器排在所有上下文构建插件之后，才能看到它们加入的图片。
        """
        self._strip_broken_images(req)

    # ══════════════════════════════════════════════════════════
    #  事件：发送后收口
    # ══════════════════════════════════════════════════════════

    @filter.on_decorating_result(priority=_PRIORITY_DECORATE)
    async def drop_origin_reply_after_leaving(self, event: AstrMessageEvent):
        """本轮已前往其他会话时，丢弃原会话的收尾发言。

        若缺少该步骤，会出现「前往目标会话发言后又返回原会话汇报」的行为，
        与真实的跨会话发言不符。发言应发生在目标会话，而非原会话。

        注意这不影响调用 `say` **之前**的发言：该部分在工具调用时已发送，
        因此「好，我这就去」一类过渡语仍会送达，仅丢弃离开后的收尾汇报。

        该处理器使用最低优先级，待其它插件的结果装饰完成后，再决定是否发送。
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
                    "[什亭之枢] 已前往 %s，原会话本轮发言已省略",
                    get_extra(event, EXTRA_LEFT_TO, ""),
                )
        except Exception as error:
            logger.error("[什亭之枢] 抑制原会话发言失败: %s", error)

    @filter.after_message_sent(priority=_PRIORITY_AFTER_SENT)
    async def after_message_sent(self, event: AstrMessageEvent):
        """记录助手自身的回复，并在后台触发压缩与维护。"""
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

        # 短期压缩按条数触发（阈值一到就压一轮），维护管线按小时节流
        self._schedule_condense_if_due()
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

    @filter.llm_tool(name="search_history")
    async def search_history(
        self,
        event: AstrMessageEvent,
        keyword: str = "",
        session: str = "",
        days: int = 7,
        limit: int = 20,
    ) -> str:
        """
        翻看以前的聊天记录，找回已经想不起来的细节。

        当你记不清某件事是不是真的发生过、想确认某人的原话，或需要一个
        具体时间点的细节时调用。它翻的是**原始记录**，包括已经不记得的那些。

        只在想确认旧事时用：不要拿它代替回忆，也不要把查到的内容当成
        本轮回复的主题。

        Args:
            keyword(string): 要找的关键词，可留空（留空就按时间取最近的记录）
            session(string): 在哪个会话说；留空表示当前会话，跨会话才填（用 list_sessions 查看）
            days(number): 往回查多少天，默认7，最多90
            limit(number): 最多看多少条，默认20，最多50
        """
        if not self._memory_active():
            return "记忆系统未启用。"
        text = (keyword or "").strip()
        try:
            limit = max(1, min(int(limit), 50))
            max_days = max(int(self.schedule_config.archive_retention_days), 1)
            days = max(1, min(int(days), max_days))
            token = (session or "").strip()
            if token:
                chat_id = self.short_term.resolve_chat(token)
                if not chat_id:
                    return f"找不到会话「{session}」，可以先用 list_sessions 看看有哪些。"
            else:
                # 默认仅查询当前会话：搜索全部群聊与私聊会导致结果超出可读范围
                chat_id = event.unified_msg_origin or ""
                if not chat_id:
                    return "当前会话标识不可用，请指定 session。"
            rows = self.short_term.search_history(
                chat_id=chat_id,
                keyword=text,
                since=time.time() - days * 86400,
                limit=limit,
            )
        except Exception as error:
            logger.error("[什亭之枢] 翻查历史失败: %s", error)
            return f"翻查失败：{error}"
        if not rows:
            scope = "当前会话" if not (session or "").strip() else f"「{session}」"
            what = f"含「{text}」的" if text else ""
            return f"{scope}近 {days} 天没有{what}记录。"

        label = str(rows[0]["chat_label"] or "") or chat_id
        what = f"含「{text}」的" if text else ""
        head = f"【{label}】近 {days} 天的{what}记录，共 {len(rows)} 条："
        logger.info(
            "[什亭之枢] 翻查历史 | %s | 关键词=%s | %d 天内 %d 条",
            chat_id,
            text or "-",
            days,
            len(rows),
        )
        return head + "\n" + "\n".join(format_message_line(row) for row in rows)

    @filter.llm_tool(name="delay_sleep")
    async def delay_sleep(self, event: AstrMessageEvent, hours: float) -> str:
        """
        今晚晚点睡。当你想把眼前这段聊完、或正做到一半不想收尾时调用。

        入睡与起床一起往后顺延，只对今晚这一轮有效，明天照常。已经躺下
        （浅睡 / 深睡）之后就不能再推迟了。

        Args:
            hours(number): 推迟几小时，0.5 到 6 之间
        """
        if not self._schedule_active():
            return "作息系统未启用，没法推迟。"
        try:
            hours = float(hours)
        except (TypeError, ValueError):
            return "错误：hours 必须是数字。"
        if not 0.5 <= hours <= 6:
            return "错误：hours 要在 0.5 到 6 小时之间。"

        now = time.time()
        cfg = self.schedule_config
        try:
            state = self.sleep_state.load()
            postponed = float(state.get("postponed_hours") or 0.0)
            window = plan_window(now, cfg, postponed)
            phase = phase_at(now, window, cfg)
        except Exception as error:
            logger.error("[什亭之枢] 推迟入睡前读取作息失败: %s", error)
            return f"推迟失败：{error}"

        if phase != AWAKE:
            return (
                f"你已经{_PHASE_LABELS.get(phase, '睡了')}了，"
                "现在推迟等于中途醒过来。今晚就这样吧。"
            )

        target = postponed + hours
        try:
            new_window = plan_window(now, cfg, target)
        except Exception as error:
            logger.error("[什亭之枢] 计算新作息窗口失败: %s", error)
            return f"推迟失败：{error}"

        # 起床时间受 wake_latest 限制时，实际可推迟时长可能小于请求值；
        # 必须如实返回，避免模型认为已推迟到目标时间而实际未达到。
        gained = (new_window.wake_at - window.wake_at) / 3600.0
        if gained < hours - 0.01:
            if gained <= 0.01:
                return (
                    f"已经到起床时间上限（{cfg.wake_latest}）了，再推迟就没得睡了。"
                )
            return f"起床最晚不能超过 {cfg.wake_latest}，现在最多还能推迟 {gained:.1f} 小时。"

        try:
            self.sleep_state.save(
                postponed_hours=target,
                sleep_at=new_window.sleep_at,
                wake_at=new_window.wake_at,
            )
            # 立即重算内存中的窗口快照，后续状态推导使用新窗口
            self._refresh_sleep_window()
        except Exception as error:
            logger.error("[什亭之枢] 保存推迟作息失败: %s", error)
            return f"推迟失败：{error}"

        def _clock(ts: float) -> str:
            return time.strftime("%H:%M", time.localtime(ts))

        logger.info(
            "[什亭之枢] 推迟入睡 | +%.1fh（累计 %.1fh）| %s 入睡 / %s 起床",
            hours,
            target,
            _clock(new_window.sleep_at),
            _clock(new_window.wake_at),
        )
        return (
            f"今晚推迟 {hours:.1f} 小时：{_clock(new_window.sleep_at)} 入睡，"
            f"{_clock(new_window.wake_at)} 起床。明天照常。"
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

        # 先看后发：未关注目标会话时先返回其近期对话，确认后再发言
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

        # AstrBot 找不到对应平台实例时仅记录警告并返回 False，不会抛出异常。
        # 若不检查返回值，发送失败会被误报为成功。
        if sent is False:
            logger.warning("[什亭之枢] 发言未被平台接受 | %s", umo)
            return (
                f"发送失败：平台未接受会话标识 {umo}。"
                "请确认目标会话是否正确，或先用 list_sessions 查看已知会话的会话ID。"
            )

        self._send_timestamps.append(time.time())
        # 已前往目标会话：刷新焦点，后续该会话的消息会正常参与
        label = self.short_term.label_for_chat(umo)
        self.focus.touch(umo, label)
        # 标记本轮已离开原会话，收尾时据此跳过原会话汇报
        if umo != (event.unified_msg_origin or ""):
            set_extra(event, EXTRA_LEFT_TO, umo)
        # 写回短期账本。主动发言不在 AstrBot 的历史中，标记 gated 供上下文
        # 构建时识别，避免模型遗漏自己说过的话。
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
        if self._is_terminating:
            return "插件正在重载，请稍后再试。"
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
        if self._is_terminating:
            return "插件正在重载，请稍后再试。"
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
    #  图片转述
    # ══════════════════════════════════════════════════════════

    def _schedule_image_caption(self, event: AstrMessageEvent, message_id: int) -> None:
        """将图片转述任务放入后台。

        不能在此等待视觉模型：记账位于事件链首环，阻塞会影响后续所有插件的
        处理。图片引用必须**当场**取出——临时文件与链接可能随后失效。
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
        """转述完成后回填账本，供判定、上下文与记忆页面使用。"""
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
        """执行一次白天的惰性维护（降级 / 遗忘 / 晋升 / 合并）。

        **不包含短期压缩、收口与清空**：

        - 收口与清空会移除当天时间线，仅能在深睡期执行，见 `_run_curation()`。
        - 短期压缩按**条数**单独触发（见 `_schedule_condense_if_due`）：其职责是
          控制上下文长度，应跟随条目数量触发；而该维护管线需要调用模型执行降级
          与合并，只能按小时节拍运行，混合后会使压缩受小时节拍限制。
        """
        try:
            stats = await self.maintainer.run(
                curate_budget=self.schedule_config.curate_max_per_run
            )
        except Exception as error:
            logger.error("[什亭之枢] 记忆维护失败: %s", error)
            stats = {}
        # 同时清理焦点记录，避免表长期膨胀
        try:
            self.focus.prune()
        except Exception as error:
            logger.error("[什亭之枢] 焦点记录清理失败: %s", error)
        return stats

    async def _run_curation(self, *, full_reset: bool) -> dict:
        """执行一次记忆整理（§6，顺序固定）。

        `full_reset=True`（深睡期）：
            1 降级 → 遗忘 → 晋升 → 合并（maintainer.run，作用于现有位置榜）
            2 短期收口：将短期层压缩为一条「当天摘要」与若干长期记忆
            3 清空短期层（归档原始消息 + 收起摘要）
            4 保留期清理

        `full_reset=False`（清醒期兜底整理，仅在睡眠窗口未出现时执行）：
            **仅执行第 1 步**，不执行收口。收口会移除当天时间线，与清空同样
            会导致当前上下文不可用。兜底策略的目标是减少操作、不影响当前上下文，
            长度控制由白天的 `condense_short_term()` 负责。

        **收口必须在清空之前**：短期层清空后，未压缩的内容将无法进入长期记忆。
        因此 `collapse.failed` 时一律跳过第 3 步并记录告警。

        收口安排在维护之后，使当天摘要**确定性地位于榜首**：它最后写入，
        同刻并列时 id 最大。榜单顺序代表记忆强度，当天摘要必须位于第一条。

        收口是「短期层 → 长期层」的唯一通道（逐批提炼已并入该步骤），
        白天积累与睡眠期暂存后放行的消息均在此步骤中处理。
        """
        if self._curating or self._condensing or self._is_terminating:
            return {}
        self._curating = True
        started = time.time()
        collapse: dict = {}
        try:
            cfg = self.schedule_config
            stats = await self.maintainer.run(curate_budget=cfg.curate_max_per_run)
            if self._is_terminating:
                return {}
            if full_reset:
                collapse = await self.maintainer.collapse_short_term(
                    max_rounds=cfg.collapse_max_rounds
                )
            stats["condensed_entries"] = collapse.get("entries", 0)
            stats["condensed_rounds"] = collapse.get("rounds", 0)
            stats["condensed_folded"] = collapse.get("folded", 0)
            stats["day_summary"] = 1 if collapse.get("day_summary") else 0

            # 清空保护：收口未成功时不执行清空。否则当天内容会静默丢失，
            # 且不会产生任何错误——这是「未完成整理不得清空账本」约束的来源。
            reason = ""
            if not full_reset:
                reason = "本次是兜底整理（清醒状态下不收口、不清账本）"
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

    def _schedule_condense_if_due(self) -> None:
        """短期层超过阈值时，将最旧的一批压缩一轮。

        与 `run_maintenance()` 分开触发：压缩的判据是**条数**而非时间——
        条目数超过阈值才执行，因此应跟随消息量触发，而不受小时节拍限制。
        调用点为「收到消息」与「发出消息」两处，两条路径都会增加账本长度。

        是否执行由 `condense_short_term()` 内部决定（先统计条目数，未达阈值
        直接返回）。本方法仅负责三件事：节流（减少热路径上的数据库查询）、
        防重入（模型调用进行中不再入队）、以及**不 await**（不阻塞消息链路）。
        """
        if not self._memory_active() or self._condensing or self._curating:
            return
        now = time.time()
        if now - self._last_condense_check < _CONDENSE_CHECK_SECONDS:
            return
        self._last_condense_check = now
        self._condensing = True
        self._spawn(self._condense_once())

    async def _condense_once(self) -> None:
        try:
            stats = await self.maintainer.condense_short_term()
            if stats.get("folded"):
                logger.info(
                    "[什亭之枢] 短期压缩触发（条目%d 折叠%d条）",
                    stats.get("entries", 0),
                    stats.get("folded", 0),
                )
        except Exception as error:
            logger.error("[什亭之枢] 短期压缩失败: %s", error)
        finally:
            self._condensing = False

    # ══════════════════════════════════════════════════════════
    #  生命周期
    # ══════════════════════════════════════════════════════════

    def on_config_saved(self):
        """WebUI 保存配置后热生效。"""
        self.memory_config = MemoryConfig.from_dict(self.config)
        self.decision_config = DecisionConfig.from_dict(self.config)
        self.focus_config = FocusConfig.from_dict(self.config)
        # 作息配置也需要重新读取，避免修改入睡/起床时间后必须重载插件才生效
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
        # 作息：使用新配置重建计数窗口并立即重算快照，无需等待下一个 tick
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
        # 扣押中的窗口事件统一按拦下处理，避免其永久处于等待状态
        await self.patrol.cleanup()
        self.db.close()
        logger.info("[什亭之枢] 已卸载，记忆库已关闭")

    # ══════════════════════════════════════════════════════════
    #  内部工具
    # ══════════════════════════════════════════════════════════

    def _memory_active(self) -> bool:
        return self.memory_config.enabled and not self._is_terminating

    def _gate_active(self) -> bool:
        return self.decision_config.enabled and not self._is_terminating

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
        """请求发出前摘除不可用的图片引用。

        上游仅接受 base64 / http(s) 形式的图片 URL。本地图片由 AstrBot 在
        provider 层转为 base64；一旦转换失败（文件被清理、路径失效、内容为空），
        引用会被**原样透传**，严格的上游会直接返回 400，AstrBot 随后将本轮
        切换到备用模型，导致语气突变、工具调用大概率失效。单个失效图片不应
        影响整个回合。
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
                "[什亭之枢] 已摘除 %d 个不可用的图片引用，避免整条请求被上游拒绝: %s",
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
        """主动发言前置检查：确认是否关注目标会话。返回 None 表示放行。

        不直接拒绝，而是返回目标会话的近期对话；确认后可带 `force=true`
        重新调用。该设计保留「先看语境再发言」的约束，同时避免工具在
        「无资格」与「无内容」之间反复调用。
        """
        if force or not self.focus_config.require_focus_for_say:
            return None
        # 目标为当前会话，直接放行
        if umo == (event.unified_msg_origin or ""):
            return None
        if self.focus.is_focused(umo):
            return None

        rows = self.short_term.recent_of_chat(umo, 12)
        label = self.short_term.label_for_chat(umo)
        if not rows:
            # 从未见过的会话没有可读语境，不额外限制
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
