"""什亭之枢 —— 让智能体拥有统一人格、跨会话记忆与主动行动能力。

## 四条链路

**拦放链路（接管官方 LLM）**
  所有消息 → 记入短期账本 + 刷新该会话焦点 → 决断层判定
    ├─ 不回 → stop_event()，AstrBot 官方 LLM 完全不触发
    └─ 要回 → 置位 is_at_or_wake_command，放行主脑生成回复

**焦点链路（她在哪）**
  她可以同时在多个会话里"在场"，像一个人同时开着几个聊天窗口。
  收到某会话的消息 → 该会话获得焦点；安静超过 N 分钟 → 焦点失效（她休息了）。
  焦点**只约束主动发言**：被动接话永远不受限，任何消息都能唤醒她。

**记忆链路**
  短期：跨会话原始消息总账，带来源标记，保证她不会把 A 群的话回到 B 群
  长期：由独立配置的轻量模型提炼的条目，会衰减、合并、遗忘
  永久：被反复用到或被明确要求记住的记忆，不衰减、不合并、不遗忘

**行动链路（跨会话穿梭）**
  list_sessions 看有哪些会话 → say 过去说一句 → leave_session / rest 离开
"""

from __future__ import annotations

import asyncio
import os
import time
from collections import deque

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Plain
from astrbot.api.star import Context, Star, register
from astrbot.core.star.star_tools import StarTools

from .core.config import DecisionConfig, FocusConfig, MemoryConfig
from .core.db import MemoryDB
from .core.decision import DecisionEngine
from .core.event_utils import (
    EXTRA_MESSAGE_ID,
    call_str,
    describe_chat,
    extract_reply_text,
    format_age,
    get_extra,
    is_private_chat,
    platform_of,
    set_extra,
)
from .core.extractor import MemoryExtractor
from .core.focus import FocusManager
from .core.history import ConversationHistoryReader, align_records
from .core.injector import (
    build_block,
    build_recent_context_block,
    format_for_log,
    inject_into_request,
)
from .core.maintenance import MemoryMaintainer
from .core.memory_store import MemoryStore
from .core.short_term import (
    HISTORY_SENDER_ID,
    ShortTermLedger,
    format_message_line,
)

# 决断层在事件链上的优先级：必须早于 AstrBot 的默认 LLM 阶段。
# 记账用 100、判定用 90，保证判定时当前消息已在账本里。
#
# 注意 90 是个偏「霸道」的取值：它会先于其它插件（priority < 90）的消息处理器
# 执行，被拦下的消息其它插件也看不到。这是「彻底接管」的代价。
# 若发现与其它插件冲突，把 _PRIORITY_GATE 调低（例如 -10，天使之心用的值）
# 即可让出优先级，只要仍高于默认 LLM 阶段就依然能拦住官方 LLM。
_PRIORITY_RECORD = 100
_PRIORITY_GATE = 90
# 上下文注入与发送后收口
_PRIORITY_INJECT = 50
_PRIORITY_AFTER_SENT = 100

# 休息状态机落地前的临时兜底：距上次维护超过该时长就在回复后补跑一次
_MAINTENANCE_FALLBACK_SECONDS = 24 * 3600


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

        # ── 存储与各层组件 ────────────────────────────────────
        data_dir = StarTools.get_data_dir("astrbot_plugin_shittim")
        self.db = MemoryDB(data_dir)
        self.short_term = ShortTermLedger(self.db, self.memory_config)
        self.memory_store = MemoryStore(self.db, self.memory_config)
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
        self.history_reader = ConversationHistoryReader(self.context)

        # ── 运行时状态 ────────────────────────────────────────
        self._background_tasks: set[asyncio.Task] = set()
        self._is_terminating = False
        self._extracting = False
        self._last_maintenance_at = 0.0
        self._send_timestamps: deque[float] = deque()
        self._allowed_targets = self._load_allowed_targets()
        # 已尝试过从官方历史回填的会话（每进程每个会话只试一次）
        self._backfilled: set[str] = set()

        tiers = self.memory_store.count_by_tier()
        logger.info(
            "[什亭之枢] 初始化完成 | 接管=%s 决策模型=%s | 记忆=%s 提炼模型=%s | "
            "短期%d条 长期%d条 永久%d条 | 关注中%d个会话",
            "开" if self.decision_config.enabled else "关",
            self._decision_provider_label(),
            "启用" if self.memory_config.enabled else "停用",
            self.memory_config.extract_provider_id or "未配置（自动提炼关闭）",
            self.short_term.count(),
            tiers.get("long", 0),
            tiers.get("permanent", 0),
            len(self.focus.list_focused(200)),
        )

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
                return
            chat_kind, chat_label = describe_chat(event)
            chat_id = event.unified_msg_origin or ""

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
            )
            if message_id:
                # 供决断层在拦下这条消息时回填标记
                set_extra(event, EXTRA_MESSAGE_ID, message_id)
            self.memory_store.upsert_user(
                platform_of(event) or "unknown",
                sender_id,
                sender_name,
            )
            # 顺手把插件不在场期间的官方历史补进来（每个会话只做一次）
            self._schedule_backfill(chat_id, chat_label, chat_kind)
        except Exception as error:
            logger.error("[什亭之枢] 记录消息失败: %s", error)

    # ══════════════════════════════════════════════════════════
    #  事件：拦放（接管官方 LLM）
    # ══════════════════════════════════════════════════════════

    @filter.event_message_type(filter.EventMessageType.ALL, priority=_PRIORITY_GATE)
    async def gate_official_llm(self, event: AstrMessageEvent):
        """在 AstrBot 官方 LLM 被触发之前做拦放判定。

        要回 → 置位 is_at_or_wake_command 强行唤醒主脑，让它生成回复；
        不回 → stop_event()，官方 LLM 完全不触发。
        """
        if not self._gate_active():
            return
        try:
            decision = await self.decider.decide(event)
        except Exception as error:
            # 决断层自己出问题不该让机器人陪葬：放行，交给原生链路
            logger.error("[什亭之枢] 决断层异常，已放行: %s", error)
            return

        if decision.passthrough:
            logger.debug("[什亭之枢] 原样让路 [%s] %s", decision.source, decision.reason)
            return

        if decision.should_reply:
            try:
                # 群聊里 AstrBot 可能判定未唤醒而不调用主脑，这里强行唤醒
                event.is_at_or_wake_command = True
            except Exception:
                pass
            logger.info("[什亭之枢] 放行主脑 [%s] %s", decision.source, decision.reason)
            return

        event.stop_event()
        self._mark_gated(event)
        logger.info("[什亭之枢] 拦截官方 LLM [%s] %s", decision.source, decision.reason)

    # ══════════════════════════════════════════════════════════
    #  事件：召回并注入记忆
    # ══════════════════════════════════════════════════════════

    @filter.on_llm_request(priority=_PRIORITY_INJECT)
    async def inject_memory_context(self, event: AstrMessageEvent, req):
        """在请求发给主模型前，注入跨会话近期上下文与相关长期记忆。

        priority 高于下游记忆类插件（常用 40），保证注入结果不会被它们的
        整体覆盖冲掉。
        """
        if not self._memory_active():
            return
        try:
            chat_id = event.unified_msg_origin or ""

            # 1) 跨会话近期原始上下文：让她知道「别处发生了什么」
            recent_text = self.short_term.build_recent_block(exclude_chat_id=chat_id)
            if recent_text:
                inject_into_request(req, build_recent_context_block(recent_text))

            # 2) 长期记忆双通道召回：消息相关 + 发送者相关
            recalled = self.memory_store.recall(
                query_text=self._build_query_text(chat_id),
                user_id=call_str(event, "get_sender_id") or None,
            )
            block, used_ids = build_block(
                recalled, self.memory_config.recall_token_budget
            )
            if block and inject_into_request(req, block):
                self.memory_store.mark_recalled(used_ids)
                logger.debug(
                    "[什亭之枢] 召回 %d 条记忆: %s",
                    len(used_ids),
                    format_for_log(recalled),
                )
        except Exception as error:
            logger.error("[什亭之枢] 记忆注入失败: %s", error)

    # ══════════════════════════════════════════════════════════
    #  事件：发送后收口
    # ══════════════════════════════════════════════════════════

    @filter.after_message_sent(priority=_PRIORITY_AFTER_SENT)
    async def after_message_sent(self, event: AstrMessageEvent):
        """记录她自己的回复，并在后台触发提炼与维护。"""
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

        # 提炼与维护都放后台，绝不阻塞消息链路
        self._schedule_extraction_if_needed()
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
            sent = await self.context.send_message(umo, [Plain(message)])
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
                strength=10.0,
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
        主动回忆与某个话题或某个人相关的记忆。
        当你需要确认某件旧事、某人的偏好或身份时调用。

        Args:
            query(string): 要回忆的内容关键词
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
            return "没有找到相关的记忆。"
        return "回忆到以下内容：\n" + "\n".join(
            f"- {item.memory.content}" for item in recalled
        )

    # ══════════════════════════════════════════════════════════
    #  官方历史回填
    # ══════════════════════════════════════════════════════════

    def _schedule_backfill(self, chat_id: str, chat_label: str, chat_kind: str) -> None:
        """每个会话在本次进程里只尝试回填一次。"""
        if not self.memory_config.backfill_astrbot_history:
            return
        if not chat_id or chat_id in self._backfilled:
            return
        if not self.history_reader.available:
            return
        # 先登记再提交任务，避免同一条消息触发两次并发回填
        self._backfilled.add(chat_id)
        self._spawn(self._run_backfill(chat_id, chat_label, chat_kind))

    async def _run_backfill(
        self, chat_id: str, chat_label: str, chat_kind: str
    ) -> None:
        """把 AstrBot 官方历史里我们没记过的对话补进短期账本。

        我们的账本只记「插件在场时」的消息；插件装上之前、被禁用或出错期间的
        对话只在 AstrBot 自己那里。回填让她的短期记忆不再有这段空白。

        时间戳不是随便塞的：沿 AstrBot 历史顺序与账本做锚点对齐，让新记录落到
        真实的时间位置。这既决定显示顺序，也决定它算不算「比实时消息更早」。
        """
        try:
            records = await self.history_reader.read(
                chat_id, self.memory_config.backfill_max_messages
            )
            if not records:
                return

            # 对齐到账本时间轴：既得到真实时间戳，也决定哪些该由我们兜底注入
            placed = align_records(
                records, self.short_term.all_of_chat(chat_id), time.time()
            )
            if not placed:
                return

            rows = [
                {
                    "chat_id": chat_id,
                    "chat_label": self.short_term.label_for_chat(chat_id) or chat_label,
                    "chat_kind": chat_kind,
                    # 官方历史不存发送者，群聊里无法还原是谁说的
                    "sender_id": HISTORY_SENDER_ID if record["role"] == "user" else "",
                    "sender_name": "",
                    "role": record["role"],
                    "content": record["content"],
                    "created_at": timestamp,
                    # 比账本里一切记录都早的那批标 gated：AstrBot 一旦压缩掉它们，
                    # 主脑就再也看不到，必须由我们兜底注入。
                    # 落在实时消息之间的，AstrBot 自己会保留，不必重复注入。
                    "gated": is_older_than_known,
                }
                for record, timestamp, is_older_than_known in placed
            ]
            # extracted=True：回填是历史沉淀，不参与长期记忆提炼。
            # 否则一次几百条回填会突然触发大量模型调用。
            inserted = self.short_term.add_batch(rows, extracted=True)
            guarded = sum(1 for _, _, older in placed if older)
            logger.info(
                "[什亭之枢] 官方历史回填完成 | %s | 读到%d条 补入%d条（其中%d条标为兜底）",
                chat_id,
                len(records),
                inserted,
                guarded,
            )
        except Exception as error:
            logger.error("[什亭之枢] 官方历史回填失败 | %s | %s", chat_id, error)

    # ══════════════════════════════════════════════════════════
    #  记忆维护
    # ══════════════════════════════════════════════════════════

    async def run_maintenance(self) -> dict:
        """执行一次完整记忆维护。

        设计上应由「她进入休息」这一时机触发。在休息状态机落地之前，
        这里由 after_message_sent 上的 24 小时惰性兜底调用，
        保证维护不会因为状态机未完成而永远不执行。状态机完成后，
        在进入休息的分支里直接 await 本方法、删掉兜底即可。
        """
        try:
            stats = await self.maintainer.run()
        except Exception as error:
            logger.error("[什亭之枢] 记忆维护失败: %s", error)
            stats = {}
        # 焦点记录也在这里顺手清理，避免表长期膨胀
        try:
            self.focus.prune()
        except Exception as error:
            logger.error("[什亭之枢] 焦点记录清理失败: %s", error)
        return stats

    def _schedule_maintenance_if_due(self) -> None:
        if not self.memory_config.maintenance_enabled:
            return
        now = time.time()
        if now - self._last_maintenance_at < _MAINTENANCE_FALLBACK_SECONDS:
            return
        self._last_maintenance_at = now
        self._spawn(self.run_maintenance())

    def _schedule_extraction_if_needed(self) -> None:
        if self._extracting or not self.extractor.available:
            return
        if self.short_term.pending_count() < self.memory_config.extract_batch_size:
            return
        self._spawn(self._run_extraction())

    async def _run_extraction(self) -> None:
        """把累积的短期消息提炼成长期记忆。"""
        self._extracting = True
        try:
            rows = self.short_term.pending(self.memory_config.extract_batch_size)
            if not rows:
                return
            records_text = "\n".join(format_message_line(row) for row in rows)
            items = await self.extractor.extract(records_text)

            name_to_id = {
                str(row["sender_name"]): str(row["sender_id"])
                for row in rows
                if str(row["sender_name"] or "").strip()
            }
            chat_ids = {str(row["chat_id"]) for row in rows}
            single_chat = chat_ids.pop() if len(chat_ids) == 1 else None
            chat_label = str(rows[-1]["chat_label"]) if single_chat else None

            created = 0
            for item in items:
                name = item.get("user_name") or ""
                _, is_new = self.memory_store.reinforce_or_add(
                    content=item["content"],
                    source_quote=item["source_quote"],
                    tags=item["tags"],
                    keywords=item["keywords"],
                    chat_id=single_chat,
                    chat_label=chat_label,
                    user_id=name_to_id.get(name),
                    user_name=name or None,
                    strength=item["strength"],
                )
                created += 1 if is_new else 0

            # 无论是否提炼出内容都标记已处理，否则这批消息会被反复重试
            self.short_term.mark_extracted([int(row["id"]) for row in rows])
            logger.info(
                "[什亭之枢] 记忆提炼完成: %d条消息 → 新增%d条 强化%d条",
                len(rows),
                created,
                len(items) - created,
            )
        except Exception as error:
            logger.error("[什亭之枢] 记忆提炼失败: %s", error)
        finally:
            self._extracting = False

    # ══════════════════════════════════════════════════════════
    #  生命周期
    # ══════════════════════════════════════════════════════════

    def on_config_saved(self):
        """WebUI 保存配置后热生效。"""
        self.enabled = bool(self.config.get("enabled", True))
        self.memory_config = MemoryConfig.from_dict(self.config)
        self.decision_config = DecisionConfig.from_dict(self.config)
        self.focus_config = FocusConfig.from_dict(self.config)
        self.short_term.update_config(self.memory_config)
        self.memory_store.update_config(self.memory_config)
        self.extractor.update_config(self.memory_config)
        self.maintainer.update_config(self.memory_config)
        self.decider.update_config(self.decision_config, self.memory_config)
        self.focus.update_config(self.focus_config)
        self._allowed_targets = self._load_allowed_targets()
        logger.info(
            "[什亭之枢] 配置已热更新 | 接管=%s 决策模型=%s",
            "开" if self.decision_config.enabled else "关",
            self._decision_provider_label(),
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
        rows = self.short_term.recent(self.memory_config.short_term_context_messages)
        same_chat = [str(row["content"]) for row in rows if str(row["chat_id"]) == chat_id]
        chosen = same_chat if same_chat else [str(row["content"]) for row in rows[-4:]]
        return "\n".join(chosen[-8:])

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
