"""各域配置的数据类定义。

配置来源是 AstrBot 注入的插件配置 dict（由 _conf_schema.json 定义），
所有取值都带默认值回退，配置缺失不会导致初始化失败。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _get(cfg: dict, path: str, default: Any) -> Any:
    """按 "a.b.c" 路径取值，任一层缺失或为 None 时回退默认值。"""
    node: Any = cfg
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return default if node is None else node


def _get_float(cfg: dict, path: str, default: float) -> float:
    try:
        return float(_get(cfg, path, default))
    except (TypeError, ValueError):
        return default


def _get_int(cfg: dict, path: str, default: int) -> int:
    try:
        return int(_get(cfg, path, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class MemoryConfig:
    """记忆系统的全部可调参数。"""

    enabled: bool = True

    # ── 压缩 ──────────────────────────────────────────────
    extract_provider_id: str = ""
    extract_max_items: int = 12           # 单次压缩同时产出的长期记忆条数上限
    extract_min_strength: float = 1.0     # 压缩产出的记忆强度低于该值丢弃（1~10；默认不过滤）
    # 压缩时是否纳入当天事件及其结论，而非仅保留稳定事实。关闭后长期记忆更精简，
    # 代价是无法回忆前一天的对话（见计划 §9.4）
    extract_include_events: bool = True
    # 遗忘条件（§10.1）：名次进入榜尾后，还需同时满足降级次数与未召回时长
    forget_min_curations: int = 6         # 所需最少降级轮次
    forget_stale_days: int = 7            # 连续未召回天数

    # ── 短期层 ────────────────────────────────────────────
    # 跨会话取样：按会话配额读取，而非全局取最近 N 条，避免高频会话
    # 挤占其它会话的展示配额。per_chat / chats 同时供决断层、巡检与
    # 主脑跨会话背景使用；tokens 只用于判定侧的 `build_recent_block`。
    short_term_context_per_chat: int = 6  # 每个会话的消息条数
    short_term_context_chats: int = 6     # 最多纳入的会话数
    short_term_context_tokens: int = 2000 # 判定侧跨会话语境的 token 预算
    # 短期压缩（§4.2）：条目总数超过阈值时，将最旧的一批压缩为一条摘要。
    # 单次整理仅执行一轮压缩，避免消息高峰期一次性压缩全部条目。
    condense_threshold: int = 300         # 触发压缩的条目总数阈值
    condense_batch: int = 100             # 单次压缩的条目数
    # ── 主脑上下文 ────────────────────────────────────────
    # 是否接管主脑上下文：开启后用本插件的全局时间线整体替换 req.contexts；
    # 关闭（默认）时沿用 AstrBot 原生按会话历史，记忆与跨会话背景通过
    # req.extra_user_content_parts 的临时内容块注入，不修改 req.contexts。
    context_takeover: bool = False
    # ── 跨会话背景 / 全局时间线 ───────────────────────────
    # 默认模式生成跨会话背景块：每个会话保留最近 short_term_context_per_chat 条，
    # 当前会话始终入选并补齐被拦下的消息。开启接管后改为渲染完整全局时间线。
    # 两者的取样上限与预算共用以下两项；超出后从最旧条目开始丢弃，
    # 优先保留最近发生的对话。见 memory/timeline.py。
    timeline_max_messages: int = 400      # 单次返回的条目数上限（SQL 层硬上限）
    timeline_token_budget: int = 8000     # 背景块 / 整条时间线的 token 上限
    # 请求发出前移除无法读取的图片引用：单张坏图会导致上游以 400 拒绝整条请求，
    # AstrBot 随后会将本轮切换到备用模型（人格与工具均会变化）
    strip_invalid_images: bool = True

    # ── 图片转文字 ────────────────────────────────────────
    # AstrBot 仅提供「[图片]」占位，账本需要补充图片描述，
    # 否则判定与记忆页面只能看到占位符。
    vision_enabled: bool = True
    vision_provider_id: str = ""          # 留空则使用 AstrBot 配置的图片转述模型
    vision_prompt: str = ""               # 留空则使用内置中文提示词
    vision_max_images: int = 2            # 单条消息最多转述的图片数

    # ── 合并转发 ──────────────────────────────────────────
    # AstrBot 仅提供「[转发消息]」占位，内层聊天记录需单独调用接口取回，
    # 否则判定与主脑都看不到转发内容。
    forward_enabled: bool = True
    forward_max_items: int = 20           # 单条转发最多读取的聊天记录条数
    forward_max_chars: int = 1200         # 单条转发写入账本的最大字符数

    # ── 记忆块（自动插入）────────────────────────────────
    # 榜前 N 条提供完整正文；其后 M 条仅提供关键词索引（用于提示仍存在
    # 未展开的记忆）。更靠后的条目不进入上下文。
    auto_top: int = 10                    # 提供完整正文的榜前条数
    auto_index: int = 30                  # 仅提供关键词索引的后续条数
    # 自动插入的 token 预算：永久记忆、榜前正文与关键词索引共用
    recall_token_budget: int = 1600
    # 工具召回参数：按关键词相关性排序，与榜序无关
    recall_top_k: int = 5                 # 话题相关记忆条数
    recall_user_top_k: int = 3            # 用户相关记忆条数
    recall_min_score: float = 0.18        # 低于该分数的记忆不返回

    # ── 维护 ──────────────────────────────────────────────
    # 只控制维护步骤（降级 / 遗忘 / 晋升 / 合并）；深睡是否整理由
    # schedule.curation_enabled 决定。
    maintenance_enabled: bool = True
    decay_half_life_days: float = 30.0    # 召回打分里的时间近因半衰期（天）
    promote_min_recall: int = 8           # 晋升永久记忆所需召回次数
    promote_min_cross_chat: int = 2       # 晋升永久记忆所需跨会话数
    merge_min_age_days: float = 14.0      # 只合并创建时间早于此值的长期记忆
    merge_min_cluster: int = 3            # 聚类规模达到此值才提交 LLM 合并
    merge_similarity: float = 0.34        # 聚类时的关键词/标签相似度阈值

    @classmethod
    def from_dict(cls, cfg: dict | None) -> "MemoryConfig":
        cfg = cfg if isinstance(cfg, dict) else {}
        return cls(
            enabled=bool(_get(cfg, "memory.enabled", True)),
            extract_provider_id=str(_get(cfg, "memory.extract_provider_id", "") or ""),
            extract_max_items=_get_int(cfg, "memory.extract_max_items", 12),
            extract_min_strength=_get_float(cfg, "memory.extract_min_strength", 1.0),
            extract_include_events=bool(
                _get(cfg, "memory.extract_include_events", True)
            ),
            forget_min_curations=_get_int(cfg, "memory.forget_min_curations", 6),
            forget_stale_days=_get_int(cfg, "memory.forget_stale_days", 7),
            short_term_context_per_chat=_get_int(
                cfg, "memory.short_term_context_per_chat", 6
            ),
            short_term_context_chats=_get_int(cfg, "memory.short_term_context_chats", 6),
            short_term_context_tokens=_get_int(
                cfg, "memory.short_term_context_tokens", 2000
            ),
            condense_threshold=_get_int(cfg, "memory.condense_threshold", 300),
            condense_batch=_get_int(cfg, "memory.condense_batch", 100),
            context_takeover=bool(_get(cfg, "memory.context_takeover", False)),
            timeline_max_messages=_get_int(
                cfg, "memory.timeline_max_messages", 400
            ),
            timeline_token_budget=_get_int(
                cfg, "memory.timeline_token_budget", 8000
            ),
            strip_invalid_images=bool(
                _get(cfg, "memory.strip_invalid_images", True)
            ),
            vision_enabled=bool(_get(cfg, "memory.vision_enabled", True)),
            vision_provider_id=str(_get(cfg, "memory.vision_provider_id", "") or ""),
            vision_prompt=str(_get(cfg, "memory.vision_prompt", "") or ""),
            vision_max_images=_get_int(cfg, "memory.vision_max_images", 2),
            forward_enabled=bool(_get(cfg, "memory.forward_enabled", True)),
            forward_max_items=_get_int(cfg, "memory.forward_max_items", 20),
            forward_max_chars=_get_int(cfg, "memory.forward_max_chars", 1200),
            auto_top=_get_int(cfg, "memory.auto_top", 10),
            auto_index=_get_int(cfg, "memory.auto_index", 30),
            recall_top_k=_get_int(cfg, "memory.recall_top_k", 5),
            recall_user_top_k=_get_int(cfg, "memory.recall_user_top_k", 3),
            recall_min_score=_get_float(cfg, "memory.recall_min_score", 0.18),
            recall_token_budget=_get_int(cfg, "memory.recall_token_budget", 1600),
            maintenance_enabled=bool(_get(cfg, "memory.maintenance_enabled", True)),
            decay_half_life_days=_get_float(cfg, "memory.decay_half_life_days", 30.0),
            promote_min_recall=_get_int(cfg, "memory.promote_min_recall", 8),
            promote_min_cross_chat=_get_int(cfg, "memory.promote_min_cross_chat", 2),
            merge_min_age_days=_get_float(cfg, "memory.merge_min_age_days", 14.0),
            merge_min_cluster=_get_int(cfg, "memory.merge_min_cluster", 3),
            merge_similarity=_get_float(cfg, "memory.merge_similarity", 0.34),
        )


@dataclass(frozen=True)
class DecisionConfig:
    """决断层配置：控制「官方 LLM 什么时候被触发」。

    这是接管对话的总开关。enabled=False 时插件不拦截任何消息，交由 AstrBot 原生流程处理。
    """

    enabled: bool = True
    # 决策模型；留空时复用记忆压缩模型；两者均未配置时仅启用规则层判定
    provider_id: str = ""
    # 私聊直接放行（不调用决策模型）
    always_reply_private: bool = True
    # 静默防抖：私聊与群内 @ 在最后一条消息之后静默该时长才唤醒主脑（秒），
    # 0 表示立即回复。用于避免对方连续发送多条消息时被中途打断：连续多条
    # 视为同一轮，聚合后再回复。文本提到名字走巡检窗口，不受此项控制。
    private_debounce_seconds: float = 10.0
    # 静默窗口的最长等待：持续有人说话时也到点放行，避免回复被无限推迟。
    # 0 表示不封顶。
    private_debounce_max_seconds: float = 60.0
    # 被 @ 时直接判定要回（仍走静默防抖）；文本出现名称时交给判定模型确认
    always_reply_when_mentioned: bool = True
    # 上游指令（/命令、skill）优先放行，否则指令将无法执行
    allow_upstream_commands: bool = True
    # 智能体名称 / 唤醒词，文本命中时视为被点名
    aliases: tuple[str, ...] = ()
    # 判定上下文中每个会话的近期消息条数上限
    context_messages: int = 6
    # ── 群聊巡检 ──────────────────────────────────────────
    # 不逐条判定，而是等对话安静下来后统一判定：既降低 token 消耗，
    # 也避免单条判定缺少对话上下文。
    patrol_enabled: bool = True
    # 最后一条群消息之后安静该时长才判定；期间出现新消息会重新计时。
    # 持续发言时不封顶判定——只在真正无人说话时才询问模型。
    patrol_window_seconds: float = 10.0
    patrol_cooldown_seconds: float = 60.0  # 判定为不回复后的会话冷却时长
    # 开口后的最小回复间隔：活跃群里避免连续接话。@ / 私聊不受此项限制。
    patrol_reply_cooldown_seconds: float = 60.0
    patrol_noise_filter: bool = True       # 整段仅含图片/重复内容/连续短句时跳过模型调用
    # 移除 AstrBot 内置的 send_message_to_user。
    # 该工具参数错误时会静默发送到当前会话（实测已造成误发），且绕过会话解析
    # 与焦点校验。跨会话发言统一由 say 处理，因此默认移除该工具。
    disable_builtin_send_tool: bool = True

    @classmethod
    def from_dict(cls, cfg: dict | None) -> "DecisionConfig":
        cfg = cfg if isinstance(cfg, dict) else {}
        raw_aliases = _get(cfg, "decision.aliases", [])
        if isinstance(raw_aliases, str):
            aliases = tuple(
                part.strip() for part in raw_aliases.split(",") if part.strip()
            )
        elif isinstance(raw_aliases, (list, tuple)):
            aliases = tuple(str(part).strip() for part in raw_aliases if str(part).strip())
        else:
            aliases = ()
        return cls(
            enabled=bool(_get(cfg, "decision.enabled", True)),
            provider_id=str(_get(cfg, "decision.provider_id", "") or ""),
            always_reply_private=bool(_get(cfg, "decision.always_reply_private", True)),
            private_debounce_seconds=_get_float(
                cfg, "decision.private_debounce_seconds", 10.0
            ),
            private_debounce_max_seconds=_get_float(
                cfg, "decision.private_debounce_max_seconds", 60.0
            ),
            always_reply_when_mentioned=bool(
                _get(cfg, "decision.always_reply_when_mentioned", True)
            ),
            allow_upstream_commands=bool(
                _get(cfg, "decision.allow_upstream_commands", True)
            ),
            aliases=aliases,
            context_messages=_get_int(cfg, "decision.context_messages", 6),
            patrol_enabled=bool(_get(cfg, "decision.patrol_enabled", True)),
            patrol_window_seconds=_get_float(
                cfg, "decision.patrol_window_seconds", 10.0
            ),
            patrol_cooldown_seconds=_get_float(
                cfg, "decision.patrol_cooldown_seconds", 60.0
            ),
            patrol_reply_cooldown_seconds=_get_float(
                cfg, "decision.patrol_reply_cooldown_seconds", 60.0
            ),
            patrol_noise_filter=bool(
                _get(cfg, "decision.patrol_noise_filter", True)
            ),
            disable_builtin_send_tool=bool(
                _get(cfg, "decision.disable_builtin_send_tool", True)
            ),
        )


@dataclass(frozen=True)
class FocusConfig:
    """会话焦点配置：约束主动发言范围。

    焦点只约束「主动发言」。任何会话收到消息都会触发唤醒并进入决断层判断，
    因此调小 timeout_minutes 只会降低主动发言频率，不会影响消息接收。
    """

    # 会话静默多久后收回主动发言资格
    timeout_minutes: float = 30.0
    # 主动发言前是否必须持有焦点
    require_focus_for_say: bool = True

    @classmethod
    def from_dict(cls, cfg: dict | None) -> "FocusConfig":
        cfg = cfg if isinstance(cfg, dict) else {}
        return cls(
            timeout_minutes=_get_float(cfg, "focus.timeout_minutes", 30.0),
            require_focus_for_say=bool(
                _get(cfg, "focus.require_focus_for_say", True)
            ),
        )


@dataclass(frozen=True)
class ScheduleConfig:
    """全局作息状态机配置。

    仅提供「时刻」与「阈值」，状态本身由时间轴推导（见 core/schedule/state.py），
    因此重启后无需恢复中间状态。enabled=False 时睡眠门不介入。
    """

    enabled: bool = True
    # 基础作息：入睡 / 起床 / 起床上限，格式 "HH:MM"
    sleep_at: str = "23:30"
    wake_at: str = "07:00"
    wake_latest: str = "12:00"
    # 起床前多久切换为浅睡（该时段允许叫醒）
    wake_notice_lead_minutes: int = 120
    # 浅睡持续多久后转入深睡
    light_until_deep_minutes: int = 120
    # 叫醒信号：同一会话在时间窗口内累计的 @/私聊次数达到阈值即叫醒
    wake_signal_window_minutes: int = 5
    wake_signal_threshold: int = 3
    # 被叫醒后的清醒宽限时长
    wake_grace_minutes: int = 10

    # ── 深睡期记忆整理（§9）────────────────────────────────
    # 总开关：关闭后维护、短期收口与清空都不会自动执行。
    curation_enabled: bool = True
    # 主触发条件与本轮深睡窗口绑定，每次进入 DEEP 后整理一次。
    # 该参数为兜底策略：距上次整理超过该时长仍未整理（例如深睡窗口未出现），
    # 则强制执行一次维护，但不收口、不清理账本。
    curation_force_after_hours: int = 48
    # 单次整理中短期收口的安全轮数上限（树状压缩，100:1 收缩）；
    # 达到上限仍存在多条摘要时跳过清空，保留数据
    collapse_max_rounds: int = 20
    # 单轮降级最多重写的正文条数（控制模型成本）
    curate_max_per_run: int = 20
    # 整理后将清醒期账本移入归档（从常规查询排除，数据保留）
    daily_reset_enabled: bool = True
    # 归档消息保留天数，到期物理删除
    archive_retention_days: int = 90

    @classmethod
    def from_dict(cls, cfg: dict | None) -> "ScheduleConfig":
        cfg = cfg if isinstance(cfg, dict) else {}

        def _time(path: str, default: str) -> str:
            raw = str(_get(cfg, path, default) or "").strip()
            return raw or default

        return cls(
            enabled=bool(_get(cfg, "schedule.enabled", True)),
            sleep_at=_time("schedule.sleep_at", "23:30"),
            wake_at=_time("schedule.wake_at", "07:00"),
            wake_latest=_time("schedule.wake_latest", "12:00"),
            wake_notice_lead_minutes=_get_int(
                cfg, "schedule.wake_notice_lead_minutes", 120
            ),
            light_until_deep_minutes=_get_int(
                cfg, "schedule.light_until_deep_minutes", 120
            ),
            wake_signal_window_minutes=_get_int(
                cfg, "schedule.wake_signal_window_minutes", 5
            ),
            wake_signal_threshold=_get_int(cfg, "schedule.wake_signal_threshold", 3),
            wake_grace_minutes=_get_int(cfg, "schedule.wake_grace_minutes", 10),
            curation_enabled=bool(_get(cfg, "schedule.curation_enabled", True)),
            curation_force_after_hours=_get_int(
                cfg, "schedule.curation_force_after_hours", 48
            ),
            collapse_max_rounds=_get_int(cfg, "schedule.collapse_max_rounds", 20),
            curate_max_per_run=_get_int(cfg, "schedule.curate_max_per_run", 20),
            daily_reset_enabled=bool(_get(cfg, "schedule.daily_reset_enabled", True)),
            archive_retention_days=_get_int(
                cfg, "schedule.archive_retention_days", 90
            ),
        )
