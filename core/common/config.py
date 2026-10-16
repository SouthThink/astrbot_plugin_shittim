"""记忆系统配置。

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
    extract_max_items: int = 12           # 单次压缩最多顺带产出几条长期记忆
    extract_min_strength: float = 1.0     # 低于该强度的压缩结果直接丢弃（仅作质量闸门，不再入库）
    # 压缩时是否纳入「当天发生的事」，而不只是稳定事实。关掉可让长期记忆更精简，
    # 代价是「第二天问昨天那件事」会失忆（见计划 §9.4）
    extract_include_events: bool = True
    # 遗忘阶梯（§10.1）：名次沉到榜尾后，还要满足「降级次数」与「连续多久没被召回」
    forget_min_curations: int = 6         # 至少经历过这么多轮降级
    forget_stale_days: int = 7            # 连续这么多天未被召回

    # ── 短期层 ────────────────────────────────────────────
    # 跨会话语境：按会话配额取，而不是全局取最近 N 条——
    # 否则一个话痨群能把其它所有会话挤出她的经历之外。
    # 只给决断层与巡检看（它们的提示词装不下整条时间线）
    short_term_context_per_chat: int = 6  # 每个会话取最近几条
    short_term_context_chats: int = 6     # 最多纳入几个会话
    short_term_context_tokens: int = 2000 # 这块的 token 预算
    # 短期压缩（§4.2）：条目总数超过阈值时，把最旧的一批压成一条摘要。
    # 每次整理**只压一轮**，爆发日不会一口气压到底。
    condense_threshold: int = 300         # 条目总数超过它才压缩
    condense_batch: int = 100             # 每次压缩取最旧多少条
    # ── 全局时间线（主脑唯一的上下文）────────────────────
    # 所有会话按时间排成同一条脉络，切会话不换上下文。见 memory/timeline.py。
    # 它是「今天」的全部，所以预算要按模型窗口给足：超了从**最旧的**开始丢，
    # 她可以忘记很早以前说的，但不能忘记刚刚发生的。
    timeline_max_messages: int = 400      # 单次最多取多少条条目（SQL 层硬上限）
    timeline_token_budget: int = 8000     # 整条时间线的 token 上限
    # 请求发出前摘掉读不到的图片引用：一张坏图会让整条请求被上游 400 拒绝，
    # 而 AstrBot 会因此把这轮切给备用模型（人格与工具全变）
    strip_invalid_images: bool = True

    # ── 图片转文字 ────────────────────────────────────────
    # AstrBot 只把图片转成「[图片]」占位，账本里得自己补一句描述，
    # 否则她「看不到」群里发的图，记忆页面也只显示一个占位。
    vision_enabled: bool = True
    vision_provider_id: str = ""          # 留空则用 AstrBot 设置的「图片转述模型」
    vision_prompt: str = ""               # 留空则用内置的中文提示词
    vision_max_images: int = 2            # 单条消息最多转述几张图

    # ── 记忆块（自动插入）────────────────────────────────
    # 榜前 N 条给正文；其后 M 条只给一行关键词索引（元记忆：让她知道自己
    # 还忘了些什么，才会去 recall）。再往后不进上下文。
    auto_top: int = 10                    # 榜前多少条给完整正文
    auto_index: int = 30                  # 之后多少条只给关键词索引
    # 自动插入的 token 预算：永久记忆 + 榜前正文 + 索引三段共用
    recall_token_budget: int = 1600
    # 工具召回（「我想想」）的参数：按关键词相关性取，与榜序无关
    recall_top_k: int = 5                 # 消息相关记忆条数
    recall_user_top_k: int = 3            # 用户相关记忆条数
    recall_min_score: float = 0.18        # 低于该分数的记忆不返回

    # ── 维护 ──────────────────────────────────────────────
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

    这是接管对话的总闸。enabled=False 时插件完全不拦，退回 AstrBot 原生行为。
    """

    enabled: bool = True
    # 决策模型；留空则复用记忆压缩模型；都留空则走规则兜底
    provider_id: str = ""
    # 私聊直接放行（不花决策模型的 token）
    always_reply_private: bool = True
    # 私聊防抖：发出后静默这么久没有下一条消息才唤醒主脑（秒），0 表示立即回。
    # 防的是「对方还在一条条打字，她就已经插嘴了」——连续几条应是同一轮，
    # 攒齐再一起看，回一句就够了。
    private_debounce_seconds: float = 10.0
    # 被 @ / 叫到名字直接放行
    always_reply_when_mentioned: bool = True
    # 上游指令（/命令、skill）必须让路，否则指令会失效
    allow_upstream_commands: bool = True
    # 她的名字 / 唤醒词，命中即视为被叫到
    aliases: tuple[str, ...] = ()
    # 决策时给她看的近期消息条数（每个会话各取这么多）
    context_messages: int = 6
    # ── 群聊巡检 ──────────────────────────────────────────
    # 不逐条判定，而是把一小段时间的消息攒起来判一次：既是省 token，
    # 更是因为单条判定会让模型看不到「这轮对话已经闭环了」。
    patrol_enabled: bool = True
    patrol_window_seconds: float = 5.0     # 群里安静这么久就判一次
    patrol_max_wait_seconds: float = 20.0  # 一直有人说话时的封顶，否则永远不判
    patrol_cooldown_seconds: float = 60.0  # 判「不回」后，该会话多久不再开窗
    patrol_noise_filter: bool = True       # 整段只有图片/复读/连刷时不花模型调用
    # 移除 AstrBot 内置的 send_message_to_user。
    # 该工具在参数写错时会静默把消息发到当前会话（实测已造成一次错发），
    # 而且绕过会话解析与焦点校验。既然由我们的 say 接管跨会话发言，就该关掉它。
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
                cfg, "decision.patrol_window_seconds", 5.0
            ),
            patrol_max_wait_seconds=_get_float(
                cfg, "decision.patrol_max_wait_seconds", 20.0
            ),
            patrol_cooldown_seconds=_get_float(
                cfg, "decision.patrol_cooldown_seconds", 60.0
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
    """会话焦点配置：约束她的主动发言范围。

    焦点只约束「主动发言」。任何会话来了消息都会唤醒她并进入决断层判断，
    所以把 timeout_minutes 调小只会让她更难主动开口，不会让她收不到消息。
    """

    # 会话安静多久后收回她的主动发言资格
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

    只提供「时刻」与「阈值」，状态本身由时间轴推导（见 core/schedule/state.py）——
    这样重启后不需要恢复任何中间态，算一次就知道她在哪一档。
    enabled=False 时睡眠门完全不介入，退回 7×24 在线。
    """

    enabled: bool = True
    # 基础作息：入睡 / 起床 / 起床上限，形如 "23:30"
    sleep_at: str = "23:30"
    wake_at: str = "07:00"
    wake_latest: str = "12:00"
    # 起床前多久转为浅睡（这段时间可被叫醒）
    wake_notice_lead_minutes: int = 120
    # 浅睡满多久转入深睡
    light_until_deep_minutes: int = 120
    # 叫醒信号：同一会话、窗口内累计多少次 @/私聊即叫醒
    wake_signal_window_minutes: int = 5
    wake_signal_threshold: int = 3
    # 被叫醒后的清醒宽限时长
    wake_grace_minutes: int = 10

    # ── 深睡期记忆整理（§9）────────────────────────────────
    curation_enabled: bool = True
    # 距上次整理满这么久才整理（「每晚一次」）
    curation_min_interval_hours: int = 24
    # 距上次整理超过这么久则强制整理一次，但**不清空、不重建**
    # （防 DEEP 窗口被挤没：睡得太短时深睡阶段可能整段不出现）
    curation_force_after_hours: int = 48
    # 单次整理内「短期收口」的安全轮数上限（树状压缩，100:1 收缩）；
    # 触顶仍剩多条摘要则跳过清空短期层——宁可留着也不能丢
    collapse_max_rounds: int = 20
    # 单轮降级最多重写多少条正文（控模型成本）
    curate_max_per_run: int = 20
    # 整理后把清醒期账本移入归档（从常规查询排除，数据保留）
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
            curation_min_interval_hours=_get_int(
                cfg, "schedule.curation_min_interval_hours", 24
            ),
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
