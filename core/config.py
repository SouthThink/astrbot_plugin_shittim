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

    # ── 提炼 ──────────────────────────────────────────────
    extract_provider_id: str = ""
    extract_batch_size: int = 40          # 累积多少条未提炼消息触发一次提炼
    extract_max_items: int = 5            # 单次提炼最多产出几条长期记忆
    extract_min_strength: float = 1.0     # 低于该强度的提炼结果直接丢弃

    # ── 短期层 ────────────────────────────────────────────
    short_term_max_messages: int = 400    # 每个会话各保留多少条（按会话分组淘汰）
    short_term_context_messages: int = 20 # 每个请求带上的近期原始消息条数
    short_term_context_tokens: int = 2000 # 近期原始消息的 token 预算
    # 从 AstrBot 官方历史回填插件不在场期间的对话
    backfill_astrbot_history: bool = True
    backfill_max_messages: int = 200      # 每个会话最多回填多少条

    # ── 召回 ──────────────────────────────────────────────
    recall_top_k: int = 5                 # 消息相关记忆条数
    recall_user_top_k: int = 3            # 用户相关记忆条数
    recall_min_score: float = 0.18        # 低于该分数的记忆不注入
    recall_token_budget: int = 900        # 记忆注入块的 token 预算

    # ── 维护 ──────────────────────────────────────────────
    maintenance_enabled: bool = True
    decay_half_life_days: float = 30.0    # 强度半衰期
    forget_threshold: float = 0.6         # 长期记忆强度低于此值即遗忘
    promote_min_recall: int = 8           # 晋升永久记忆所需召回次数
    promote_min_strength: float = 12.0    # 晋升永久记忆所需强度
    promote_min_cross_chat: int = 2       # 晋升永久记忆所需跨会话数
    merge_max_strength: float = 3.5       # 只合并强度不高于此值的长期记忆
    merge_min_age_days: float = 14.0      # 只合并创建时间早于此值的长期记忆
    merge_min_cluster: int = 3            # 聚类规模达到此值才提交 LLM 合并
    merge_similarity: float = 0.34        # 聚类时的关键词/标签相似度阈值

    @classmethod
    def from_dict(cls, cfg: dict | None) -> "MemoryConfig":
        cfg = cfg if isinstance(cfg, dict) else {}
        return cls(
            enabled=bool(_get(cfg, "memory.enabled", True)),
            extract_provider_id=str(_get(cfg, "memory.extract_provider_id", "") or ""),
            extract_batch_size=_get_int(cfg, "memory.extract_batch_size", 40),
            extract_max_items=_get_int(cfg, "memory.extract_max_items", 5),
            extract_min_strength=_get_float(cfg, "memory.extract_min_strength", 1.0),
            short_term_max_messages=_get_int(cfg, "memory.short_term_max_messages", 400),
            short_term_context_messages=_get_int(
                cfg, "memory.short_term_context_messages", 20
            ),
            short_term_context_tokens=_get_int(
                cfg, "memory.short_term_context_tokens", 2000
            ),
            backfill_astrbot_history=bool(
                _get(cfg, "memory.backfill_astrbot_history", True)
            ),
            backfill_max_messages=_get_int(cfg, "memory.backfill_max_messages", 200),
            recall_top_k=_get_int(cfg, "memory.recall_top_k", 5),
            recall_user_top_k=_get_int(cfg, "memory.recall_user_top_k", 3),
            recall_min_score=_get_float(cfg, "memory.recall_min_score", 0.18),
            recall_token_budget=_get_int(cfg, "memory.recall_token_budget", 900),
            maintenance_enabled=bool(_get(cfg, "memory.maintenance_enabled", True)),
            decay_half_life_days=_get_float(cfg, "memory.decay_half_life_days", 30.0),
            forget_threshold=_get_float(cfg, "memory.forget_threshold", 0.6),
            promote_min_recall=_get_int(cfg, "memory.promote_min_recall", 8),
            promote_min_strength=_get_float(cfg, "memory.promote_min_strength", 12.0),
            promote_min_cross_chat=_get_int(cfg, "memory.promote_min_cross_chat", 2),
            merge_max_strength=_get_float(cfg, "memory.merge_max_strength", 3.5),
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
    # 决策模型；留空则复用记忆提炼模型；都留空则走规则兜底
    provider_id: str = ""
    # 私聊直接放行（不花决策模型的 token）
    always_reply_private: bool = True
    # 被 @ / 叫到名字直接放行
    always_reply_when_mentioned: bool = True
    # 上游指令（/命令、skill）必须让路，否则指令会失效
    allow_upstream_commands: bool = True
    # 她的名字 / 唤醒词，命中即视为被叫到
    aliases: tuple[str, ...] = ()
    # 决策时给她看的近期消息条数
    context_messages: int = 12

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
            always_reply_when_mentioned=bool(
                _get(cfg, "decision.always_reply_when_mentioned", True)
            ),
            allow_upstream_commands=bool(
                _get(cfg, "decision.allow_upstream_commands", True)
            ),
            aliases=aliases,
            context_messages=_get_int(cfg, "decision.context_messages", 12),
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
