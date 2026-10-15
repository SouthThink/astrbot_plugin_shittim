"""决断层：在 AstrBot 官方 LLM 被触发之前，决定这条消息要不要回。

这是「思考与发送分离」里的思考层门闩：
- 判定不回 → 调用方 `stop_event()`，官方 LLM 完全不触发
- 判定回   → 调用方置位 `is_at_or_wake_command` 放行主脑生成

「规则优先、模型兜底」是刻意的成本设计：能用规则确定的（上游指令、私聊、
被点名）不花模型 token，只有群聊里的普通消息才交给轻量模型判断。

所有异常路径一律判定「不回」：宁可她安静，也不要不受控地开口。
"""

from __future__ import annotations

try:
    from astrbot.api import logger
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

from dataclasses import dataclass

from .event_utils import (
    describe_chat,
    get_message_text,
    is_private_chat,
    is_upstream_command_event,
)
from .extractor import parse_model_json

SOURCE_RULE = "rule"
SOURCE_MODEL = "model"
SOURCE_FALLBACK = "fallback"

# 决策模型输出里 should_reply 的各种真值写法
_TRUE_WORDS = {"true", "1", "yes", "y", "是", "真", "需要", "回复"}

_DECIDE_PROMPT = """你是「{alias}」的思考层。请判断下面这条新消息，是否需要你亲自开口回复。

【你最近看到的事】
{recent}

【你记得的事】
{memory}

【需要判断的新消息】
来源：{chat_label}
{content}

【判断标准】
需要回复：
- 有人直接叫你、在问你、或在等你回应
- 话题与你的记忆、身份或职责有关
- 群里聊到了你能补充的信息
不需要回复：
- 别人之间的对话，你插进去很突兀
- 与你无关的闲聊、复读、表情、刷屏
- 你没有什么可说的

宁可不回，也不要硬凑话，沉默是完全允许的。
如果你刚刚已经回应过同一个人、同一话题，不要重复回应。

【输出】
只输出 JSON，不要解释，不要代码块围栏：
{{"should_reply": true, "reason": "一句话理由"}}
"""


@dataclass
class Decision:
    """一次拦放判定。

    passthrough=True 表示「完全不要动这个事件」：既不拦也不置唤醒标志，
    原样交给 AstrBot 原生流程。上游指令走这条路，避免指令消息被我们
    的置位动作额外喂给 LLM。
    """

    should_reply: bool
    reason: str
    source: str  # rule | model | fallback
    passthrough: bool = False


class DecisionEngine:
    """规则短路 + 轻量模型判断的拦放判据。"""

    def __init__(
        self,
        astr_context,
        decision_config,
        memory_config,
        short_term,
        memory_store,
    ):
        self._astr_context = astr_context
        self._config = decision_config
        self._memory_config = memory_config
        self._short_term = short_term
        self._memory_store = memory_store

    def update_config(self, decision_config, memory_config) -> None:
        self._config = decision_config
        self._memory_config = memory_config

    # ── 对外入口 ──────────────────────────────────────────

    async def decide(self, event) -> Decision:
        """判定这条消息要不要回。任何异常都收敛为「不回」。"""
        try:
            ruled = self._rule_decide(event)
            if ruled is not None:
                return ruled
            return await self._model_decide(event)
        except Exception as error:
            logger.error("[什亭之枢] 决断层判定异常，按不回复处理: %s", error)
            return Decision(False, "决断层异常", SOURCE_FALLBACK)

    # ── 规则层 ────────────────────────────────────────────

    def _rule_decide(self, event) -> Decision | None:
        """能靠规则定的直接定，返回 None 表示需要交给模型。"""
        chat_id = getattr(event, "unified_msg_origin", "") or ""
        text = get_message_text(event)

        # 上游指令必须让路，否则 /命令 会失效。这里连唤醒标志都不碰。
        if self._config.allow_upstream_commands and is_upstream_command_event(event):
            return Decision(True, "上游指令，原样让路", SOURCE_RULE, passthrough=True)

        if not text:
            return Decision(False, "没有可回应的文本内容", SOURCE_RULE)

        if is_private_chat(chat_id) and self._config.always_reply_private:
            return Decision(True, "私聊", SOURCE_RULE)

        if self._is_mentioned(event, text) and self._config.always_reply_when_mentioned:
            return Decision(True, "被点名", SOURCE_RULE)

        return None

    def _is_mentioned(self, event, text: str) -> bool:
        """被 @ / 命中 AstrBot 唤醒 / 文本里出现她的名字。"""
        if bool(getattr(event, "is_at_or_wake_command", False)):
            return True
        return any(alias and alias in text for alias in self._config.aliases)

    # ── 模型层 ────────────────────────────────────────────

    async def _model_decide(self, event) -> Decision:
        provider = self._resolve_provider()
        if provider is None:
            # 未配置决策模型时的兜底：群聊里的普通消息不介入
            return Decision(
                False, "未配置决策模型，按规则不介入群聊普通消息", SOURCE_FALLBACK
            )

        chat_id = getattr(event, "unified_msg_origin", "") or ""
        _, chat_label = describe_chat(event)
        text = get_message_text(event)
        prompt = _DECIDE_PROMPT.format(
            alias=self._alias_label(),
            recent=self._short_term.build_recent_block(
                limit=self._config.context_messages
            )
            or "（暂无近期消息）",
            memory=self._format_memories(text),
            chat_label=chat_label,
            content=text,
        )

        try:
            token = await provider.text_chat(prompt=prompt)
            raw = (getattr(token, "completion_text", "") or "").strip()
        except Exception as error:
            logger.error("[什亭之枢] 决策模型调用失败: %s", error)
            return Decision(False, "决策模型调用失败", SOURCE_FALLBACK)

        payload = parse_model_json(raw)
        if not isinstance(payload, dict) or "should_reply" not in payload:
            logger.warning("[什亭之枢] 决策模型输出无法解析，按不回复处理: %s", raw[:120])
            return Decision(False, "决策模型输出无法解析", SOURCE_FALLBACK)

        should_reply = self._coerce_bool(payload.get("should_reply"))
        reason = str(payload.get("reason") or "").strip()[:80]
        return Decision(
            should_reply,
            reason or ("决定回应" if should_reply else "决定不回应"),
            SOURCE_MODEL,
        )

    @staticmethod
    def _coerce_bool(value) -> bool:
        """模型可能把布尔写成字符串，这里统一收敛。"""
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in _TRUE_WORDS
        return False

    def _resolve_provider(self):
        provider_id = self._config.provider_id or self._memory_config.extract_provider_id
        if not provider_id or self._astr_context is None:
            return None
        try:
            return self._astr_context.get_provider_by_id(provider_id)
        except Exception:
            return None

    def _alias_label(self) -> str:
        return self._config.aliases[0] if self._config.aliases else "你"

    def _format_memories(self, text: str) -> str:
        """给决策层看的记忆片段。取不到就返回占位说明，不影响判定。"""
        try:
            recalled = self._memory_store.recall(
                query_text=text, top_k=3, user_top_k=0
            )
        except Exception:
            return "（记忆检索失败）"
        if not recalled:
            return "（暂无相关记忆）"
        return "\n".join(f"- {item.memory.content}" for item in recalled)
