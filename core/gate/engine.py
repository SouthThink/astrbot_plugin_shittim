"""决断层：在 AstrBot 官方 LLM 被触发之前判定是否回复。

分为两层，职责边界明确：

- **规则层** `rule_decide()`：同步执行、不消耗 token。上游指令、空文本、私聊、
  被点名均在此直接判定；返回 None 表示「交由巡检层继续判断」。私聊判定结果
  为「需要回复」，但并非立即回复——它带有 `debounce` 标记，由上层在静默时长
  满足后再唤醒主脑。
- **模型层** `judge_window()`：对窗口内的一段对话进行判定。调用方是巡检器
  （`patrol.PatrolManager`），由巡检器累积一段时间后调用，而非每条消息都
  触发一次模型调用。

两层判定失败时统一收敛为「不回复」：优先保证行为可控，避免无约束发言。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from astrbot.api import logger

from ..common.event_utils import (
    get_message_text,
    is_private_chat,
    is_upstream_command_event,
)
from ..common.model_json import parse_model_json
from ..memory.store import TIER_PERMANENT

SOURCE_RULE = "rule"
SOURCE_MODEL = "model"
SOURCE_FALLBACK = "fallback"

# 决策模型输出里 should_reply 的各种真值写法
_TRUE_WORDS = {"true", "1", "yes", "y", "是", "真", "需要", "回复"}

# 人格提示词的最大字数：判定需要的是人设特征，而非完整人设文档。
# 1200 字足以覆盖常用的精简人设（实测「爱丽丝简化版」812 字），
# 继续增大只会使每次判定多消耗数千 token，而判定本身只需要性格与语气。
_PERSONA_MAX_CHARS = 1200
# 人格极少变化，取一次并缓存一段时间即可，避免每个窗口都重新读取配置
_PERSONA_CACHE_SECONDS = 300.0

_WINDOW_PROMPT = """你是「{alias}」，正在「{chat_label}」。

【你是什么样的人】
{persona}

【你最近看到的事】
{recent}

【你记得的事】
{memory}

【这段对话之前】（只是用来理解上下文，不是要你回应的）
{history}

【这段对话】（按发生顺序）
{content}

【判断标准】
你是一个有自己性格、记忆和情绪的人，正在这个群里生活。你要判断的不是
「有没有人叫我」，而是「这段话里有没有我想接的东西」。

想开口，通常是因为：
- 有人直接叫你、问你，或在等你回应
- 话题碰到你的记忆、经历、喜好、身份，你有话可说
- 有人抛出观点或情绪，你有自己的看法、感受，或者想追问
- 气氛正适合你插一句：吐槽、附和、玩笑、补充都算

可以不开口：
- 纯粹的刷屏、复读、表情、链接搬运，没有任何可接的内容
- 这明显是两个人之间的私事，你插进去会打断
- 同一个话题你刚说过，再说就是重复

参与群聊是被允许的，不必等到被点名。正常的对话里，多数时候你都能找到
自然的接话点；但如果确实没有，沉默也没问题。不要为了说话而说话，
也不要因为怕打扰而一直旁观。

【输出】
只输出 JSON，不要解释，不要代码块围栏：
{{"should_reply": true, "reason": "一句话理由"}}
"""


@dataclass
class Decision:
    """一次拦放判定。

    passthrough=True 表示「不修改该事件」：既不拦截，也不设置唤醒标志，
    原样交给 AstrBot 原生流程。上游指令走此路径，避免指令消息因唤醒标志
    被额外送入 LLM。
    """

    should_reply: bool
    reason: str
    source: str  # rule | model | fallback
    passthrough: bool = False
    # 需要回复，但不立即回复：交由上层聚合一段静默时间（私聊防抖），避免打断用户连续发言
    debounce: bool = False


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
        # 人格文本缓存：chat_id -> (读取时刻, 文本)
        self._persona_cache: dict[str, tuple[float, str]] = {}

    def update_config(self, decision_config, memory_config) -> None:
        self._config = decision_config
        self._memory_config = memory_config

    # ── 规则层 ────────────────────────────────────────────

    def rule_decide(self, event) -> Decision | None:
        """可由规则判定的在此直接判定；返回 None 表示「交由巡检窗口继续判断」。

        规则层同步执行、不消耗 token：上游指令、空文本、私聊、被点名均在此
        直接判定。剩余的群聊普通消息交由巡检处理——累积一小段后整体判定一次，
        而非每条消息都调用一次模型。
        """
        chat_id = getattr(event, "unified_msg_origin", "") or ""
        text = get_message_text(event)

        # 上游指令必须放行，否则 /命令 会失效。此路径不修改唤醒标志。
        if self._config.allow_upstream_commands and is_upstream_command_event(event):
            return Decision(True, "上游指令，原样放行", SOURCE_RULE, passthrough=True)

        if not text:
            return Decision(False, "没有可回应的文本内容", SOURCE_RULE)

        if is_private_chat(chat_id) and self._config.always_reply_private:
            # 私聊必须回复，但不立即回复：对方常把一句话拆成多条发送，
            # 等待其安静下来后再一次性回应，避免在中间打断。
            return Decision(True, "私聊", SOURCE_RULE, debounce=True)

        if self._is_mentioned(event, text) and self._config.always_reply_when_mentioned:
            return Decision(True, "被点名", SOURCE_RULE)

        return None

    def _is_mentioned(self, event, text: str) -> bool:
        """被 @ / 命中 AstrBot 唤醒 / 文本中出现别名。"""
        if bool(getattr(event, "is_at_or_wake_command", False)):
            return True
        return any(alias and alias in text for alias in self._config.aliases)

    # ── 模型层 ────────────────────────────────────────────

    async def judge_window(
        self,
        chat_id: str,
        chat_label: str,
        lines: list[str],
        before: list[str] | None = None,
    ) -> Decision:
        """判定一段时间内的对话是否需要接话。

        与逐条判定的区别在于判据粒度：输入是一段连续对话，而非彼此
        缺乏关联的单条消息，因此模型可以判断话题是否已经结束、是否在等待
        其他人回应，而不是仅根据孤立语句推测上下文。

        `before` 是窗口开始前该会话的若干条消息。判定对象仍为窗口内的增量，
        但「不懂」「那他呢」一类语句需要紧邻前文才能判断指代对象；
        缺少前文时，模型只能以「无明确上下文」为由判定为不回复。
        """
        provider = self._resolve_provider()
        if provider is None:
            # 未配置决策模型时的兜底：群聊里的普通消息不介入
            return Decision(
                False, "未配置决策模型，按规则不介入群聊普通消息", SOURCE_FALLBACK
            )

        body = "\n".join(line for line in lines if line).strip()
        if not body:
            return Decision(False, "没有可判断的内容", SOURCE_FALLBACK)

        history = "\n".join(line for line in (before or []) if line).strip()
        persona = await self._persona_block(chat_id)

        prompt = _WINDOW_PROMPT.format(
            alias=self._alias_label(),
            persona=persona or "（没有额外的人格设定）",
            recent=self._short_term.build_recent_block(
                per_chat=self._config.context_messages,
                max_chats=3,
                exclude_chat_id=chat_id,
            )
            or "（暂无近期消息）",
            memory=self._format_memories(body),
            chat_label=chat_label or "群聊",
            history=history or "（这段对话就是开头）",
            content=body,
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

    async def _persona_block(self, chat_id: str) -> str:
        """读取当前生效的人格提示词，供判定层判断「这个人会如何接话」。

        判定层原本只读取窗口与记忆，模型只能依据「是否有人叫我」判断，
        因而倾向于把沉默当作安全答案。人格是「是否愿意开口」的另一半依据：
        活泼、偏好接话的人格与冷淡、克制的人格，对同一段对话的判断本就不同。

        读取失败时返回空串，判定照常进行——人格是增强项，不是前置条件。
        """
        now = time.time()
        cached = self._persona_cache.get(chat_id or "")
        if cached is not None and now - cached[0] < _PERSONA_CACHE_SECONDS:
            return cached[1]

        manager = getattr(self._astr_context, "persona_manager", None)
        if manager is None:
            return ""
        try:
            persona = await manager.get_default_persona_v3(chat_id)
        except Exception as error:
            logger.debug("[什亭之枢] 读取人格失败: %s", error)
            return ""
        if isinstance(persona, dict):
            prompt = persona.get("prompt") or ""
        else:
            prompt = getattr(persona, "prompt", "") or ""
        text = str(prompt).strip()[:_PERSONA_MAX_CHARS]
        self._persona_cache[chat_id or ""] = (now, text)
        return text

    def _format_memories(self, text: str) -> str:
        """提供给决策层的记忆片段：永久记忆 + 按这段话召回的长期记忆。

        永久记忆是稳定认知（已确认的事、重要的人），判定时优先列出，
        模型才能判断哪些话题与自身相关；召回部分仅在存在相关记忆时追加。
        读取失败时返回占位说明，不影响判定。
        """
        lines: list[str] = []
        seen: set[str] = set()

        def push(content: str) -> None:
            value = str(content or "").strip()
            if value and value not in seen:
                seen.add(value)
                lines.append(f"- {value}")

        try:
            for item in self._memory_store.list_by_tier(TIER_PERMANENT, limit=4):
                push(getattr(item, "content", ""))
        except Exception:
            pass
        try:
            recalled = self._memory_store.recall(
                query_text=text, top_k=3, user_top_k=0
            )
        except Exception:
            recalled = []
        for item in recalled:
            push(getattr(getattr(item, "memory", None), "content", ""))

        return "\n".join(lines) or "（暂无相关记忆）"
