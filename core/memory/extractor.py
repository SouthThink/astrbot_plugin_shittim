"""长期记忆的提炼与合并：调用独立配置的轻量模型。

可靠性约定（这些不是可选项，是防止记忆污染的必要约束）：
- 提炼结果必须带 source_quote 原文锚点，禁止模型推断或改写
- 明确要求保留否定词，防止「不喜欢」被写成「喜欢」
- 合并只允许使用已有信息，禁止引入新信息
- 模型输出一律经过 _normalize_item 校验，越界字段被丢弃或截断
"""

from __future__ import annotations

import json
import re

try:
    from astrbot.api import logger
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

from .memory_store import Memory
from .tokenizer import extract_keywords

# 单条记忆正文与原文锚点的长度上限
_MAX_CONTENT_CHARS = 120
_MAX_QUOTE_CHARS = 200
_MAX_KEYWORDS = 12
_MAX_TAGS = 8

_EXTRACT_PROMPT = """你是一个记忆提炼器。下面的对话记录来自同一个人参与的多个会话，\
每行方括号内标注了消息来源。

请从中提炼出**值得长期记住**的信息。

【值得记住】
- 用户的稳定偏好、习惯、忌讳
- 关于某人的事实：身份、关系、称呼、生日、职业
- 约定、承诺、待办事项
- 重要事件及其结果

【不值得记住】
- 寒暄、玩笑、一次性的闲聊
- 助手自己说的话
- 没有长期价值的即时信息

【硬性要求】
1. 只依据对话原文，禁止任何推测、补全或改写。
2. 必须保留原意，尤其是否定词：「不喜欢猫」绝不可写成「喜欢猫」。
3. 每条 content 不超过 60 字，且**必须写明主语**：写成「小明不喜欢猫」，
   不能写成「不喜欢猫」。缺主语的记忆在注入时会被误读为在说别人。
4. source_quote 必须逐字摘录原文里的一句话，不得改写、不得拼接。
5. strength 取 1-10 的整数：偏好、约定、身份类给 6-10，一般事实给 3-6。
6. 最多输出 {max_items} 条；没有值得记住的内容就输出 []。

【输出格式】
只输出 JSON 数组，不要解释，不要代码块围栏：
[
  {{
    "content": "客观陈述句",
    "source_quote": "原文逐字摘录",
    "tags": ["分类标签"],
    "keywords": ["检索关键词"],
    "user_name": "这句话的说话人名字，无法判断则留空字符串",
    "strength": 8
  }}
]

【对话记录】
{records}
"""

_MERGE_PROMPT = """你在整理长期记忆。下面是一组内容相关、都比较旧、也很少被用到的记忆。

请把它们合并成尽量少的条目。

【硬性要求】
1. 只能使用下面记忆中已有的信息，绝对禁止引入新信息或做任何推测。
2. 必须保留否定词，不得把「不喜欢」写成「喜欢」。
3. 合并后的 content 不超过 80 字，写成客观陈述句。
4. tags 与 keywords 取并集并去重。
5. 最多输出 2 条。

【输出格式】
只输出 JSON，不要解释，不要代码块围栏：
{{"items": [{{"content": "...", "tags": ["..."], "keywords": ["..."]}}]}}

【记忆列表】
{memories}
"""


def parse_model_json(raw: str):
    """从模型输出里健壮地抠出 JSON，容忍代码块围栏与前后赘述。"""
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        pass
    # 兜底：截取最外层的数组或对象
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except (TypeError, ValueError):
                continue
    return None


def _clamp_str(value, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()[:limit]


def _str_list(value, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = _clamp_str(item, 40)
        if text and text not in result:
            result.append(text)
        if len(result) >= limit:
            break
    return result


class MemoryExtractor:
    """把短期消息提炼成长期记忆条目，以及合并陈旧记忆。"""

    def __init__(self, astr_context, config):
        self._astr_context = astr_context
        self._config = config

    def update_config(self, config) -> None:
        self._config = config

    # ── 对外接口 ──────────────────────────────────────────

    @property
    def available(self) -> bool:
        """未配置提炼模型时不启用提炼，避免静默走主模型烧额度。"""
        if not self._config.extract_provider_id:
            return False
        return self._get_provider() is not None

    async def extract(self, records_text: str) -> list[dict]:
        """从短期消息文本中提炼长期记忆，返回已校验的条目列表。"""
        records_text = (records_text or "").strip()
        if not records_text or not self.available:
            return []
        prompt = _EXTRACT_PROMPT.format(
            max_items=self._config.extract_max_items, records=records_text
        )
        raw = await self._text_chat(prompt)
        payload = parse_model_json(raw)
        if not isinstance(payload, list):
            logger.warning("[什亭之枢] 记忆提炼返回非数组，已丢弃本次结果")
            return []
        items: list[dict] = []
        for entry in payload:
            normalized = self._normalize_item(entry)
            if normalized is None:
                continue
            if normalized["strength"] < self._config.extract_min_strength:
                continue
            items.append(normalized)
            if len(items) >= self._config.extract_max_items:
                break
        return items

    async def merge(self, memories: list[Memory]) -> list[dict]:
        """把一组陈旧弱记忆合并成更少的条目；失败返回空列表（调用方放弃合并）。"""
        if len(memories) < 2 or not self.available:
            return []
        lines = [f"- {m.content}" for m in memories if m.content]
        if len(lines) < 2:
            return []
        prompt = _MERGE_PROMPT.format(memories="\n".join(lines))
        raw = await self._text_chat(prompt)
        payload = parse_model_json(raw)
        if not isinstance(payload, dict):
            return []
        items = payload.get("items")
        if not isinstance(items, list):
            return []
        merged: list[dict] = []
        for entry in items:
            normalized = self._normalize_item(entry, require_quote=False)
            if normalized is not None:
                merged.append(normalized)
            if len(merged) >= 2:
                break
        # 合并必须真的减少条目数，否则等于白白引入一次改写风险
        if len(merged) >= len(memories):
            return []
        return merged

    # ── 内部实现 ──────────────────────────────────────────

    def _get_provider(self):
        provider_id = self._config.extract_provider_id
        if not provider_id or self._astr_context is None:
            return None
        try:
            return self._astr_context.get_provider_by_id(provider_id)
        except Exception:
            return None

    async def _text_chat(self, prompt: str) -> str:
        provider = self._get_provider()
        if provider is None:
            return ""
        try:
            token = await provider.text_chat(prompt=prompt)
            return (getattr(token, "completion_text", "") or "").strip()
        except Exception as error:
            logger.error("[什亭之枢] 记忆模型调用失败: %s", error)
            return ""

    def _normalize_item(self, entry, require_quote: bool = True) -> dict | None:
        """校验并规范化模型输出的一条记忆；不合格返回 None。"""
        if not isinstance(entry, dict):
            return None
        content = _clamp_str(entry.get("content"), _MAX_CONTENT_CHARS)
        if not content:
            return None
        quote = _clamp_str(entry.get("source_quote"), _MAX_QUOTE_CHARS)
        if require_quote and not quote:
            # 没有原文锚点的记忆无法回溯，宁可丢弃
            logger.debug("[什亭之枢] 丢弃无原文锚点的提炼结果: %s", content[:40])
            return None
        try:
            strength = float(entry.get("strength", 5))
        except (TypeError, ValueError):
            strength = 5.0
        strength = max(1.0, min(strength, 10.0))
        tags = _str_list(entry.get("tags"), _MAX_TAGS)
        keywords = _str_list(entry.get("keywords"), _MAX_KEYWORDS)
        if not keywords:
            keywords = extract_keywords(content, _MAX_KEYWORDS)
        return {
            "content": content,
            "source_quote": quote,
            "tags": tags,
            "keywords": keywords,
            "user_name": _clamp_str(entry.get("user_name"), 40),
            "strength": strength,
        }
