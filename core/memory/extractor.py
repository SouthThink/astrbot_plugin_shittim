"""长期记忆的压缩与合并：调用独立配置的轻量模型。

`condense()` 是短期层通向长期层的**唯一**通道：把一段对话压成摘要的同时，
一并产出值得长期保留的条目。`compress()` 使旧记忆的正文变模糊，`merge()`
把陈旧且相似的记忆并成一条。

可靠性约束（均为强制项，用于防止记忆污染）：
- 产出必须带 source_quote 原文锚点，禁止模型推断或改写
- 必须保留否定词，防止「不喜欢」被写成「喜欢」
- 合并只允许使用已有信息，禁止引入新信息
- 模型输出一律经过 _normalize_item 校验，越界字段被丢弃或截断
"""

from __future__ import annotations

from astrbot.api import logger

from ..common.model_json import parse_model_json
from ..common.tokenizer import extract_keywords, is_generic_term
from .store import Memory

# 单条记忆正文与原文锚点的长度上限
_MAX_CONTENT_CHARS = 160
_MAX_QUOTE_CHARS = 200
_MAX_KEYWORDS = 12
_MAX_TAGS = 8

# 用户画像的输出上限：描述是「印象」，不是传记；关键词是标签，不是句子
_MAX_PROFILE_SUMMARY = 240
_MAX_PROFILE_KEYWORDS = 10
_MAX_PROFILE_KEYWORD_CHARS = 10
# 单次允许「吸收」的记忆条数上限：超过这个数说明模型在乱列，宁可一条不删
_MAX_ABSORBED = 20

# 扩大覆盖范围的关键条款：除「稳定事实」外，也记录「当天发生的事件」。
# 旧提示词把此类内容归入「一次性闲聊」，导致次日追问昨日事件时无法回忆。
# 覆盖扩大后条目数量会增加，因此提供 extract_include_events 开关，
# 并与名次驱动的降级配合：压缩阶段多记录以扩大广度，名次下沉后再压缩正文长度。
_EVENTS_CLAUSE = """- 今天聊到的事及其结论：谁在做什么、讨论出了什么结果。哪怕只是一次
  临时讨论，只要有了结论就要记下来
"""

_EXTRACT_RULES = """- 用户的稳定偏好、习惯、忌讳
- 关于某人的事实：身份、关系、称呼、生日、职业
- 约定、承诺、待办事项
{events}"""

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


_COMPRESS_PROMPT = """把下面这条记忆压缩得更短，只保留主语与核心信息。

【硬性要求】
1. 只能删除，不得新增信息、不得推测、不得改变原意，尤其是否定词：
   「小明不喜欢猫」不能压成「小明喜欢猫」。
2. 不超过 {max_chars} 字，写成一句完整的话。
3. 只输出压缩后的这句话，不要解释、不要引号、不要代码块。

原句：{content}
"""

_CONDENSE_PROMPT = """她正在整理自己的一段旧记忆。下面是这段时间里她经历过的对话，\
按时间顺序排列，每行方括号内标注了消息来源。

请做两件事。

【第一件：summary】
把这一整段压成一段概括，不超过 {max_summary_chars} 字。
- 写清谁、大致什么时候、发生了什么、结果如何——让人读一遍就知道这段时间的轮廓
- 保留结论与变化，省略寒暄、表情、过程
- 用第三人称陈述，不要写成对话，也不要逐条罗列

【第二件：memories】
从这段话里提炼**值得长期记住**的信息，规则如下：
{extract_rules}

【不值得记住】
- 寒暄、玩笑、一次性的闲聊
- 助手自己说的话
- 没有长期价值的即时信息

【硬性要求】
1. 只依据对话原文，禁止任何推测、补全或改写。
2. 必须保留原意，尤其是否定词：「不喜欢猫」绝不可写成「喜欢猫」。
3. summary 用第三人称；memories 每条 content 不超过 80 字，且**必须写明主语**
   （写成「小明不喜欢猫」，不能写成「不喜欢猫」）。
4. source_quote 必须逐字摘录原文里的一句话，不得改写、不得拼接。
5. strength 取 1-10 的整数：偏好、约定、身份类给 6-10，一般事实给 3-6。
6. memories 最多 {max_items} 条；这段里没有值得长期记住的内容就输出 []。

【输出格式】
只输出 JSON，不要解释，不要代码块围栏：
{{
  "summary": "一段概括",
  "memories": [
    {{
      "content": "客观陈述句",
      "source_quote": "原文逐字摘录",
      "tags": ["分类标签"],
      "keywords": ["检索关键词"],
      "user_name": "这句话的说话人名字，无法判断则留空字符串",
      "strength": 8
    }}
  ]
}}

【对话记录】
{records}
"""


_PROFILE_PROMPT = """你在整理一个人的长期画像。这个人会出现在私聊和不同的群里，\
但都是同一个人（账号 {user_id}）。

【对象】
- 现在的名字：{display_name}
- 用过的名字：{aliases}

【已有画像】
描述：{summary}
关键词：{keywords}

【关于他的长期记忆】（编号 + 内容；平时沉淀下来的事实，可能已经过时或重复）
{memories}

【最近的发言】（按时间顺序，来自不同会话）
{records}

【要做的事】
根据上面的材料更新你对这个人的印象：
1. 保留仍然成立的旧信息，修正与最近发言冲突的内容，删除已经过时或被明确否定的内容。
2. 只写这个人自己表现出来的东西：身份、兴趣、偏好、习惯、表达过的观点、与人相处的方式。
3. 不要写助手自己的话，不要推测，不要因为一次玩笑或一句气话就下结论。
4. 描述用第三人称，1~3 句，不超过 {max_summary} 字；关键词 {max_kw} 个以内，每个不超过 {max_kw_chars} 字。
5. 信息不足就保持原样；宁可不写，也不要编造。
6. absorbed：只有当某条长期记忆的内容**已经完整地**体现在你新写的描述或关键词里，
   而且它不含事件经过、约定、待办、时间、数字等任何额外信息时，才把它的编号放进来。
   拿不准就不要放——多留一条只是冗余，删错一条就真的丢了。

【输出】
只输出 JSON，不要解释，不要代码块围栏：
{{"summary": "对他的印象", "keywords": ["关键词"], "absorbed": [编号]}}
"""


def _clamp_str(value, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()[:limit]


def _str_list(value, limit: int, max_chars: int = 40) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = _clamp_str(item, max_chars)
        if text and text not in result:
            result.append(text)
        if len(result) >= limit:
            break
    return result


def _int_list(value, limit: int) -> list[int]:
    """从模型输出里取一组正整数编号；非法项静默跳过。"""
    if not isinstance(value, list):
        return []
    result: list[int] = []
    for item in value:
        try:
            number = int(item)
        except (TypeError, ValueError):
            continue
        if number > 0 and number not in result:
            result.append(number)
        if len(result) >= limit:
            break
    return result


_WORTHLESS_PROMPT = """你在清理长期记忆。下面每一条都注明了编号、正文、原话与关键词。

判断哪些条目**已经完全没有任何可用信息**：
- 正文只剩分类词、标签词或占位词（例如「事实 / 结论」「约定 / 爱好」），读不出具体的人、事、物；
- 原话为空，或者原话本身也只是这类词；
- 关键词同样全是这类分类词，没有任何指向具体内容的词。

以上三条要**同时**成立才算无价值。只要还能看出一点具体内容（谁、做了什么、什么时候、
和谁有关），它就仍然有价值，不要列出来。

被列出的条目会在下一次整理时被删除，所以拿不准就不要列。

【输出】
只输出 JSON，不要解释，不要代码块围栏：
{{"worthless": [编号, ...]}}

【记忆列表】
{memories}
"""


def _render_worthless(memories: list[Memory]) -> str:
    """把候选记忆渲染成「编号 + 正文 + 原话 + 关键词」，供无价值判断。"""
    lines: list[str] = []
    for memory in memories:
        quote = (memory.source_quote or "").strip() or "（空）"
        keywords = " / ".join(memory.keywords[:8]) or "（空）"
        lines.append(
            f"#{memory.id} 正文：{memory.content}｜原话：{quote}｜关键词：{keywords}"
        )
    return "\n".join(lines)


class MemoryExtractor:
    """把短期消息压成摘要并顺带产出长期记忆条目，以及压缩正文、合并陈旧记忆。"""

    def __init__(self, astr_context, config):
        self._astr_context = astr_context
        self._config = config

    def update_config(self, config) -> None:
        self._config = config

    # ── 对外接口 ──────────────────────────────────────────

    @property
    def available(self) -> bool:
        """未配置压缩模型时不启用压缩，避免静默回退到主模型消耗额度。"""
        if not self._config.extract_provider_id:
            return False
        return self._get_provider() is not None

    async def compress(self, content: str, max_chars: int) -> str:
        """把一条记忆压得更短（§9.6 的「精简」级）。

        与压缩 / 合并共用同一个模型与同一份配置，不引入第三套模型设置——
        多套模型处理同一批数据且规则不同，出问题后无法回溯是哪一次修改导致。

        `source_quote` 不参与压缩：锚点始终指向原文，压缩只降低正文精度。
        无法压缩（输出为空或未变短）时返回空串，由调用方保留原正文。
        """
        text = (content or "").strip()
        if not text or not self.available:
            return ""
        raw = await self._text_chat(
            _COMPRESS_PROMPT.format(max_chars=int(max_chars), content=text)
        )
        result = (raw or "").strip().strip("\"'“”").replace("\n", " ").strip()
        if not result or len(result) >= len(text):
            return ""
        return result[:max_chars]

    async def condense(
        self, records_text: str, *, max_summary_chars: int
    ) -> tuple[str, list[dict]] | None:
        """把一段对话压成摘要，并**同时**提炼长期记忆（§4.2）。

        这是「短期层 → 长期层」唯一的通道：每一次调用同时产出「1 条摘要」与
        「N 条长期记忆」，两者都直接来自原文，互为**兄弟**关系；若改为「先摘
        摘要、再从摘要提炼」的父子关系，等于压缩两次，细节会大量丢失。

        `max_summary_chars` 由调用方指定：中间层摘要只需作为背景，最后一次
        压缩产出的是「当天摘要」，需要承载一整天的轮廓。

        返回 None 表示本次压缩不成立（模型未给出可用摘要）。调用方因此**必须**
        保留原始条目、不标记 `condensed_at`，否则该段内容会永久退出上下文且
        不产生任何报错。
        """
        text = (records_text or "").strip()
        if not text or not self.available:
            return None
        events = _EVENTS_CLAUSE if self._config.extract_include_events else ""
        prompt = _CONDENSE_PROMPT.format(
            max_summary_chars=max(int(max_summary_chars), 1),
            max_items=self._config.extract_max_items,
            extract_rules=_EXTRACT_RULES.format(events=events),
            records=text,
        )
        raw = await self._text_chat(prompt)
        payload = parse_model_json(raw)
        if not isinstance(payload, dict):
            logger.warning("[什亭之枢] 短期压缩返回非对象，已放弃本次压缩")
            return None
        summary = _clamp_str(payload.get("summary"), int(max_summary_chars))
        if not summary:
            logger.warning("[什亭之枢] 短期压缩没有产出摘要，已放弃本次压缩")
            return None

        items: list[dict] = []
        raw_items = payload.get("memories")
        if isinstance(raw_items, list):
            for entry in raw_items:
                normalized = self._normalize_item(entry)
                if normalized is None:
                    continue
                if normalized["strength"] < self._config.extract_min_strength:
                    continue
                items.append(normalized)
                if len(items) >= self._config.extract_max_items:
                    break
        return summary, items

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
        # 合并必须确实减少条目数，否则只会额外引入一次改写风险
        if len(merged) >= len(memories):
            return []
        return merged

    async def judge_worthless(self, memories: list[Memory]) -> list[int]:
        """判断哪些长期记忆已经没有任何可用信息，返回这些记忆的 id。

        只看三个来源：正文、原文锚点、关键词。三者都只剩分类词才算无价值——
        判断的是「信息被榨干了」，不是「当下用不上」。调用方只会打标记，
        真正的删除留到下一次整理，期间被召回会解除标记。
        """
        if not memories or not self.available:
            return []
        rendered = _render_worthless(memories)
        if not rendered:
            return []
        raw = await self._text_chat(_WORTHLESS_PROMPT.format(memories=rendered))
        payload = parse_model_json(raw)
        if not isinstance(payload, dict):
            return []
        return _int_list(payload.get("worthless"), len(memories))

    async def build_profile(
        self,
        *,
        user_id: str,
        display_name: str,
        aliases: list[str],
        summary: str,
        keywords: list[str],
        records: str,
        memories: str = "",
    ) -> tuple[str, list[str], list[int]] | None:
        """根据「已有画像 + 这个人的长期记忆 + 最近发言」重写画像。

        输出是**完整替换**：模型看到旧画像与新素材，自行决定保留、改写还是
        删除。因此关键词被删掉的方式就是「新列表里不再出现它」，不需要额外
        的删除指令。

        同时返回 `absorbed`：内容已被新画像完整覆盖、不再有独立价值的记忆编号。
        删除由调用方执行——只有它知道哪些条目允许删（例如手动记住的不算）。

        Returns:
            (描述, 关键词, absorbed)；模型不可用、输出无法解析时返回 None，
            调用方保持原画像不变，下一轮整理再试。
        """
        text = (records or "").strip()
        if not text or not self.available:
            return None
        prompt = _PROFILE_PROMPT.format(
            user_id=user_id,
            display_name=display_name or "（未知）",
            aliases="、".join(aliases[:8]) or "（暂无）",
            summary=summary or "（暂无）",
            keywords=" / ".join(keywords) or "（暂无）",
            memories=(memories or "").strip() or "（暂无）",
            records=text,
            max_summary=_MAX_PROFILE_SUMMARY,
            max_kw=_MAX_PROFILE_KEYWORDS,
            max_kw_chars=_MAX_PROFILE_KEYWORD_CHARS,
        )
        raw = await self._text_chat(prompt)
        payload = parse_model_json(raw)
        if not isinstance(payload, dict):
            logger.warning("[什亭之枢] 用户画像输出无法解析，保留原画像")
            return None

        raw_summary = payload.get("summary")
        new_summary = (
            _clamp_str(raw_summary, _MAX_PROFILE_SUMMARY)
            if isinstance(raw_summary, str)
            else summary
        )
        raw_keywords = payload.get("keywords")
        if isinstance(raw_keywords, list):
            new_keywords = _str_list(
                raw_keywords, _MAX_PROFILE_KEYWORDS, _MAX_PROFILE_KEYWORD_CHARS
            )
        else:
            new_keywords = list(keywords)
        absorbed = _int_list(payload.get("absorbed"), _MAX_ABSORBED)
        return new_summary, new_keywords, absorbed

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
            # 缺少原文锚点的记忆无法回溯，直接丢弃
            logger.debug("[什亭之枢] 丢弃无原文锚点的提炼结果: %s", content[:40])
            return None
        try:
            strength = float(entry.get("strength", 5))
        except (TypeError, ValueError):
            strength = 5.0
        strength = max(1.0, min(strength, 10.0))
        tags = _str_list(entry.get("tags"), _MAX_TAGS)
        keywords = [
            keyword
            for keyword in _str_list(entry.get("keywords"), _MAX_KEYWORDS)
            if not is_generic_term(keyword)
        ]
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
