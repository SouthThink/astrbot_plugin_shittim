"""跨会话时间线的两种呈现方式。

- **默认（不接管上下文）**：`build_timeline_block()` 生成一段临时背景文本，
  由 main 通过 `req.extra_user_content_parts` 追加（`mark_as_temp()`）。
  AstrBot 原生按会话历史保持不变，前缀缓存稳定。
- **接管模式**：`build_timeline()` 把短期层渲染成完整的 `req.contexts`，
  整体替换原生历史。开启 `memory.context_takeover` 后才使用。

## 默认模式的内容边界

当前会话的完整对话已由 AstrBot 原生历史提供，因此背景块只补充：

- 其它会话的消息与摘要：统一人格需要知道别处发生了什么；
- 当前会话中被拦下、未进入原生历史的消息（`gated = 1`）；
- 原生历史为空时，当前会话的全部条目（新会话或历史被重置时的兜底）。

摘要没有 `chat_id`，无法按会话区分，统一保留：它们是压缩后的概要，
即使与原生历史略有重叠，代价也远小于丢失跨会话记忆。

## 文本必须逐字稳定

prompt 缓存按**前缀**匹配，追加块内每一行的文本必须逐字稳定：时间标签用
绝对时间戳（相对时间每轮都在变），条目顺序完全由账本决定，同一份账本状态
必须渲染出同一段文本。

## 接管模式的 assistant 回合不带前缀

`assistant` 回合仅包含原话。若将 `[私聊·白纸] 你 (...): …` 整行写入
assistant 回合，会形成自我陈述而非对话发言。

摘要使用 `user` 角色 + `<earlier_summary>` 标签：它不属于发言内容，
而是一段已模糊的客观记录。
"""

from __future__ import annotations

from ..common.tokenizer import estimate_tokens
from .ledger import format_message_line, format_speech

# 默认模式的跨会话背景块包裹标签。
_CROSS_OPEN = "<cross_session_timeline>"
_CROSS_CLOSE = "</cross_session_timeline>"

# 表示当前对话状态的那一行。位于时间线**末尾**（紧邻当前消息）：
# 它表示当前所处会话，而非背景资料。
_MARKER_TEMPLATE = (
    "（你现在在「{label}」说话。上面的方括号标的是你当时所在的会话，"
    "没标的就是这里。）"
)

# 短期摘要的包裹标签：表示一段已被压缩的旧经历，细节已模糊。
# 单独标注以避免与真实回合混淆——真实回合为完整记忆，摘要仅保留概要。
_SUMMARY_OPEN = "<earlier_summary>"
_SUMMARY_CLOSE = "</earlier_summary>"


def build_timeline_block(
    entries: list[dict],
    *,
    current_chat_id: str = "",
    chat_label: str = "",
    include_current: bool = False,
    token_budget: int,
) -> str:
    """把短期层条目渲染成一段跨会话背景文本（默认模式）。

    Args:
        entries: 按时间正序的条目（`ShortTermLedger.timeline_entries_balanced`）。
            每条是 `{"kind": "message"|"summary", "at": ..., "row": ...}`。
        current_chat_id: 当前会话标识，用于排除原生历史已覆盖的当前会话消息。
        chat_label: 当前会话标签，写入块尾作为位置提示。
        include_current: 是否保留当前会话的全部消息。原生历史为空时置 True。
        token_budget: 文本块的 token 上限。超出后从**最旧**的条目开始丢弃，
            优先保留最近发生的对话。

    Returns:
        以 `<cross_session_timeline>` 包裹的文本；没有可用条目时返回空串。
    """
    if not entries or token_budget <= 0:
        return ""

    header = (
        f"{_CROSS_OPEN}\n"
        "以下是你跨会话的近期经历（按时间排序）。当前会话的对话历史已由系统单独提供，"
        "这里只补充其它会话的消息，以及当前会话中被拦下、未进入系统历史的消息。"
        "它们只是背景，不要复述，也不要因此改变正在进行的对话。"
    )
    if chat_label:
        footer = f"（你现在在「{chat_label}」说话。）\n{_CROSS_CLOSE}"
    else:
        footer = _CROSS_CLOSE

    used = estimate_tokens(header) + estimate_tokens(footer)
    if used >= token_budget:
        return ""

    lines: list[str] = []
    for entry in reversed(entries):
        if not _include(entry, current_chat_id, include_current):
            continue
        line = _render_line(entry)
        if not line:
            continue
        cost = estimate_tokens(line)
        if lines and used + cost > token_budget:
            break
        lines.append(line)
        used += cost

    if not lines:
        return ""
    lines.reverse()
    return "\n".join([header, *lines, footer])


def _include(entry: dict, current_chat_id: str, include_current: bool) -> bool:
    """判断该条目是否应进入跨会话文本块。"""
    if entry.get("kind") == "summary":
        return True
    row = entry.get("row") or {}
    try:
        chat_id = str(row["chat_id"] or "")
        gated = int(row["gated"] or 0)
    except (KeyError, IndexError, TypeError, ValueError):
        chat_id, gated = "", 0
    if include_current or chat_id != current_chat_id:
        return True
    return gated == 1


def _render_line(entry: dict) -> str:
    """一个时间线条目 → 背景块中的一行文本。"""
    if entry.get("kind") == "summary":
        return _render_summary_line(entry["row"])
    return format_message_line(entry["row"])


def _render_summary_line(row) -> str:
    """摘要行：用固定标签包裹以区别于原始消息。"""
    content = str(row["content"] or "").strip()
    if not content:
        return ""
    return f"{_SUMMARY_OPEN}\n{content}\n{_SUMMARY_CLOSE}"


def build_timeline(
    entries: list[dict],
    *,
    current_prompt: str = "",
    token_budget: int,
) -> list[dict]:
    """把短期层的条目渲染成 req.contexts 形式的全局时间线（仅接管模式）。

    Args:
        entries: 按时间正序的条目（`ShortTermLedger.timeline_entries`）。
            每条是 `{"kind": "message"|"summary", "at": ..., "row": ...}`。
        current_prompt: 当前这条消息的文本。它由 `req.prompt` 承载，不得
            再出现在 contexts 中，否则模型会将其视为两轮对话；这里按「最后一条
            内容相同的 user 消息」将其剔除。
        token_budget: 整条时间线的 token 上限。超出后从**最旧**的条目开始丢弃，
            优先保留最近发生的对话。

    Returns:
        OpenAI 消息数组（不含当前消息）。每条都带 `_no_save`：时间线是每轮
        临时重建的视图，不能让它被固化进 AstrBot 的官方历史。
    """
    if not entries or token_budget <= 0:
        return []

    prompt_text = (current_prompt or "").strip()
    skip_index = -1
    if prompt_text:
        for index in range(len(entries) - 1, -1, -1):
            entry = entries[index]
            if entry.get("kind") != "message":
                continue
            row = entry["row"]
            if str(row["role"]) != "user":
                continue
            # 纯转发消息的 prompt 由「[转发消息]（转发内容：…）」构成，
            # 与账本里的占位正文不同，必须按渲染后的发言比对才能剔除。
            if (
                str(row["content"] or "").strip() == prompt_text
                or format_speech(row).strip() == prompt_text
            ):
                skip_index = index
                break

    # 从最新往回取，直到超出预算；再翻转回时间正序
    rendered: list[dict] = []
    used = 0
    for index in range(len(entries) - 1, -1, -1):
        if index == skip_index:
            continue
        item = _render(entries[index])
        if item is None:
            continue
        cost = estimate_tokens(str(item["content"]))
        if rendered and used + cost > token_budget:
            break
        rendered.append(item)
        used += cost
    rendered.reverse()

    # 当前会话的标签取自当前这条消息（它一定在时间线里），取不到就退回占位
    label = "当前会话"
    if 0 <= skip_index < len(entries):
        row = entries[skip_index].get("row") or {}
        label = str(row["chat_label"] or "") or label
    rendered.append(_entry("user", _MARKER_TEMPLATE.format(label=label)))
    return rendered


def _render(entry: dict) -> dict | None:
    """一个时间线条目 → 一个 contexts 条目；没有内容可呈现的返回 None。"""
    if entry.get("kind") == "summary":
        return _render_summary(entry["row"])
    return _render_message(entry["row"])


def _render_message(row) -> dict | None:
    speech = format_speech(row)
    if not speech:
        return None
    if str(row["role"] or "user") == "assistant":
        # 助手发言：干净的 assistant 回合，不带任何前缀
        return _entry("assistant", speech)
    # 他人发言：正文自带「[会话] 发言人 (多久前): 内容」，用于标明发言人及会话
    return _entry("user", format_message_line(row))


def _render_summary(row) -> dict | None:
    """摘要：一段被压缩的旧经历。

    使用 `user` 角色 + 标签包裹，而非 `assistant`：它不属于「助手说过的话」，
    而是一段已模糊的客观记录。混入 assistant 会使「我说过什么」失真。
    """
    content = str(row["content"] or "").strip()
    if not content:
        return None
    return _entry("user", f"{_SUMMARY_OPEN}\n{content}\n{_SUMMARY_CLOSE}")


def _entry(role: str, content: str) -> dict:
    return {
        "role": role,
        "content": content,
        "_no_save": True,
        "_shittim_timeline": True,
    }
