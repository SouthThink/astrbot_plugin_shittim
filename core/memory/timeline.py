"""跨会话时间线：将其它会话的近期经历汇总为一段临时背景文本。

## 为什么输出文本块，而不是替换 req.contexts

AstrBot 的原生历史按会话隔离，并且是 prompt 缓存前缀的组成部分。整体替换
`req.contexts` 会丢弃原生历史，同时使每条请求的前缀都发生变化，破坏缓存命中率，
也违反插件市场对「严重修改模型 Context」的限制。

因此本模块只生成**一段临时文本**，由 main 通过 `req.extra_user_content_parts`
追加（`TextPart(...).mark_as_temp()`）。原生上下文保持不变，缓存前缀稳定。

## 位置：只能追加在最后一条 user 消息之后

prompt 缓存按**前缀**匹配：开头不变时，后面的变化只影响它之后的部分。
因此追加位置直接决定了缓存代价：

- 写进 `system_prompt` 或插到历史中间 → 变化点之后的所有 token 全部失效；
- 追加到最后一条 user 消息内部 → 只有这段追加内容本身不计入命中，
  system 与全部历史仍是稳定前缀。

`extra_user_content_parts` 属于后者。同理，块内每一行的文本必须**逐字稳定**：
时间标签用绝对时间戳（相对时间每轮都在变），条目顺序完全由账本决定，
同一份账本状态必须渲染出同一段文本。

## 内容边界

当前会话的完整对话已由 AstrBot 原生历史提供，因此这里只补充：

- 其它会话的消息与摘要：统一人格需要知道别处发生了什么；
- 当前会话中被拦下、未进入原生历史的消息（`gated = 1`）；
- 原生历史为空时，当前会话的全部条目（新会话或历史被重置时的兜底）。

摘要没有 `chat_id`，无法按会话区分，统一保留：它们是压缩后的概要，
即使与原生历史略有重叠，代价也远小于丢失跨会话记忆。
"""

from __future__ import annotations

from ..common.tokenizer import estimate_tokens
from .ledger import format_message_line

_BLOCK_OPEN = "<cross_session_timeline>"
_BLOCK_CLOSE = "</cross_session_timeline>"

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
    """把短期层条目渲染成一段跨会话背景文本。

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
        f"{_BLOCK_OPEN}\n"
        "以下是你跨会话的近期经历（按时间排序）。当前会话的对话历史已由系统单独提供，"
        "这里只补充其它会话的消息，以及当前会话中被拦下、未进入系统历史的消息。"
        "它们只是背景，不要复述，也不要因此改变正在进行的对话。"
    )
    if chat_label:
        footer = f"（你现在在「{chat_label}」说话。）\n{_BLOCK_CLOSE}"
    else:
        footer = _BLOCK_CLOSE

    used = estimate_tokens(header) + estimate_tokens(footer)
    if used >= token_budget:
        return ""

    lines: list[str] = []
    for entry in reversed(entries):
        if not _include(entry, current_chat_id, include_current):
            continue
        line = _render(entry)
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


def _render(entry: dict) -> str:
    if entry.get("kind") == "summary":
        return _render_summary(entry["row"])
    return _render_message(entry["row"])


def _render_message(row) -> str:
    """消息行：带会话标签、发言人与绝对时间。"""
    return format_message_line(row)


def _render_summary(row) -> str:
    """摘要：一段已被压缩的旧经历，用固定标签包裹以区别于原始消息。"""
    content = str(row["content"] or "").strip()
    if not content:
        return ""
    return f"{_SUMMARY_OPEN}\n{content}\n{_SUMMARY_CLOSE}"
