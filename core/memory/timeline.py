"""单一全局时间线：所有会话按时间排成同一条脉络，作为主脑**唯一**的上下文。

## 不采用「按会话隔离 + 别处经历块」的原因

人的记忆只有一条流。切换聊天窗口不会导致失忆，只是更换了说话对象。若按会话
隔离历史，再把别处内容压成一段带标签的「近期经历」置于最前，模型读到的
是一份**由他人撰写的简报**——被问「最近有人找你吗」时，其第一反应是调用
`list_sessions` 查询，因为它不认为那是自己已知的信息。

时间线将这一关系还原为事实：模型说过的话即 `assistant` 回合，他人说的话即
`user` 回合，并在正文中标注「在哪儿、谁说的、多久前」。切换会话时上下文完全
不变，只有最后一条消息换了发言人——等同于人转身与另一个人说话。

## 只保留最近的条目

数据源是短期层中的有效条目：原始消息满足 `archived_at = 0 AND condensed_at = 0
AND staged = 0`，外加仍有效的摘要。因此它天然是「最近这一段」，此为刻意设计：
短期记忆代表今天，前几天的经历由长期记忆承载。

条目的两条出路都会使其从此处消失：压入摘要（原消息由摘要代表），或随
深睡整理清空（连同摘要一起，当天摘要已提升为长期记忆）。**摘要同样参与归档**：
整理完成后短期层应为空，否则无法满足「起床时短期记忆为空」的约定。

## assistant 回合不带前缀

`assistant` 回合仅包含原话。若将 `[私聊·白纸] 你 (3分钟前): …` 整行写入
assistant 回合，会形成自我陈述而非对话发言。

摘要使用 `user` 角色 + `<earlier_summary>` 标签：它不属于发言内容，
而是一段已模糊的客观记录。
"""

from __future__ import annotations

from ..common.tokenizer import estimate_tokens
from .ledger import format_message_line, format_speech

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


def build_timeline(
    entries: list[dict],
    *,
    current_prompt: str = "",
    token_budget: int,
) -> list[dict]:
    """把短期层的条目渲染成 req.contexts 形式的全局时间线。

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
            if (
                str(row["role"]) == "user"
                and str(row["content"] or "").strip() == prompt_text
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
