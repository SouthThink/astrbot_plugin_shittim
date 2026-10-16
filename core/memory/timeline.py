"""单一全局时间线：所有会话按时间排成同一条脉络，作为主脑**唯一**的上下文。

## 为什么不是「按会话隔离 + 别处经历块」

人的记忆只有一条流。换聊天窗口不会让人失忆，只是换了个说话对象。按会话
隔离历史、再把别处的内容压成一段贴着标签的「近期经历」塞在最前面，模型读到的
是一份**别人写给它的简报**——被问「最近有人找你吗」时，它的第一反应是去调
`list_sessions` 查一遍，因为它不觉得那是自己知道的。

时间线把这件事还原成事实：她说过的话就是 `assistant` 回合，别人说的话就是
`user` 回合、并在正文里带上「在哪儿、谁说的、多久前」。切会话时上下文完全
不变，只有最后那条消息换了个人——就像人转过身去跟另一个人说话。

## 只保留最近的条目

数据源是短期层「活着的条目」：原始消息走 `archived_at = 0 AND condensed_at = 0
AND staged = 0`，外加还活着的摘要。因此它天然是「最近这一段」——这是刻意的：
短期记忆就是今天，前几天的经历由长期记忆来承载。

条目的两条出路都会让它从这里消失：被压进摘要（原消息让位给摘要），或者随
深睡整理清空（连同摘要一起，当天摘要已提升为长期记忆）。**摘要不豁免归档**：
整理完短期层就该是空的，否则「刚起床时短期记忆是空的」就不成立。

## 她说过的话不带前缀

`assistant` 回合只放原话。若把 `[私聊·白纸] 你 (3分钟前): …` 整行塞进
assistant 回合，她会像是在陈述「我」而不是在说话。

摘要走 `user` 角色 + `<earlier_summary>` 标签：它不是她说过的话，
而是一段变模糊的客观记录。
"""

from __future__ import annotations

from ..common.tokenizer import estimate_tokens
from .ledger import format_message_line, format_speech

# 告诉她「你现在在跟谁说话」的那一行。放在时间线**末尾**（紧邻当前消息）：
# 它不是背景资料，而是她此刻的状态——人开口前不会先回忆自己在哪个房间。
_MARKER_TEMPLATE = (
    "（你现在在「{label}」说话。上面的方括号标的是你当时所在的会话，"
    "没标的就是这里。）"
)

# 短期摘要的包裹标签：它代表一段**被压扁**的旧经历，细节已经模糊。
# 单独标出来是为了不让它和真实回合混淆——真实回合是她记得的，这是她只记得个大概的。
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
        current_prompt: 当前这条消息的文本。它由 `req.prompt` 承载，绝不能
            再出现在 contexts 里，否则模型会以为是两轮；这里按「最后一条
            内容相同的 user 消息」把它剔掉。
        token_budget: 整条时间线的 token 上限。超了就从**最旧的**开始丢——
            她可以忘记很久以前说的，但不能忘记刚刚发生的。

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
        # 她说过的话：干净的 assistant 回合，不带任何前缀
        return _entry("assistant", speech)
    # 别人的话：正文自带「[会话] 发言人 (多久前): 内容」，她才知道是谁在哪儿说的
    return _entry("user", format_message_line(row))


def _render_summary(row) -> dict | None:
    """摘要：一段被压扁的旧经历。

    用 `user` 角色 + 标签包裹，而不是 `assistant`：它不是「她说过的话」这个
    事实，而是一段变模糊的客观记录。混进 assistant 会让「我说过什么」失真。
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
