"""睡前道别与起床问候的提示词。

这两次唤醒都不是一次对话：没有用户消息、没有工具、也不进 AstrBot 的历史。
主脑要做的只是一个决定——**说，还是不说；对谁说；说什么**。

因此提示词里「可以不说话」必须写在最前面。写成「请说一句晚安」，
模型就会每天准点发一句，久了就是一句她自己也不信的仪式；写成
「你今天还有没有话要说」，答案才可能是「没有」。
"""

from __future__ import annotations

from typing import Sequence

from ..common.time_utils import format_clock

# 单条消息上限：告别与问候都是一句话的事，超过就变成汇报
_MAX_TEXT_CHARS = 80

# 每个会话保留的最后几条发言：够看出「刚才聊到哪儿」即可
_MAX_TRANSCRIPT_LINES = 6

_ACTION_OPEN = "<wakeup_call>"
_ACTION_CLOSE = "</wakeup_call>"

_FORMAT_NOTE = (
    "只回一个 JSON，不要解释：\n"
    '{"speak": false}\n'
    "或者\n"
    '{"speak": true, "session": "会话标签、群号或 QQ 号", "text": "你要说的话"}'
)


def _persona_line(persona: str) -> str:
    return f"你的人格设定：\n{persona}\n" if persona else ""


def _transcript_block(transcripts: Sequence[tuple[str, str, str]]) -> str:
    """把会话片段渲染成「谁、在哪个会话、说了什么」。"""
    if not transcripts:
        return ""
    chunks: list[str] = []
    for label, _chat_id, body in transcripts:
        chunks.append(f"【{label}】\n{body}")
    return "\n\n".join(chunks)


def build_farewell_prompt(
    *,
    now: float,
    persona: str = "",
    transcripts: Sequence[tuple[str, str, str]] = (),
    max_text_chars: int = _MAX_TEXT_CHARS,
) -> str:
    """入睡时刻的唤醒提示：今天最后一次开口的机会。"""
    body = [
        "---",
        f"现在是 {format_clock(now)}，你准备睡了。",
        "",
        "这不是一次对话：没有人在等你回话，也没有哪一句非说不可。",
        "这是你今天最后一次开口的机会——想说就说，不想说就安静地睡。",
        "",
        _persona_line(persona) + "你最近在跟这些人说话：",
        _transcript_block(transcripts) or "（今天没有人找过你）",
        "",
        "如果要说话：",
        f"- 只挑一个会话，一句话，不超过 {max_text_chars} 字。不要挨个告别。",
        "- 挑你今天真的在聊、而且这个点还醒着的人。群聊里随口道晚安会很奇怪，私聊更自然。",
        "- 告别要像人：不要说「我去睡了」「我要休息了」「明天见」，也不要总结今天的聊天。"
        "人是随口一句「早点睡」「我先撤了」，不是提交日报。",
        "- 「session」必须填上面出现过的会话标签、群号或 QQ 号，填不准就别说话。",
        "",
        _FORMAT_NOTE,
        "---",
    ]
    return "\n".join([_ACTION_OPEN, *body, _ACTION_CLOSE])


def build_greeting_prompt(
    *,
    now: float,
    sleep_at: float = 0.0,
    persona: str = "",
    total: int = 0,
    unseen: Sequence[tuple[str, str]] = (),
    transcripts: Sequence[tuple[str, str, str]] = (),
    max_text_chars: int = _MAX_TEXT_CHARS,
) -> str:
    """起床时刻的唤醒提示：昨晚的消息要不要回。

    `unseen` 是「睡着时有人说话的会话」，`total` 是这段时间入账的条数。
    两者都要给：只给条数她不知道找谁，只给会话她不知道有多少在等。
    """
    if sleep_at:
        slept = f"你从 {format_clock(sleep_at)} 一直睡到现在。"
    else:
        slept = "你刚睡醒。"

    if total <= 0:
        night = "你睡着的时候没有人找过你。"
    else:
        night = f"你睡着的时候收到了 {total} 条消息。"
        if unseen:
            names = "、".join(label for label, _chat in unseen[:6])
            night = f"{night}这些东西来自：{names}。"

    body = [
        "---",
        f"现在是 {format_clock(now)}。你刚睡醒。{slept}",
        "",
        "起床这件事只有你自己知道——不要主动告诉别人你睡了多久、刚醒、"
        "睡了几个小时。人不会这么讲话。",
        "",
        _persona_line(persona) + night,
        "",
        "这些消息是**你睡着的时候**收到的，你只是现在才翻到：不是刚刚发生的事，"
        "也不是有人正站在你面前等你回话。半夜随口说的一句和清早认真的问候，"
        "分量不一样，别用同一句话回。",
        "",
        _transcript_block(transcripts) or "（没有留下什么内容）",
        "",
        "要不要开口，你自己决定：",
        "- 不想说话就不说。早上没有规定必须发早安。",
        f"- 想说就只挑一个会话，一句话，不超过 {max_text_chars} 字。不要挨个回。",
        "- 不要复述对方说了什么，也不要说「我看到你昨晚发的消息」。",
        "- 「session」必须填上面出现过的会话标签、群号或 QQ 号，填不准就别说话。",
        "",
        _FORMAT_NOTE,
        "---",
    ]
    return "\n".join([_ACTION_OPEN, *body, _ACTION_CLOSE])
