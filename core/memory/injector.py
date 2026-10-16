"""把召回的记忆构造成注入块。

这里承担整套设计里最关键的防线：**防止记忆过拟合**。

记忆注入最常见的失败模式是模型开始复述记忆、把记忆当成回复主题，
变成「只会说记忆相关内容」。因此：
- 注入块带显式反指令，明确记忆只是背景，不得成为回复主题
- 注入到 req.contexts 的独立消息里，且标 _no_save，**不写进 system_prompt**
  （system_prompt 属于强指令区，放那里会放大记忆的权重）
- 受 token 预算与条数上限双重约束
- 得分低于阈值的记忆在召回阶段就已被过滤，宁缺毋滥
"""

from __future__ import annotations

import time

from .memory_store import RecalledMemory
from .tokenizer import estimate_tokens

_BLOCK_OPEN = "<memory_reference>"
_BLOCK_CLOSE = "</memory_reference>"

_BLOCK_HEADER = """{open}
下面是你脑海中浮现的一些零散记忆片段，仅作背景参考。

使用规则：
1. 不要主动提起这些记忆，不要复述，不要说「我记得」「你上次说过」这类话。
2. 只有当它们与当前话题自然相关时，才像人一样顺带用上。
3. 记忆可能不完整、过时或有误，一切以当前对话为准。
4. 这些记忆绝不能成为你回复的主题，你的回复主体永远是回应眼前这条消息。
""".format(open=_BLOCK_OPEN)

_BLOCK_FOOTER = _BLOCK_CLOSE


_RECENT_HEADER = """<your_recent_experience>
这是你最近亲身经历的事，方括号内是你当时所在的会话。
这些不是别人的对话，而是你自己听到和说过的话——你一直是同一个人，
只是同时在几个地方活动。

- 你现在正在当前会话里说话，不要在这些片段里找「该回复谁」
- 不要用旁观者的口吻谈论它们，不要说「我查了一下」「那边的记录显示」这类话
- 只有在与当前话题自然相关时，才像人一样自然地带出来
- 不要逐条复述，也不要说明这些信息的来源
"""

def build_recent_context_block(body: str) -> str:
    """把跨会话近期原始消息包成注入块。"""
    body = (body or "").strip()
    if not body:
        return ""
    return f"{_RECENT_HEADER}{body}\n</your_recent_experience>"


def build_injected_entry(block: str) -> dict:
    """构造一条注入条目。

    单独暴露构造、而不是直接追加到 req.contexts，是因为注入顺序就是模型的
    读取顺序：我们需要把背景块**放在原生历史之前**，而不是一律追加到末尾。
    """
    return {
        "role": "user",
        "content": block,
        "_no_save": True,
        # 供排障：注入时刻，便于对照日志确认是否命中
        "_shittim_injected_at": time.time(),
    }


def build_block(
    memories: list[RecalledMemory], token_budget: int
) -> tuple[str, list[int]]:
    """构造注入块，同时返回真正被写进块里的记忆 id 列表。

    返回 id 列表是为了只对这些记忆计入召回次数——没进块的不该被算作「被想起」。
    """
    if not memories or token_budget <= 0:
        return "", []

    lines: list[str] = []
    used_ids: list[int] = []
    used_tokens = estimate_tokens(_BLOCK_HEADER) + estimate_tokens(_BLOCK_FOOTER)

    for recalled in memories:
        content = (recalled.memory.content or "").strip()
        if not content:
            continue
        line = f"- {content}"
        cost = estimate_tokens(line)
        if used_tokens + cost > token_budget:
            break
        lines.append(line)
        used_ids.append(recalled.memory.id)
        used_tokens += cost

    if not lines:
        return "", []

    block = "\n".join([_BLOCK_HEADER, *lines, _BLOCK_FOOTER])
    return block, used_ids


def format_for_log(memories: list[RecalledMemory]) -> str:
    """一行摘要，供日志确认本次召回了什么。"""
    if not memories:
        return "无"
    parts = [
        f"#{r.memory.id}({r.channel},{r.score:.2f}){r.memory.content[:18]}"
        for r in memories
    ]
    return " | ".join(parts)
