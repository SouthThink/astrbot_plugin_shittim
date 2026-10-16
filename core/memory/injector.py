"""把长期记忆与作息事件构造成注入块。

这里承担整套设计里最关键的防线：**防止记忆过拟合**。

记忆注入最常见的失败模式是模型开始复述记忆、把记忆当成回复主题，
变成「只会说记忆相关内容」。因此：
- 注入块带显式反指令，明确记忆只是背景，不得成为回复主题
- 注入到 req.contexts 的独立消息里，且标 _no_save，**不写进 system_prompt**
  （system_prompt 属于强指令区，放那里会放大记忆的权重）
- 受 token 预算与条数上限双重约束

## 三段记忆块

| 段 | 内容 | 依据 |
| --- | --- | --- |
| ① 永久记忆 | 全量正文 | 她确信的事，不参与榜单 |
| ② 榜前 N 条 | 完整正文 | 位置榜（谁是最近被想起的） |
| ③ 关键词索引 | **只给一行词，不给正文** | 元记忆 |

③ 是这套设计能跑起来的关键：只给榜前 N 条的话，第 N+1 条之后对她完全不可见，
她不会去 `recall` 一个不知道存在的东西。一行关键词代价极小，却告诉她
「你还忘了这些」，她才会主动去想。

作息事件提示（被叫醒 / 刚起床）走同一个文件的另一个入口：它是**时效性事件
通知**，与记忆背景块方向相反——记忆块插在时间线之前，它追加在最新处。
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from ..common.time_utils import format_clock
from ..common.tokenizer import estimate_tokens
from .store import Memory

_BLOCK_OPEN = "<memory_reference>"
_BLOCK_CLOSE = "</memory_reference>"

_BLOCK_RULES = """使用规则：
1. 不要主动提起这些记忆，不要复述，不要说「我记得」「你上次说过」这类话。
2. 只有当它们与当前话题自然相关时，才像人一样顺带用上。
3. 记忆可能不完整、过时或有误，一切以当前对话为准。
4. 这些记忆绝不能成为你回复的主题，你的回复主体永远是回应眼前这条消息。
"""

_PERPETUAL_HEADER = f"""{_BLOCK_OPEN}
下面是你确信的事——长期相处中确认过、不会轻易改变的，仅作背景参考。
{_BLOCK_RULES}"""

_BOARD_HEADER = f"""{_BLOCK_OPEN}
下面是你记得最牢的事（按「最近一次被想起」的顺序排）。仅作背景参考。
{_BLOCK_RULES}"""

# 索引段不带那四条「不要复述」的常规规则——它给的恰恰是「去回想」的入口
_INDEX_HEADER = f"""{_BLOCK_OPEN}
后面这些你**只记得个大概**：它们躺在记忆深处，细节已经模糊了。
如果眼前的话题碰到其中任何一条，先调用 recall 想一想，不要凭印象说。
"""

# 索引行最多并列几个词：它只是「提示有这么件事」，不是内容本身
_INDEX_TERMS_MAX = 6
# 取不到标签与关键词时的兜底长度
_INDEX_FALLBACK_CHARS = 24


def index_line(memory: Memory) -> str:
    """索引行：只给标签或关键词，不给正文。

    标签是模型生成的分类词，比 char bigram 关键词更可读，所以优先用它。
    """
    terms = [t.strip() for t in memory.tags if t and t.strip()]
    if not terms:
        terms = [k.strip() for k in memory.keywords if k and k.strip()]
    if terms:
        return "- " + " / ".join(terms[:_INDEX_TERMS_MAX])
    text = " ".join(str(memory.content or "").split())
    return f"- {text[:_INDEX_FALLBACK_CHARS]}" if text else ""


def build_memory_blocks(
    *,
    permanent: list[Memory],
    bodies: list[Memory],
    index: list[Memory],
    token_budget: int,
) -> list[str]:
    """拼出 ① 永久记忆 → ② 榜前正文 → ③ 关键词索引 三段块。

    token 预算**依次共用**：永久记忆先占，剩下的才轮到榜前正文，最后才是索引。
    与其让索引挤掉她确信的事，不如索引少列两行。
    """
    budget = int(token_budget)
    if budget <= 0:
        return []
    blocks: list[str] = []
    for header, memories in ((_PERPETUAL_HEADER, permanent), (_BOARD_HEADER, bodies)):
        lines = [
            f"- {(m.content or '').strip()}" for m in memories if (m.content or "").strip()
        ]
        block, used = _render_block(header, lines, budget)
        if block:
            blocks.append(block)
            budget -= used
    index_lines = [line for line in (index_line(m) for m in index) if line]
    block, _ = _render_block(_INDEX_HEADER, index_lines, budget)
    if block:
        blocks.append(block)
    return blocks


def _render_block(header: str, lines: list[str], budget: int) -> tuple[str, int]:
    """按预算裁剪成一块，返回 (块, 实际占用)。放不下就返回空串。"""
    if budget <= 0 or not lines:
        return "", 0
    used = estimate_tokens(header) + estimate_tokens(_BLOCK_CLOSE)
    if used > budget:
        return "", 0
    kept: list[str] = []
    for line in lines:
        cost = estimate_tokens(line)
        if used + cost > budget:
            break
        kept.append(line)
        used += cost
    if not kept:
        return "", 0
    return "\n".join([header, *kept, _BLOCK_CLOSE]), used


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


# ── 作息事件提示 ──────────────────────────────────────────

NOTICE_WAKE = "wake"
NOTICE_AWAKENED = "awakened"

_NOTICE_OPEN = "<schedule_notice>"
_NOTICE_CLOSE = "</schedule_notice>"

# 提示的 token 预算：它是时效通知不是背景资料，塞太多会喧宾夺主
_NOTICE_TOKEN_BUDGET = 400


def build_schedule_notice(
    *,
    kind: str,
    now: float,
    sleep_at: float = 0.0,
    total: int = 0,
    digest: Sequence[tuple[str, int, str]] = (),
    token_budget: int = _NOTICE_TOKEN_BUDGET,
) -> str:
    """构造作息事件提示（被叫醒 / 刚起床）。

    叫醒与起床共用同一个通道：它们都是「刚刚发生在她身上的事」，都要放在
    最新处、都只注入一次。区别只在内容。

    起床提示要给她两个锚点：现在几点（知道自己睡了多久）、睡着时谁找过她
    （否则她醒来面对的是"一片空白的一天"）。摘要按预算截断，宁可少说。
    """
    if kind == NOTICE_AWAKENED:
        body = [f"你被叫醒了，现在是 {format_clock(now)}。"]
        return "\n".join([_NOTICE_OPEN, *body, _NOTICE_CLOSE])

    head = [f"你刚睡醒，现在是 {format_clock(now)}。"]
    if sleep_at:
        head.append(f"你从 {format_clock(sleep_at)} 一直睡到现在。")
    if total <= 0:
        head.append("")
        head.append("你睡着的时候没有人找过你。")
        return "\n".join([_NOTICE_OPEN, *head, _NOTICE_CLOSE])

    head.append("")
    head.append(f"你睡着的时候收到了 {total} 条消息，来自这几个会话：")
    used = estimate_tokens(_NOTICE_OPEN) + estimate_tokens(_NOTICE_CLOSE)
    used += sum(estimate_tokens(line) for line in head)

    lines: list[str] = []
    for label, count, last_line in digest:
        text = (last_line or "").strip() or f"[{label}]"
        line = f"- {text}（该会话共 {count} 条）"
        cost = estimate_tokens(line)
        if used + cost > token_budget:
            break
        lines.append(line)
        used += cost
    if not lines:
        return "\n".join([_NOTICE_OPEN, *head, _NOTICE_CLOSE])

    tail = "你可以自己决定要不要去这些会话看看，不必都回。"
    return "\n".join([_NOTICE_OPEN, *head, *lines, tail, _NOTICE_CLOSE])


def format_for_log(memories: list[RecalledMemory]) -> str:
    """一行摘要，供日志确认本次召回了什么。"""
    if not memories:
        return "无"
    parts = [
        f"#{r.memory.id}({r.channel},{r.score:.2f}){r.memory.content[:18]}"
        for r in memories
    ]
    return " | ".join(parts)
