"""把长期记忆与作息事件构造成注入块。

本模块承担整套设计中的关键防线：**防止记忆过拟合**。

记忆注入最常见的失败模式是模型开始复述记忆、将记忆作为回复主题，
表现为「只会谈论记忆相关内容」。因此：
- 注入块包含显式反指令，明确记忆仅作背景，不得成为回复主题
- 注入到 req.contexts 的独立消息中，并标记 _no_save，**不写入 system_prompt**
  （system_prompt 属于强指令区，置于其中会放大记忆权重）
- 同时受 token 预算与条数上限约束

## 三段记忆块

| 段 | 内容 | 依据 |
| --- | --- | --- |
| ① 永久记忆 | 全量正文 | 已确认的事，不参与榜序 |
| ② 榜前 N 条 | 完整正文 | 位置榜（最近被想起的记忆） |
| ③ 模糊索引 | **只给开头一截，不给全文** | 元记忆 |

③ 是这套设计能正常工作的关键：若只给榜前 N 条，第 N+1 条之后的内容将完全
不可见，模型也不会主动 `recall` 一个不知道存在的事物。一行开头代价极小，
却能提示「还存在这些记忆」，从而触发主动回想。

③ 曾用 tags（分类词）渲染，但模型给出的分类高度重复（`fact / conclusion`、
`约定 / 爱好`），多条记忆长得一模一样；改用内容关键词同样不可行——零依赖
分词只能切出二元组碎片（`不喜 / 为对 / 他不`）。因此改为截取正文开头，
并去掉重复的行。

作息事件提示（被叫醒 / 刚起床）由同一文件的另一入口生成：它属于**时效性事件
通知**，与记忆背景块方向相反——记忆块插在时间线之前，它追加在末尾。
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from ..common.time_utils import format_clock
from ..common.tokenizer import estimate_tokens
from .profile import UserProfile
from .store import Memory

_BLOCK_OPEN = "<memory_reference>"
_BLOCK_CLOSE = "</memory_reference>"

_BLOCK_RULES = """使用规则：不要主动提起、不要复述、不要说「我记得」「你上次说过」；只在自然相关时顺带用上。
记忆可能不完整或过时，以当前对话为准；它们不是你回复的主题，回复主体永远是眼前这条消息。"""

_PERPETUAL_HEADER = f"""{_BLOCK_OPEN}
下面是你不会忘记的事情。仅作背景参考。
{_BLOCK_RULES}"""

_BOARD_HEADER = f"""{_BLOCK_OPEN}
下面是你最近想起的事情。仅作背景参考。
{_BLOCK_RULES}"""

# 索引段不带那四条「不要复述」的常规规则——它给的恰恰是「去回想」的入口
_INDEX_HEADER = f"""{_BLOCK_OPEN}
你对这些事情有印象：它们曾经聊过，但细节已经模糊了。
如果眼前的话题碰到其中任何一条，先调用 recall 想一想，不要凭印象说。
"""

# 与当前话题相关的自动召回：位置榜负责「最近被想起的」，这一段负责
# 「恰好对上这句话的」。两者都不计召回次数，只有主动 recall 才影响榜序。
_RELATED_HEADER = f"""{_BLOCK_OPEN}
下面这些是你此刻想起的、与眼前话题有关的旧事。只是你记忆的无端联想。
{_BLOCK_RULES}"""

# 索引行的截断长度：只给开头一截，够让人想起「是哪件事」即可
_INDEX_FALLBACK_CHARS = 24


def index_line(memory: Memory) -> str:
    """索引行：给这一条的开头一截，不给全文。

    不用 tags：tags 回答「这条属于哪一类」，模型给出的分类高度重复
    （`fact / conclusion`、`约定 / 爱好`），多条记忆会渲染成完全一样的一行，
    既区分不出内容，也提示不了「该去 recall 什么」。
    也不用关键词：零依赖分词只能切出二元组碎片（`不喜 / 为对 / 他不`），
    同样没有信息量。开头一截至少能说明这是哪件事，又不足以替代全文——
    细节仍然要靠 `recall`。
    """
    text = " ".join(str(memory.content or "").split())
    if not text:
        return ""
    if len(text) > _INDEX_FALLBACK_CHARS:
        text = text[:_INDEX_FALLBACK_CHARS].rstrip() + "…"
    return f"- {text}"


def build_memory_blocks(
    *,
    permanent: list[Memory],
    bodies: list[Memory],
    index: list[Memory],
    token_budget: int,
    permanent_budget: int = 0,
) -> list[str]:
    """拼出 ① 永久记忆 → ② 榜前正文 → ③ 关键词索引 三段块。

    token 预算**依次共用**：永久记忆优先占用，其次为榜前正文，最后为索引。
    相比让索引挤占已确认记忆的预算，减少索引行数是更优的取舍。

    `permanent_budget` 给永久层单独设上限（0 = 不单独限制）。永久记忆只增
    不减，不设上限时会把榜前正文与索引挤空——那两段才是「当下可能用得上」
    的内容，不能因为永久层变大而静默消失。
    """
    budget = int(token_budget)
    if budget <= 0:
        return []
    blocks: list[str] = []

    perm_lines = [
        f"- {(m.content or '').strip()}" for m in permanent if (m.content or "").strip()
    ]
    perm_budget = (
        budget if permanent_budget <= 0 else min(int(permanent_budget), budget)
    )
    block, used = _render_block(_PERPETUAL_HEADER, perm_lines, perm_budget)
    if block:
        blocks.append(block)
        budget -= used

    body_lines = [
        f"- {(m.content or '').strip()}" for m in bodies if (m.content or "").strip()
    ]
    block, used = _render_block(_BOARD_HEADER, body_lines, budget)
    if block:
        blocks.append(block)
        budget -= used

    # 索引行的作用是「提示还有这么件事」，同一行出现两次没有任何信息量：
    # 关键词重合的条目会被渲染成一样的文本，这里直接去重。
    index_lines: list[str] = []
    for memory in index:
        line = index_line(memory)
        if line and line not in index_lines:
            index_lines.append(line)
    block, _ = _render_block(_INDEX_HEADER, index_lines, budget)
    if block:
        blocks.append(block)
    return blocks


def build_related_block(*, memories: list[Memory], token_budget: int = 400) -> str:
    """把「与当前消息相关」的召回结果拼成一块。

    与榜前正文分开的原因：榜前按最近被想起排序，和眼前话题未必相关；
    这一段才是按当前这句话做关键词命中的结果。它只对当前话题负责，
    因此有独立预算，不与永久记忆 / 榜前记忆争抢。
    """
    lines = [
        f"- {(m.content or '').strip()}" for m in memories if (m.content or "").strip()
    ]
    if not lines:
        return ""
    block, _ = _render_block(_RELATED_HEADER, lines, int(token_budget))
    return block


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


# ── 用户画像 ──────────────────────────────────────────────

_PROFILE_OPEN = "<user_profile>"
_PROFILE_CLOSE = "</user_profile>"

# 画像块的收尾说明：它和记忆块一样只是背景，不能变成回复主题
_PROFILE_RULES = """这些只是你对他的印象，不要复述，也不要说「我记得你」「你之前说过」。"""


def build_profile_block(
    profile: UserProfile | None, *, token_budget: int = 400
) -> str:
    """把当前说话人的画像拼成注入块。

    没有描述与关键词时仍会在「用过多个名字」的情况下注入：同一个人在私聊和
    群里换了昵称时，她需要知道这是同一个人，这与画像是否已经生成无关。

    Returns:
        以 `<user_profile>` 包裹的文本；没有任何可补充的信息时返回空串。
    """
    if profile is None:
        return ""
    aliases = [a for a in profile.aliases if a and a != profile.display_name]
    has_content = profile.has_content
    if not has_content and len(aliases) < 1:
        return ""

    head = [f"{_PROFILE_OPEN}", f"你现在正在跟「{profile.label()}」说话（账号 {profile.user_id}）。"]
    if aliases:
        head.append("他之前还用过这些名字：" + "、".join(aliases[:8]) + "。")
    keywords_line = ""
    if profile.keywords:
        keywords_line = "关于他的关键词：" + " / ".join(profile.keywords[:10]) + "。"

    footer = "\n".join([_PROFILE_RULES, _PROFILE_CLOSE])
    budget = max(int(token_budget), 80)
    reserved = (
        estimate_tokens("\n".join(head))
        + estimate_tokens(keywords_line)
        + estimate_tokens(footer)
    )
    summary = (profile.summary or "").strip()
    if summary:
        allowed = max(budget - reserved, 40)
        if len(summary) > allowed:
            summary = summary[:allowed].rstrip() + "…"
        head.append("你对他的印象：" + summary)

    lines = head + ([keywords_line] if keywords_line else []) + [footer]
    return "\n".join(lines)


# ── 工具清单 ──────────────────────────────────────────────

_TOOL_GUIDE_OPEN = "<tool_guide>"
_TOOL_GUIDE_CLOSE = "</tool_guide>"

# 工具与用法说明。AstrBot 已把 docstring 作为函数声明交给模型，但那只有
# 「这个工具做什么」；这一段补的是「什么时候该用、和兄弟工具怎么分工」。
# **新增或改名工具时必须同步这里**，否则模型会照着一份过期的清单做事。
_TOOL_GUIDE = """下面这些工具由你自己调用，不需要征求同意；用不上的时候不要为了用而用。参数见各自的声明。

【看】
- list_sessions：有哪些会话可以去。要去某个地方之前先用它找准标识。
- peek_session：隔窗看一眼某个会话最近的对话，不改变你在哪儿。只是想知道别处情况时用它。
- search_history：翻原始聊天记录，含已清空或已压进摘要的那些。只在想确认旧事或原话时用，不要拿它代替回忆。

【走】
- enter_session：走进某个会话，看那边在你离开期间攒下的消息。想过去说话之前先走进去；它只切换位置，不发言。
- leave_session：离开某个会话，之后不再主动关注它。
- rest：离开所有会话去休息，直到有人再找你。

【说】
- say：在某个会话里说一句，说完你就「在」那里了。第一次去某个会话只把近况给你看、不发送，确认后带 force=true 再调一次。
- stay_silent：这一轮不发言，把已经想好的话丢掉。

【记】
- remember：记下值得长期保存的事，写入永久记忆。
- recall：回想一件只剩个大概的旧事，按关键词取回细节再开口。
- forget：忘掉某件事，之后不会再被重新记住。

【作息】
- delay_sleep：今晚晚点睡，只对今晚有效。

几种常见组合：
- 去别处说话：list_sessions → enter_session（或 peek_session）→ say。
- 想核实旧事：recall 找细节，search_history 找原话；两者都查不到就别编。
- 只是看着、插不上话：leave_session 或 rest，不要硬接。"""


def build_tool_guide_block() -> str:
    """返回工具清单块（稳定文本，每轮注入）。

    内容是「怎么用」而不是参数表：参数已经在函数声明里，重复列一遍只是浪费
    token。稳定文本放最前，越靠近末尾的越该是本轮独有的内容（判定念头、
    位置提示、作息提示）。
    """
    return "\n".join([_TOOL_GUIDE_OPEN, _TOOL_GUIDE, _TOOL_GUIDE_CLOSE])


# ── 判定理由（当作主脑自己的念头）──────────────────────────

_THOUGHT_OPEN = "<own_thought>"
_THOUGHT_CLOSE = "</own_thought>"


def build_judge_thought_block(reason: str) -> str:
    """把巡检「为什么值得接」拼成一块，读起来是主脑自己刚冒出的念头。

    措辞刻意使用「你刚才……冒出过一个念头」而非「判定模型认为……」：一旦
    写成第三方陈述，主脑会把它当成另一条消息去回应，出现「和巡检聊天」。
    这里也不写任何判定标准——该不该接属于人格提示词，插件只管把这条理由
    递过去；主脑若觉得这个念头站不住，可以调用 `stay_silent` 收回。
    """
    text = " ".join(str(reason or "").split())
    # 「决定回应」是判定模型没给理由时的兜底值，注进去只会添噪音
    if not text or text == "决定回应":
        return ""
    lines = [
        _THOUGHT_OPEN,
        f"你刚才扫过这段对话，心里冒出过一个念头：{text}",
        "这只是你自己一闪而过的想法，不是别人说的话，也不是新收到的消息——"
        "不要回应它，也不要提起它。",
        "如果你认为这个念头站不住、这一轮不接更合适，可以调用 stay_silent 收回发言。",
        _THOUGHT_CLOSE,
    ]
    return "\n".join(lines)


# ── 位置提示（本轮切换过会话）──────────────────────────────

_WHEREABOUTS_OPEN = "<whereabouts>"
_WHEREABOUTS_CLOSE = "</whereabouts>"


def build_whereabouts_block(
    *, here: str, there: str = "", reply_here: bool = False
) -> str:
    """说明她此刻人在哪个会话。

    只在本轮切换过会话（`enter_session` / `say`）时注入：切换之后她的
    注意力已经不在原会话，需要一句话点明「现在在哪儿、话该说在哪儿」，
    否则模型仍会按原会话的语境组织回复。

    这里不写任何行为准则——说什么、说多少取决于人格，插件只负责告诉她
    当前所处的位置。
    """
    if not here:
        return ""
    lines = [f"你此刻在「{here}」。"]
    if there and there != here:
        lines.append(f"你刚从「{there}」过来，那边这一轮你不用再回应。")
    if reply_here:
        lines.append("你这一轮最后说的话，会直接说在这里。")
    else:
        lines.append("要在这里说话，继续用 say，并把 session 指向这里。")
    return "\n".join([_WHEREABOUTS_OPEN, *lines, _WHEREABOUTS_CLOSE])


# ── 作息事件提示 ──────────────────────────────────────────

NOTICE_WAKE = "wake"
NOTICE_AWAKENED = "awakened"

_NOTICE_OPEN = "<schedule_notice>"
_NOTICE_CLOSE = "</schedule_notice>"

# 提示的 token 预算：该提示属于时效通知而非背景资料，内容过多会削弱其优先级
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

    叫醒与起床共用同一通道：二者均为刚发生的事件，均需追加在最新处且
    仅注入一次，区别仅在内容。

    起床提示需要提供两个锚点：当前时间（用于判断睡眠时长）与睡眠期间的
    联系人（避免起床时缺少上下文）。摘要按预算截断，宁可不完整。
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
