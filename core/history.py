"""从 AstrBot 官方会话历史回填短期记忆。

我们的短期账本只记「插件在场时」的消息。插件装上之前、插件被禁用或出错期间
的对话，AstrBot 自己有记录，但对她不可见。这个模块把它们补进来，让她的
短期记忆尽量完整。

## 对应关系

    unified_msg_origin
        └─> conversation_manager.get_curr_conversation_id(umo)   → conversation_id
        └─> conversation_manager.get_conversation(umo, cid)      → Conversation
        └─> json.loads(conversation.history)                     → OpenAI 消息数组

## 已知的信息损失

AstrBot 的历史只存 role 与 content，**不含发送者**。所以回填进群聊的
user 消息无法知道是谁说的，只能标记为「旧记录」。这是数据源的硬限制，
不是实现偷懒。
"""

from __future__ import annotations

import json
import os
from collections import deque

try:
    from astrbot.api import logger
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

# 只回填这两个角色：tool 结果与纯工具调用的 assistant 对记忆没有价值
_ALLOWED_ROLES = ("user", "assistant")

# 上游只接受这几种图片引用。其余（本地路径）由 AstrBot 在 provider 层转 base64；
# 转换失败的会**原样透传**，严格的上游就会直接 400 并让整条请求失败。
_REMOTE_IMAGE_PREFIXES = ("data:", "http://", "https://")
_FILE_PREFIX = "file://"

# 整条消息只剩坏图时留下的文本，让「这里本来有张图」这件事不至于彻底消失
BROKEN_IMAGE_PLACEHOLDER = "[图片已失效]"


def is_usable_image_ref(ref: str, cache_path: str = "") -> bool:
    """这个图片引用能不能安全地交给上游。

    远端引用与 data URL 一律放行；本地路径要求文件确实存在且非空，
    否则它一定会被原样透传出去触发 400。
    """
    ref = (ref or "").strip()
    if not ref:
        return False
    if ref.startswith(_REMOTE_IMAGE_PREFIXES):
        return True
    path = (cache_path or "").strip()
    if not path:
        path = ref[len(_FILE_PREFIX) :] if ref.startswith(_FILE_PREFIX) else ref
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def _image_ref_of(part: dict) -> str:
    image = part.get("image_url")
    if isinstance(image, dict):
        url = image.get("url")
        return url.strip() if isinstance(url, str) else ""
    return image.strip() if isinstance(image, str) else ""


def strip_invalid_images(contexts: list) -> tuple[list, list[str]]:
    """摘掉读不到的图片引用，返回 (处理后的 contexts, 被摘掉的引用)。

    为什么必须做：一张失效的图片会让**整条请求**被上游 400 拒绝，而 AstrBot
    随后会把这一轮切给备用模型——语气突变、工具调用多半失效。代价远大于
    「这张图没传过去」。这里把确定读不到的摘掉，让坏图只影响它自己。
    """
    if not isinstance(contexts, list):
        return contexts, []

    dropped: list[str] = []
    result: list = []
    for entry in contexts:
        if not isinstance(entry, dict):
            result.append(entry)
            continue
        content = entry.get("content")
        if not isinstance(content, list):
            result.append(entry)
            continue

        kept: list = []
        for part in content:
            if not (isinstance(part, dict) and part.get("type") == "image_url"):
                kept.append(part)
                continue
            ref = _image_ref_of(part)
            cache_path = part.get("cache_path")
            if is_usable_image_ref(
                ref, cache_path if isinstance(cache_path, str) else ""
            ):
                kept.append(part)
                continue
            dropped.append(ref or "<空引用>")

        if len(kept) == len(content):
            result.append(entry)
            continue
        new_entry = dict(entry)
        # 整条消息只剩坏图时补一个文本占位，别让这条消息变成空内容
        new_entry["content"] = kept or [
            {"type": "text", "text": BROKEN_IMAGE_PLACEHOLDER}
        ]
        result.append(new_entry)

    return result, dropped


def strip_invalid_image_urls(urls: list) -> tuple[list, list[str]]:
    """同上，但作用于 req.image_urls 这种「字符串列表」形式。"""
    if not isinstance(urls, list):
        return urls, []
    kept: list = []
    dropped: list[str] = []
    for url in urls:
        if isinstance(url, str) and is_usable_image_ref(url):
            kept.append(url)
        else:
            dropped.append(str(url))
    return kept, dropped


def extract_text(content) -> str:
    """从 OpenAI 格式的 content 里取出可读文本。"""
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type == "text":
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        elif part_type == "image_url":
            parts.append("[图片]")
    return " ".join(parts).strip()


def convert_records(records: list, limit: int) -> list[dict]:
    """把 AstrBot 历史转成短期账本能用的 [{role, content}]，按时间正序。

    从最新往回取够 limit 条，因此拿到的是「最近的」历史。
    """
    if not isinstance(records, list) or limit <= 0:
        return []
    selected: list[dict] = []
    for record in reversed(records):
        if not isinstance(record, dict):
            continue
        role = record.get("role")
        if role not in _ALLOWED_ROLES:
            continue
        text = extract_text(record.get("content"))
        if not text:
            # assistant 只调工具不說話、或内容为空的消息直接跳过
            continue
        selected.append({"role": role, "content": text})
        if len(selected) >= limit:
            break
    selected.reverse()
    return selected


def align_records(
    records: list[dict], known_rows: list, now: float
) -> list[tuple[dict, float, bool]]:
    """把 AstrBot 历史记录对齐到账本的时间轴。

    沿 AstrBot 历史的时间顺序与账本做锚点匹配：匹配上的记录跳过（我们已有），
    同时把它的时间当作锚点；没匹配上的记录就落到当前锚点稍后的位置。

    返回值 [(记录, 时间戳, 是否早于账本里的一切)]。第三项决定这条回填要不要
    标成 gated 由我们兜底注入——比账本里所有记录都早的那批，正是 AstrBot 最
    可能压缩掉的旧历史；而落在实时消息之间的，AstrBot 自己会保留。

    known_rows: 账本侧记录，需含 role / content / created_at 且按时间正序。
    """
    pools: dict[tuple[str, str], deque] = {}
    earliest_known: float | None = None
    for row in known_rows:
        created_at = float(row["created_at"])
        if earliest_known is None or created_at < earliest_known:
            earliest_known = created_at
        pools.setdefault((str(row["role"]), str(row["content"])), deque()).append(
            created_at
        )

    deferred: list[dict] = []  # 还没出现任何锚点前遇到的新记录
    placed: list[tuple[dict, float, bool]] = []
    cursor: float | None = None
    for record in records:
        pool = pools.get((record["role"], record["content"]))
        if pool:
            cursor = pool.popleft()
            continue
        if cursor is None:
            deferred.append(record)
        else:
            cursor += 1e-3
            placed.append((record, cursor, False))

    base = (earliest_known if earliest_known is not None else now) - len(deferred) - 1.0
    for index, record in enumerate(deferred):
        placed.append((record, base + index, True))

    placed.sort(key=lambda item: item[1])
    return placed


def merge_missing_into_contexts(
    contexts: list, ledger_rows: list, current_prompt: str = ""
) -> list:
    """把账本里有、AstrBot 的 contexts 里没有的消息，插回正确的时间位置。

    ## 为什么需要这个

    `context.send_message` 发出的消息**不进 AstrBot 的会话管线**，所以它
    永远不在 `req.contexts` 里。自那个工具接管跨会话发言后，她在 B 会所说
    的「在吗」就变成了主脑眼里的空白——她自己都不记得说过。

    以前的做法是额外附加一个「你的近期经历」块。但那读起来是**背景资料**，
    她会用「我查了一下」这种旁观者口吻谈论自己的话。合并进 contexts 之后，
    它就是一条真正的 assistant 回合——「我说过的话」。

    ## 为什么不全量覆盖

    官方文档推荐 `req.contexts = 自己重建的历史`（天使之心也这么做），但那样
    会丢掉 AstrBot contexts 里的**工具调用链与多模态图片**。这里改成合并：
    保留原生条目，只插入它缺的那些。

    ## 只处理实时记录的消息

    回填的历史又旧又多（单会话可能上百条），塞进 contexts 会直接撑爆窗口，
    所以它们不在这里处理，交给带 token 预算的跨会话语境块兜底。

    ## 当前消息

    `current_prompt` 承载着当前这条用户消息，它绝不能出现在 contexts 里，
    否则模型会以为是两轮。账本里记着它（记账发生在决断层之前），这里按
    「最后一条内容相同的 user 消息」把它剔掉。

    ## 调用时机

    必须在追加任何注入块**之前**调用，且只喂它 AstrBot 的原生 contexts。
    否则我们自己追加的块会被当成原生条目参与对齐，缺口位置会算错。
    """
    if not isinstance(contexts, list):
        return contexts

    # 在原生 contexts 里为「AstrBot 也有的消息」建位置索引
    positions: dict[tuple[str, str], deque] = {}
    for index, entry in enumerate(contexts):
        if not isinstance(entry, dict):
            continue
        key = (str(entry.get("role") or ""), extract_text(entry.get("content")))
        positions.setdefault(key, deque()).append(index)

    # 找出当前这条消息，它由 req.prompt 承载，不进 contexts
    prompt_text = (current_prompt or "").strip()
    skip_index = -1
    if prompt_text:
        for index in range(len(ledger_rows) - 1, -1, -1):
            row = ledger_rows[index]
            if (
                str(row["role"]) == "user"
                and str(row["content"] or "").strip() == prompt_text
            ):
                skip_index = index
                break

    inserts: dict[int, list[dict]] = {}
    cursor = 0
    for index, row in enumerate(ledger_rows):
        if index == skip_index:
            continue
        role = str(row["role"] or "user")
        text = str(row["content"] or "").strip()
        if not text:
            continue
        pool = positions.get((role, text))
        if pool:
            # AstrBot 自己就有这条：拿它当前进指针，后面的缺口插在它之后
            cursor = pool.popleft() + 1
            continue
        if int(row["gated"] or 0) != 1:
            continue
        if str(row["source"] or "live") == "backfill":
            continue
        inserts.setdefault(cursor, []).append({"role": role, "content": text})

    if not inserts:
        return contexts

    merged: list = []
    for index in range(len(contexts) + 1):
        merged.extend(inserts.get(index, ()))
        if index < len(contexts):
            merged.append(contexts[index])
    return merged


class ConversationHistoryReader:
    """读取 AstrBot 官方会话历史。"""

    def __init__(self, astr_context):
        self._astr_context = astr_context

    @property
    def available(self) -> bool:
        return (
            self._astr_context is not None
            and getattr(self._astr_context, "conversation_manager", None) is not None
        )

    async def read(self, chat_id: str, limit: int) -> list[dict]:
        """读某会话最近的历史，取不到就返回空列表（绝不抛异常）。"""
        if not chat_id or limit <= 0 or not self.available:
            return []
        manager = self._astr_context.conversation_manager
        try:
            conversation_id = await manager.get_curr_conversation_id(chat_id)
            if not conversation_id:
                return []
            conversation = await manager.get_conversation(chat_id, conversation_id)
            raw_history = getattr(conversation, "history", None) if conversation else None
            if not raw_history:
                return []
            records = json.loads(raw_history) if isinstance(raw_history, str) else raw_history
        except Exception as error:
            logger.warning("[什亭之枢] 读取 AstrBot 会话历史失败 | %s | %s", chat_id, error)
            return []
        return convert_records(records, limit)
