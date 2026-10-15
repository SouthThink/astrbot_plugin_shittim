"""零依赖的中文分词、关键词提取与相似度打分。

不引入 jieba 等分词库的原因：召回只需要「足够的召回率」，char bigram 在中文
短文本上表现稳定且完全确定，不依赖词典，也不会因词典缺失而失效。
"""

from __future__ import annotations

import math
import re
from collections import Counter

# 英文/数字整词
_ASCII_WORD = re.compile(r"[a-zA-Z0-9_]{2,}")
# 中文连续片段
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")

# 高频虚词与口语填充：参与打分只会引入噪声
_STOPWORDS = frozenset(
    {
        "的", "了", "是", "我", "你", "他", "她", "它", "在", "有", "和", "就",
        "都", "而", "及", "与", "着", "或", "一个", "没有", "我们", "你们",
        "他们", "什么", "这个", "那个", "这些", "那些", "还是", "但是", "因为",
        "所以", "如果", "可以", "不过", "然后", "现在", "已经", "自己", "这样",
        "那样", "怎么", "为什么", "时候", "知道", "觉得", "应该", "可能",
        "the", "and", "for", "you", "are", "not", "with", "that", "this",
    }
)


def tokenize(text: str) -> list[str]:
    """把文本切成检索用 token：中文 2-gram（短片段额外保留整段）+ 英文整词。"""
    if not text:
        return []
    tokens: list[str] = [m.group(0).lower() for m in _ASCII_WORD.finditer(text)]
    for match in _CJK_RUN.finditer(text):
        run = match.group(0)
        if len(run) == 1:
            tokens.append(run)
            continue
        # 短片段保留整体，避免「群聊」被切成「群聊」以外的碎片后无法整词命中
        if len(run) <= 4:
            tokens.append(run)
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


def extract_keywords(text: str, limit: int = 12) -> list[str]:
    """按出现频次提取关键词，过滤虚词，保持稳定顺序。"""
    if not text:
        return []
    counter = Counter(t for t in tokenize(text) if t not in _STOPWORDS and len(t) > 1)
    if not counter:
        return []
    # 频次优先，频次相同则更长的 token 优先（信息量更大）
    ranked = sorted(counter.items(), key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))
    return [token for token, _ in ranked[:limit]]


def overlap_score(query_tokens: set[str], keywords: list[str]) -> float:
    """记忆关键词被查询命中多少 → 0~1。

    分母用记忆自身的关键词数：关键词越冗长的记忆越难拿高分，
    这能抑制「什么都沾一点」的宽泛记忆抢占召回。
    """
    if not query_tokens or not keywords:
        return 0.0
    hits = sum(1 for keyword in keywords if keyword in query_tokens)
    if not hits:
        return 0.0
    return hits / len(keywords)


def estimate_tokens(text: str) -> int:
    """无 tokenizer 的启发式 token 估算：中文 0.6/字，其余 0.3/字符。"""
    if not text:
        return 0
    cjk_chars = sum(len(run) for run in _CJK_RUN.findall(text))
    other_chars = max(len(text) - cjk_chars, 0)
    return int(math.ceil(cjk_chars * 0.6 + other_chars * 0.3))
