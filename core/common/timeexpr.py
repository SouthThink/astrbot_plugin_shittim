"""时间词 → 时间区间。

账本里的每一行都带**绝对时间戳**，而人说话用的是「昨天」「上周三」「前几天」。
这个减法过去每次都由模型自己做，跨月、跨年时几乎必错——而模型算不出来的东西
本来就不该让它算，属于「插件递事实、人格写准则」的分工。

几条设计约束：

- **时间不能当关键词**。中文 2-gram 会把 `2026-11-10` 切成 `20`/`02`/`2-` 这样的
  碎片，而且提炼记忆时从不要求正文写日期，倒排索引里根本没有日期可匹配。
  时间是查询的一个**条件**（`WHERE`），不是打分项。
- **产出区间而不是日期串**。同一天在跨时区、跨零点时都有边界问题，区间是唯一
  不需要二次解释的形式。
- **宁可解析不出来，也不要解析错**。解析错会让笃定地翻错一天，比不翻更糟。
  因此只覆盖「可算」的说法，模糊说法一律标 `explicit=False`，由调用方决定要不要
  当成低置信度的线索。
- **日界就是自然日**。不引入 04:00 之类的偏移：凌晨两点说的「昨天」指的是刚刚过去的
  那一天（十小时前的傍晚还在里面），把日界往后推反而会把它指到前一天去。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta

# ── 数据结构 ──────────────────────────────────────────────


@dataclass(frozen=True)
class TimeHint:
    """一句话里解析出的一个时间区间（左闭右开）。

    `label` 是给模型看的说明（「昨天（11月10日）」），**必须随结果一起注入**：
    只给数据不给标注，她还是要自己把「昨天」换算成日期。
    `explicit=False` 表示原文是「前几天」这类宽区间，起点与终点都是估的。
    """

    start: float
    end: float
    label: str
    explicit: bool
    text: str

    def contains(self, ts: float) -> bool:
        return self.start <= float(ts) < self.end

    def days(self) -> int:
        return max(int(round((self.end - self.start) / 86400.0)), 1)


# ── 基础换算 ──────────────────────────────────────────────


def _day(ts: float) -> date:
    return datetime.fromtimestamp(float(ts)).date()


def _at(day: date, hour: int = 0) -> float:
    return datetime(day.year, day.month, day.day, hour, 0, 0).timestamp()


def day_bounds(ts: float) -> tuple[float, float]:
    """`ts` 所在自然日的 [起, 止)。"""
    today = _day(ts)
    return _at(today), _at(today + timedelta(days=1))


def day_key(ts: float) -> str:
    """自然日的稳定键（本地时区 `YYYY-MM-DD`），供 `memories.covered_date` 落库。

    按天查记忆需要一个确定的键：`promoted_at` 是入榜位置，未必落在那件事发生的
    那一天（当天摘要与合并结果用的是整理时刻）。
    """
    if not ts:
        return ""
    return _day(ts).isoformat()


def _md(day: date) -> str:
    return f"{day.month}月{day.day}日"


def _span_label(name: str, start: date, end: date) -> str:
    """生成「昨天（11月10日）」或「上周（11月3日到11月9日）」。"""
    if start == end:
        return f"{name}（{_md(start)}）"
    return f"{name}（{_md(start)}到{_md(end)}）"


def _hint(
    name: str, start: date, end_exclusive: date, *, explicit: bool, text: str
) -> TimeHint:
    """按「起日 + 止日（不含）」构造区间。"""
    if end_exclusive <= start:
        end_exclusive = start + timedelta(days=1)
    last = end_exclusive - timedelta(days=1)
    return TimeHint(
        start=_at(start),
        end=_at(end_exclusive),
        label=_span_label(name, start, last),
        explicit=explicit,
        text=text,
    )


def _weekday_index(char: str) -> int:
    table = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
    if char in table:
        return table[char]
    try:
        value = int(char)
    except (TypeError, ValueError):
        return -1
    return value - 1 if 1 <= value <= 7 else -1


def _month_start(day: date) -> date:
    return day.replace(day=1)


def _add_months(day: date, months: int) -> date:
    month = day.month - 1 + months
    year = day.year + month // 12
    return date(year, month % 12 + 1, 1)


# ── 解析规则 ──────────────────────────────────────────────
#
# 顺序即优先级：越具体的说法排得越前，命中区间会被标记为「已消费」，
# 后面的规则不会再在同一段文字上重复匹配（「昨天晚上」不会被当成「昨天」）。


def _r_yesterday_night(match: re.Match, today: date) -> TimeHint:
    yesterday = today - timedelta(days=1)
    return TimeHint(
        start=_at(yesterday, 17),
        end=_at(today, 5),
        label=f"{match.group(0)}（{_md(yesterday)}傍晚到{_md(today)}凌晨）",
        explicit=True,
        text=match.group(0),
    )


def _r_today_morning(match: re.Match, today: date) -> TimeHint:
    return TimeHint(
        start=_at(today, 5),
        end=_at(today, 12),
        label=f"{match.group(0)}（{_md(today)}上午）",
        explicit=True,
        text=match.group(0),
    )


def _make_part_of_day(word: str, offset: int):
    """「昨天下午」这类：把一天切成上午 / 下午 / 晚上三段。"""

    def resolve(match: re.Match, today: date) -> TimeHint:
        target = today - timedelta(days=offset)
        if "上午" in word or "早上" in word:
            start, end = _at(target, 5), _at(target, 12)
        elif "中午" in word:
            start, end = _at(target, 11), _at(target, 14)
        elif "下午" in word:
            start, end = _at(target, 12), _at(target, 18)
        else:  # 晚上
            start, end = _at(target, 18), _at(target + timedelta(days=1), 3)
        return TimeHint(
            start=start,
            end=end,
            label=f"{match.group(0)}（{_md(target)}）",
            explicit=True,
            text=match.group(0),
        )

    return resolve


def _make_day(word: str, offset: int):
    def resolve(match: re.Match, today: date) -> TimeHint:
        target = today - timedelta(days=offset)
        return _hint(
            word,
            target,
            target + timedelta(days=1),
            explicit=True,
            text=match.group(0),
        )

    return resolve


def _r_last_week(match: re.Match, today: date) -> TimeHint:
    this_monday = today - timedelta(days=today.weekday())
    start = this_monday - timedelta(days=7)
    return _hint(
        match.group(0),
        start,
        start + timedelta(days=7),
        explicit=True,
        text=match.group(0),
    )


def _r_this_week(match: re.Match, today: date) -> TimeHint:
    start = today - timedelta(days=today.weekday())
    return _hint(
        match.group(0), start, today + timedelta(days=1), explicit=True, text=match.group(0)
    )


def _r_last_weekend(match: re.Match, today: date) -> TimeHint:
    this_monday = today - timedelta(days=today.weekday())
    last_saturday = this_monday - timedelta(days=2)
    return _hint(
        match.group(0),
        last_saturday,
        last_saturday + timedelta(days=2),
        explicit=True,
        text=match.group(0),
    )


def _r_weekday(match: re.Match, today: date) -> TimeHint:
    """「周三」「上周三」「这周三」。

    没有前缀时取**最近一次**（不超过今天）：周三说的「周三」就是今天，
    周一说的「周三」指的是上周三，不该指到未来去。
    """
    prefix = (match.group(1) or "").strip()
    index = _weekday_index(match.group(2))
    if index < 0:
        target = today
    else:
        if prefix.startswith("上"):
            base = today - timedelta(days=today.weekday()) - timedelta(days=7)
        elif prefix:
            base = today - timedelta(days=today.weekday())
        else:
            base = today - timedelta(days=today.weekday())
        target = base + timedelta(days=index)
        if target > today:
            # 「这周三」在周一说出、或裸「周三」还没到：退回最近一次
            target -= timedelta(days=7)
    return _hint(
        match.group(0),
        target,
        target + timedelta(days=1),
        explicit=True,
        text=match.group(0),
    )


def _r_last_month(match: re.Match, today: date) -> TimeHint:
    start = _add_months(_month_start(today), -1)
    return _hint(
        match.group(0), start, _month_start(today), explicit=True, text=match.group(0)
    )


def _r_this_month(match: re.Match, today: date) -> TimeHint:
    return _hint(
        match.group(0),
        _month_start(today),
        today + timedelta(days=1),
        explicit=True,
        text=match.group(0),
    )


def _r_month_day(match: re.Match, today: date) -> TimeHint:
    """「11月3号」——今年最近的这一天；若还没到，退回去年。"""
    month = int(match.group(1))
    day = int(match.group(2))
    target: date | None = None
    for year in (today.year, today.year - 1):
        try:
            candidate = date(year, month, day)
        except ValueError:
            return _r_fuzzy(match, today, 365, name="某个日子", explicit=True)
        if candidate <= today:
            target = candidate
            break
    if target is None:
        target = today
    return _hint(
        match.group(0),
        target,
        target + timedelta(days=1),
        explicit=True,
        text=match.group(0),
    )


def _r_month(match: re.Match, today: date) -> TimeHint:
    month = int(match.group(1))
    if not 1 <= month <= 12:
        return _r_fuzzy(match, today, 365, name="某个日子", explicit=True)
    year = today.year if month <= today.month else today.year - 1
    start = date(year, month, 1)
    end = _add_months(start, 1)
    return _hint(match.group(0), start, end, explicit=True, text=match.group(0))


_CN_NUM = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _r_days_ago(match: re.Match, today: date) -> TimeHint:
    raw = match.group(1)
    count = _CN_NUM.get(raw)
    if count is None:
        try:
            count = int(raw)
        except (TypeError, ValueError):
            count = 2
    count = max(count, 1)
    target = today - timedelta(days=count)
    return _hint(
        match.group(0),
        target,
        target + timedelta(days=1),
        explicit=True,
        text=match.group(0),
    )


def _r_fuzzy(match: re.Match, today: date, days: int, *, name: str = "", explicit: bool = False) -> TimeHint:
    label_name = name or match.group(0)
    return _hint(
        label_name,
        today - timedelta(days=days),
        today + timedelta(days=1),
        explicit=explicit,
        text=match.group(0),
    )


# (正则, 解析器)。解析器签名固定为 (match, today) -> TimeHint
_RULES: list[tuple[re.Pattern, object]] = [
    (re.compile(r"昨天(?:晚上|夜里|夜间)|昨晚|昨夜"), _r_yesterday_night),
    (re.compile(r"今早|今天早上|今晨|今天上午"), _r_today_morning),
    (re.compile(r"昨天(?:早上|上午|中午|下午|晚上)"), _make_part_of_day("昨天", 1)),
    (re.compile(r"前天(?:早上|上午|中午|下午|晚上)"), _make_part_of_day("前天", 2)),
    (re.compile(r"大前天"), _make_day("大前天", 3)),
    (re.compile(r"前天"), _make_day("前天", 2)),
    (re.compile(r"昨天|昨日|昨儿"), _make_day("昨天", 1)),
    (re.compile(r"今天|今日|今儿"), _make_day("今天", 0)),
    (re.compile(r"(上上|上|这|本|这个|下)?个?(?:星期|周|礼拜)([一二三四五六日天1-7])"), _r_weekday),
    (re.compile(r"(?:上上|上)个?(?:星期|周|礼拜)(?![一二三四五六日天1-7])"), _r_last_week),
    (re.compile(r"(?:这|本)个?(?:星期|周|礼拜)(?![一二三四五六日天1-7])"), _r_this_week),
    (re.compile(r"(?:上个|上)周末|上星期末"), _r_last_weekend),
    (re.compile(r"(?:上个月|上月)"), _r_last_month),
    (re.compile(r"(?:这个月|本月|这月)"), _r_this_month),
    (re.compile(r"(\d{1,2})月(\d{1,2})[号日]?"), _r_month_day),
    (re.compile(r"(\d{1,2})月份?"), _r_month),
    (re.compile(r"(\d{1,2}|[一两二三四五六七八九十])天前"), _r_days_ago),
    (re.compile(r"前几天|前两天|前些天|这几天"), lambda m, t: _r_fuzzy(m, t, 7)),
    (re.compile(r"前阵子|前段时间|那阵子|那段时间|前些日子"), lambda m, t: _r_fuzzy(m, t, 30)),
    (re.compile(r"上次|上回"), lambda m, t: _r_fuzzy(m, t, 60)),
    (re.compile(r"最近|近来|最近几天"), lambda m, t: _r_fuzzy(m, t, 7)),
]


def parse_time_hints(text: str, now: float | None = None) -> list[TimeHint]:
    """解析文本里所有可算的时间说法，按出现位置返回。

    同一段文字只会命中一次（先匹配到的规则消费掉它），因此
    「昨天晚上」不会被重复解释成「昨天」。
    """
    source = str(text or "")
    if not source:
        return []
    moment = time.time() if now is None else float(now)
    today = _day(moment)

    consumed: list[tuple[int, int]] = []
    found: list[tuple[int, TimeHint]] = []
    for pattern, resolver in _RULES:
        for match in pattern.finditer(source):
            start, end = match.start(), match.end()
            if any(start < c_end and c_start < end for c_start, c_end in consumed):
                continue
            try:
                hint = resolver(match, today)  # type: ignore[operator]
            except Exception:
                continue
            if hint.end <= hint.start:
                continue
            consumed.append((start, end))
            found.append((start, hint))
    found.sort(key=lambda item: item[0])
    return [hint for _, hint in found]


def best_time_hint(
    text: str, now: float | None = None, *, hints: list[TimeHint] | None = None
) -> TimeHint | None:
    """一句话里有多个时间说法时，挑**最具体**的那一个。

    「前天晚上你说的那个」只有一个；但「昨天和前两天聊的都算」有两个，
    把它们并成大区间等于同时翻错两天，不如只认最明确的那条。
    排序：确切可算 > 模糊；区间窄 > 宽；原文长 > 短。
    """
    items = hints if hints is not None else parse_time_hints(text, now)
    if not items:
        return None
    return min(
        items,
        key=lambda hint: (
            0 if hint.explicit else 1,
            hint.days(),
            -len(hint.text),
        ),
    )


def strip_time_phrases(text: str, hints: list[TimeHint] | None = None) -> str:
    """把时间词从查询文本里去掉，剩下的才是关键词。

    时间已经由区间条件表达；留在关键词里只会稀释 `overlap_score`
    （分母是记忆自己的关键词数），并把「今天」这种高频词喂进倒排索引。
    """
    source = str(text or "")
    items = hints if hints is not None else parse_time_hints(source)
    for hint in sorted(items, key=lambda h: -len(h.text)):
        if hint.text:
            source = source.replace(hint.text, " ")
    return " ".join(source.split())
