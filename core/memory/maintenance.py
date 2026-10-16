"""记忆维护管线：压缩短期 → 降级 → 遗忘 → 晋升 → 合并。

按设计，这条管线跟随她的「深睡」时机执行：睡前一并整理，醒来就是清理过的记忆。
管线在对话路径之外运行，失败不应影响任何消息处理。

整理产生的记忆**一律上浮到榜前**（`promoted_at = now`）——包括压缩时顺带
产出的、以及合并重写出来的。榜的语义是「最后一次被想起」，而整理就是
她「想起了一遍」；不需要给合并开特例。
"""

from __future__ import annotations

import time

try:
    from astrbot.api import logger
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

from .store import (
    LEVEL_FULL,
    LEVEL_BRIEF,
    LEVEL_KEYWORDS,
    TIER_LONG,
    Memory,
    MemoryStore,
    is_tail,
)

# 降级节拍护栏（§9.6）：整理本身已是每晚一次，这里再挡一道，防止
# 「23:59 整理过、00:01 又整理」这种跨零点的相邻整理把同一条连降两级。
_DEMOTE_INTERVAL_HOURS = 24.0
# 「精简」级的正文上限
_BRIEF_MAX_CHARS = 30
# 「只剩关键词」时最多并列几个词：太多会变成一串没有主语的碎词
# （关键词是 char bigram，本身就偏碎，这里更要收敛数量）
_KEYWORD_BODY_MAX = 5

# 合并只动榜的后半段。这是退役的 merge_max_strength（「只合并弱记忆」）
# 在名次制下的等价物：榜前的记忆是「她刚想起过的」，不该被合并掉细节。
_MERGE_SAFE_RATIO = 0.5

# 摘要长度：中间层的只要够当上下文背景；当天的要能承载一整天的轮廓，
# 所以放宽一些——它是那一整天唯一留下来的东西。
_SUMMARY_CHARS = 150
_DAY_SUMMARY_CHARS = 400


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


class MemoryMaintainer:
    """记忆的衰减、遗忘、晋升与合并。"""

    def __init__(self, store: MemoryStore, extractor, config, short_term=None):
        self._store = store
        self._extractor = extractor
        self._config = config
        self._short_term = short_term

    def update_config(self, config) -> None:
        self._config = config

    async def run(self, *, curate_budget: int = 20) -> dict:
        """执行一次完整维护，返回统计信息（供日志与排障使用）。

        `curate_budget` 是单轮降级最多重写几条正文（控模型成本），
        通常由调用方从作息配置里取来。

        顺序是固定的：**降级 → 遗忘 → 晋升 → 合并**。
        遗忘排在晋升之前，是为了让「已沉到底」的记忆有机会在晋升之前被清掉，
        而不是靠召回次数洗白。

        注意这里**不含短期收口与清空**：收口要调模型、清空要动账本，
        由 main 在确认收口成功之后按顺序单独做（见 main._run_curation）。
        """
        stats = {
            "demoted": 0,
            "compressed": 0,
            "forgotten": 0,
            "promoted": 0,
            "merge_clusters": 0,
            "merged_removed": 0,
            "merged_created": 0,
        }
        if not self._config.maintenance_enabled:
            return stats

        started = time.time()
        try:
            demote_stats = await self._demote(curate_budget)
            stats["demoted"] = demote_stats["demoted"]
            stats["compressed"] = demote_stats["compressed"]
        except Exception as error:
            logger.error("[什亭之枢] 记忆降级失败: %s", error)

        try:
            stats["forgotten"] = self._forget()
            stats["promoted"] = self._promote()
        except Exception as error:
            logger.error("[什亭之枢] 记忆遗忘/晋升失败: %s", error)

        try:
            merge_stats = await self._merge()
            stats.update(merge_stats)
        except Exception as error:
            logger.error("[什亭之枢] 记忆合并失败: %s", error)

        logger.info(
            "[什亭之枢] 记忆维护完成 用时%.0fms 降级%d(重写%d) 遗忘%d 晋升%d "
            "合并簇%d(减少%d条/新增%d条)",
            (time.time() - started) * 1000,
            stats["demoted"],
            stats["compressed"],
            stats["forgotten"],
            stats["promoted"],
            stats["merge_clusters"],
            stats["merged_removed"],
            stats["merged_created"],
        )
        return stats

    # ── 短期压缩 ──────────────────────────────────────────
    #
    # 压缩是「短期层 → 长期层」**唯一**的通道。每一次压缩调用都同时产出
    # 「1 条短期摘要」与「N 条长期记忆」，两者是从同一段原文来的**兄弟**，
    # 不是「先摘摘要、再从摘要提炼」的父子关系——后者等于压两遍，细节会掉光。

    async def condense_short_term(self) -> dict:
        """短期层超阈值时，把最旧的一批压成一条摘要。

        每次**只压一轮**：压完 300 → 201 就收工，代价可控。它的职责是
        「别让上下文太长」，不是清空短期层——清空是 `collapse_short_term` 的事。
        """
        stats = self._empty_condense_stats()
        if self._short_term is None:
            return stats

        stats["entries"] = self._short_term.live_entry_count()
        if stats["entries"] <= max(int(self._config.condense_threshold), 1):
            return stats

        picked = self._short_term.oldest_entries(
            max(int(self._config.condense_batch), 1)
        )
        stats["picked"] = len(picked)
        if not picked:
            return stats
        if await self._condense_batch(picked, stats):
            logger.info(
                "[什亭之枢] 短期压缩完成 条目%d→%d 折叠%d条 新增记忆%d条",
                stats["entries"],
                stats["entries"] - stats["folded"] + 1,
                stats["folded"],
                stats["memories_created"],
            )
        else:
            stats["failed"] = True
        return stats

    async def collapse_short_term(self, *, max_rounds: int) -> dict:
        """把短期层整体压成**一条「当天摘要」**，并把它提升进长期层。

        树状压缩：每一轮把当前条目按 `condense_batch` 分组，每组各压成一条，
        条目数按 100:1 收缩。所以 856 条只要**两层**——细节损失远小于「每次压
        100 条、反复九轮」的链式做法，后者压九层就不是「可以详细些」了。

        最后剩下的那一条就是当天摘要：它作为**一条长期记忆**写入，落在榜最前
        （所以最详细）；日后随名次下沉被逐级压模糊，细节自然递减。

        `failed=True` 时调用方**绝对不能清空短期层**，否则那一段永久消失。
        中途失败是安全的：已压好的块自身自洽，剩下的原样留着，下次继续。
        """
        stats = self._empty_condense_stats()
        if self._short_term is None:
            return stats
        if not getattr(self._extractor, "available", False):
            stats["failed"] = True
            logger.warning("[什亭之枢] 短期收口跳过：未配置压缩模型")
            return stats

        batch = max(int(self._config.condense_batch), 1)
        stats["entries"] = self._short_term.live_entry_count()
        for _ in range(max(int(max_rounds), 1)):
            entries = self._short_term.live_entries()
            if not entries:
                break
            if len(entries) == 1 and entries[0].get("kind") == "summary":
                break  # 已经只剩当天摘要了
            chunks = [entries[i : i + batch] for i in range(0, len(entries), batch)]
            # 只剩一块时就是最后一压，用更宽的长度上限：当天摘要要详细些
            final = len(chunks) == 1
            for chunk in chunks:
                if not await self._condense_batch(chunk, stats, final=final):
                    stats["failed"] = True
                    logger.warning(
                        "[什亭之枢] 短期收口中断于第%d轮，剩余条目保持原样",
                        stats["rounds"] + 1,
                    )
                    return stats
            stats["rounds"] += 1

        stats["day_summary"] = self._promote_day_summary()
        if not stats["day_summary"]:
            logger.info("[什亭之枢] 短期收口：短期层本来就是空的")
        return stats

    def _promote_day_summary(self) -> bool:
        """把短期层剩下的最后一条摘要提升为长期记忆（当天摘要）。

        提升后它就不再属于短期层：下一轮注入里它是「你记得的事」的第一条，
        而不是一段被压扁的经历。它**不进 `condense` 的下一层**——细节靠名次
        自然递减：排在榜首时是完整正文，日后沉下去才被逐级压模糊。
        """
        entries = self._short_term.live_entries() if self._short_term else []
        summaries = [entry for entry in entries if entry.get("kind") == "summary"]
        if len(summaries) != 1:
            return False
        row = summaries[0]["row"]
        content = str(row["content"] or "").strip()
        if not content:
            return False
        # 不传 created_at：promoted_at 取当下，当天摘要于是**确定性地**排在榜首
        # （同刻写入的靠 id 兜底，它的 id 最大）。这是「排最前＝最详细」的落点。
        self._store.add(content=content, tags=["当天摘要"], tier=TIER_LONG)
        self._short_term.retire_summary(int(row["id"]))
        logger.info("[什亭之枢] 当天摘要已写入长期记忆：%s", content[:40])
        return True

    async def _condense_batch(
        self, entries: list[dict], stats: dict, *, final: bool = False
    ) -> bool:
        """把一批条目压成一条摘要 + 若干长期记忆，返回是否成功。

        失败时**不改任何标记**：原始条目原样留着，等下一次。
        """
        if not entries:
            return False
        records = self._short_term.condense_records(entries)
        result = await self._extractor.condense(
            records, max_summary_chars=_DAY_SUMMARY_CHARS if final else _SUMMARY_CHARS
        )
        if result is None:
            return False
        summary, memories = result
        context = self._short_term.condense_context(entries)
        created, reinforced = self._write_condensed_memories(memories, context)
        stats["memories_created"] += created
        stats["memories_reinforced"] += reinforced
        if not self._short_term.apply_condensation(
            summary=summary, entries=entries, covered_count=len(entries)
        ):
            return False
        stats["folded"] += len(entries)
        stats["summaries"] += sum(1 for e in entries if e.get("kind") == "summary")
        return True

    @staticmethod
    def _empty_condense_stats() -> dict:
        return {
            "entries": 0,
            "picked": 0,
            "rounds": 0,
            "folded": 0,
            "summaries": 0,
            "memories_created": 0,
            "memories_reinforced": 0,
            "day_summary": False,
            "failed": False,
        }

    def _write_condensed_memories(
        self, memories: list[dict], context: dict
    ) -> tuple[int, int]:
        """把压缩时顺带产出的长期记忆写进去，返回 (新增, 强化)。"""
        created = 0
        reinforced = 0
        for item in memories:
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            if self._store.is_suppressed(content):
                logger.info("[什亭之枢] 丢弃被要求遗忘的内容: %s", content[:30])
                continue
            name = str(item.get("user_name") or "")
            _, is_new = self._store.reinforce_or_add(
                content=content,
                source_quote=str(item.get("source_quote") or ""),
                tags=item.get("tags") or [],
                keywords=item.get("keywords") or [],
                chat_id=context.get("chat_id"),
                chat_label=context.get("chat_label"),
                user_id=context["name_to_id"].get(name),
                user_name=name or None,
            )
            if is_new:
                created += 1
            else:
                reinforced += 1
        return created, reinforced

    # ── 降级（位置榜尾部变模糊）─────────────────────────────

    async def _demote(self, budget: int) -> dict:
        """把**榜尾**的记忆逐级变模糊（§3.2）。

        判据不再是积分，而是名次：榜前 60% 完整、到 85% 精简、之后只剩关键词
        （分档在 `store.curation_level`）。被召回会把自己上浮回榜前，所以常用
        的记忆根本不会被碰到——「防痴呆」从「加分」变成了「浮上去」。

        `tier='permanent'` 不在榜上，自然不参与。
        """
        stats = {"demoted": 0, "compressed": 0, "calls": 0, "budget": max(int(budget), 0)}
        now = time.time()
        interval = _DEMOTE_INTERVAL_HOURS * 3600
        for memory in self._store.board():
            if memory.forget_requested:
                # 已被要求遗忘：等 _forget 干净删掉，没必要再改写一次正文
                continue
            level = memory.level
            if level == LEVEL_FULL:
                continue
            if now - memory.last_curated_at < interval:
                continue
            fields = {
                "curation_count": memory.curation_count + 1,
                "last_curated_at": now,
            }
            # 每次都问一次正文形态，不只是「等级刚变化」时问：模型重写有单轮
            # 预算，预算用完的那批当晚只记次数、不改正文；若只在等级跃迁时改写，
            # 它们就永远等不到第二次机会（等级已经停在那一档了）。
            # `_fade_body` 内部自带幂等判断，不会因此多花模型调用。
            body = await self._fade_body(memory, level, stats)
            if body:
                fields["content"] = body
                stats["compressed"] += 1
            self._store.update(memory.id, **fields)
            stats["demoted"] += 1
        return stats

    async def _fade_body(self, memory: Memory, level: int, stats: dict) -> str:
        """按等级改写正文；返回空串表示这次不改。

        `source_quote` 永远不动：正文可以越来越模糊，但「她当时据以记住的
        那句话」必须一直指向原文，否则将来无从回溯这条记忆是怎么来的。
        """
        if level == LEVEL_KEYWORDS:
            # 「只剩关键词」这一档不依赖模型：直接用检索关键词拼，确定性、
            # 零成本，也绝不可能引入原文里没有的信息。
            keywords = [k for k in memory.keywords if k][:_KEYWORD_BODY_MAX]
            if not keywords:
                return ""
            body = " / ".join(keywords)
            return body if body != memory.content else ""

        if level != LEVEL_BRIEF:
            return ""

        if len(memory.content) <= _BRIEF_MAX_CHARS:
            # 已经是精简形态（或本来就短）：没必要每晚再花一次模型调用
            return ""

        if stats["calls"] >= stats["budget"]:
            return ""
        stats["calls"] += 1
        body = await self._extractor.compress(memory.content, _BRIEF_MAX_CHARS)
        if not body or len(body) >= len(memory.content):
            return ""
        return body

    # ── 遗忘 ──────────────────────────────────────────────

    def _forget(self) -> int:
        """确定性阶梯遗忘（§10.1）。永久记忆不在榜上，不参与。

        三条同时满足才删，不用概率——概率无法解释「这条为什么没了」：

        - 名次落在**榜尾**（`store.is_tail`）
        - 已经降级足够多次
        - 连续多日没被召回

        「被明确要求忘记」另走一条：主人开口了，下一轮整理就删掉，不等陈旧度。
        """
        now = time.time()
        stale_seconds = max(int(self._config.forget_stale_days), 0) * 86400
        min_curations = max(int(self._config.forget_min_curations), 0)
        forgotten = 0
        for memory in self._store.board():
            if memory.forget_requested:
                reason = "被要求遗忘"
            elif (
                is_tail(memory.rank, memory.rank_total)
                and memory.curation_count >= min_curations
                and now - memory.activity_at() >= stale_seconds
            ):
                reason = "榜尾久未被用到的陈旧记忆"
            else:
                continue
            self._store.delete(memory.id)
            forgotten += 1
            logger.info(
                "[什亭之枢] 遗忘记忆 #%d（%s，名次%d/%d，降级%d次）: %s",
                memory.id,
                reason,
                memory.rank,
                memory.rank_total,
                memory.curation_count,
                memory.content[:30],
            )
        return forgotten

    # ── 晋升 ──────────────────────────────────────────────

    def _promote(self) -> int:
        """晋升为永久记忆。

        刻意不使用「召回次数」单指标：召回多可能只是话题热。要求同时满足
        召回次数 + 跨会话出现，才算真正重要的记忆。
        """
        promoted = 0
        for memory in self._store.board():
            if memory.forget_requested:
                # 已经被要求忘记的记忆不该再被晋升回永久层
                continue
            if (
                memory.recall_count >= self._config.promote_min_recall
                and memory.cross_chat_count >= self._config.promote_min_cross_chat
            ):
                self._store.promote(memory.id)
                promoted += 1
        return promoted

    # ── 合并 ──────────────────────────────────────────────

    async def _merge(self) -> dict:
        """规则聚类筛出候选，再交给 LLM 合并。

        只动榜的**后半段**（`_MERGE_SAFE_RATIO`）：榜前那些是她刚想起过的，
        合并掉细节得不偿失。剩余的旧记忆里再按相似度聚类，把 LLM 的接触面
        压到最小，既省成本，也把幻觉可能影响的范围限制在少数旧记忆上。
        """
        stats = {"merge_clusters": 0, "merged_removed": 0, "merged_created": 0}
        if not getattr(self._extractor, "available", False):
            return stats

        now = time.time()
        candidates = [
            memory
            for memory in self._store.board()
            if not memory.forget_requested
            and self._is_mergeable(memory)
            and (now - memory.created_at) / 86400.0 >= self._config.merge_min_age_days
        ]
        if len(candidates) < self._config.merge_min_cluster:
            return stats

        for cluster in self._cluster(candidates):
            merged_items = await self._extractor.merge(cluster)
            if not merged_items:
                # 合并失败或没有真的减少条目：保持原样，绝不提交半成品
                continue
            applied = self._apply_merge(cluster, merged_items)
            if applied:
                stats["merge_clusters"] += 1
                stats["merged_removed"] += len(cluster)
                stats["merged_created"] += applied
        return stats

    @staticmethod
    def _is_mergeable(memory: Memory) -> bool:
        """名次是否已落到「可以合并」的后半段。"""
        if memory.rank <= 0 or memory.rank_total <= 0:
            return False
        return memory.rank / memory.rank_total > _MERGE_SAFE_RATIO

    def _cluster(self, memories: list[Memory]) -> list[list[Memory]]:
        """按「标签 ∪ 关键词」的 Jaccard 相似度做贪心聚类。

        聚类中心用并集（会随成员增长而变宽），这是刻意接受的设计：
        候选已被限制在榜后半段的旧记忆里，宽中心能让同主题的碎记忆聚到一起。
        从**榜最前**的那条开始做中心，让相对更重要的记忆当骨架。
        """
        clusters: list[tuple[set[str], list[Memory]]] = []
        for memory in sorted(memories, key=lambda m: m.rank or 0):
            terms = set(memory.tags) | set(memory.keywords)
            best_index = -1
            best_similarity = 0.0
            for index, (centroid, _) in enumerate(clusters):
                similarity = _jaccard(terms, centroid)
                if similarity >= self._config.merge_similarity and similarity > best_similarity:
                    best_index, best_similarity = index, similarity
            if best_index >= 0:
                centroid, members = clusters[best_index]
                centroid.update(terms)
                members.append(memory)
            else:
                clusters.append((set(terms), [memory]))
        return [
            members
            for _, members in clusters
            if len(members) >= self._config.merge_min_cluster
        ]

    def _apply_merge(self, cluster: list[Memory], merged_items: list[dict]) -> int:
        """用合并结果替换掉原来的记忆，返回新建条数。"""
        merged_from = [memory.id for memory in cluster]
        tags = sorted({tag for memory in cluster for tag in memory.tags})
        keywords = sorted({kw for memory in cluster for kw in memory.keywords})
        user_ids = {memory.user_id or "" for memory in cluster}
        chat_ids = {memory.chat_id or "" for memory in cluster}
        user_id = user_ids.pop() if len(user_ids) == 1 else None
        chat_id = chat_ids.pop() if len(chat_ids) == 1 else None
        user_name = cluster[0].user_name if user_id else None

        created = 0
        with self._store.lock:  # 合并的多步写入必须原子，避免中途可见半成品
            for item in merged_items:
                self._store.add(
                    content=item["content"],
                    source_quote=item.get("source_quote", ""),
                    tags=sorted(set(tags) | set(item.get("tags") or [])),
                    keywords=sorted(set(keywords) | set(item.get("keywords") or [])),
                    chat_id=chat_id,
                    chat_label=cluster[0].chat_label if chat_id else None,
                    user_id=user_id,
                    user_name=user_name,
                    tier=TIER_LONG,
                    merged_from=merged_from,
                )
                created += 1
            for memory in cluster:
                self._store.delete(memory.id)
        logger.info(
            "[什亭之枢] 合并 %d 条旧记忆为 %d 条", len(cluster), created
        )
        return created
