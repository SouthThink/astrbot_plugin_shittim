"""记忆维护管线：衰减 → 遗忘 → 晋升 → 合并。

按设计，这条管线跟随她的「休息」时机执行：睡前一并整理，醒来就是清理过的记忆。
管线在对话路径之外运行，失败不应影响任何消息处理。
"""

from __future__ import annotations

import time

try:
    from astrbot.api import logger
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

from .memory_store import TIER_LONG, Memory, MemoryStore

# 合并后强度的折算系数：取各成员强度之和的一半，而不是最大值，
# 避免「合并」变成刷强度的后门。
_MERGE_STRENGTH_RATIO = 0.5
_MAX_STRENGTH = 20.0


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

    async def run(self) -> dict:
        """执行一次完整维护，返回统计信息（供日志与排障使用）。"""
        stats = {
            "decayed": 0,
            "forgotten": 0,
            "promoted": 0,
            "merge_clusters": 0,
            "merged_removed": 0,
            "merged_created": 0,
            "pruned_messages": 0,
        }
        if not self._config.maintenance_enabled:
            return stats

        started = time.time()
        try:
            stats["decayed"] = self._decay()
            stats["forgotten"] = self._forget()
            stats["promoted"] = self._promote()
        except Exception as error:
            logger.error("[什亭之枢] 记忆衰减/遗忘/晋升失败: %s", error)

        try:
            merge_stats = await self._merge()
            stats.update(merge_stats)
        except Exception as error:
            logger.error("[什亭之枢] 记忆合并失败: %s", error)

        if self._short_term is not None:
            try:
                stats["pruned_messages"] = self._short_term.prune()
            except Exception as error:
                logger.error("[什亭之枢] 短期账本淘汰失败: %s", error)

        logger.info(
            "[什亭之枢] 记忆维护完成 用时%.0fms 衰减%d 遗忘%d 晋升%d "
            "合并簇%d(减少%d条/新增%d条) 淘汰消息%d条",
            (time.time() - started) * 1000,
            stats["decayed"],
            stats["forgotten"],
            stats["promoted"],
            stats["merge_clusters"],
            stats["merged_removed"],
            stats["merged_created"],
            stats["pruned_messages"],
        )
        return stats

    # ── 衰减 ──────────────────────────────────────────────

    def _decay(self) -> int:
        """对长期记忆做半衰期衰减。

        衰减区间 = now - max(上次衰减时间, 上次活跃时间)。
        这样「被召回过」的记忆会把衰减时钟重置到召回时刻，而不是被重复计费。
        """
        now = time.time()
        half_life = max(self._config.decay_half_life_days, 1.0)
        changed = 0
        for memory in self._store.list_long_memories():
            base = max(memory.last_decay_at, memory.activity_at())
            elapsed_days = max(now - base, 0.0) / 86400.0
            if elapsed_days <= 0:
                continue
            factor = 0.5 ** (elapsed_days / half_life)
            if factor >= 1.0:
                continue
            self._store.update(
                memory.id,
                strength=memory.strength * factor,
                last_decay_at=now,
            )
            changed += 1
        return changed

    # ── 遗忘 ──────────────────────────────────────────────

    def _forget(self) -> int:
        """强度跌破阈值的长期记忆直接遗忘。永久记忆不参与。"""
        forgotten = 0
        for memory in self._store.list_long_memories():
            if memory.strength < self._config.forget_threshold:
                self._store.delete(memory.id)
                forgotten += 1
        return forgotten

    # ── 晋升 ──────────────────────────────────────────────

    def _promote(self) -> int:
        """晋升为永久记忆。

        刻意不使用「召回次数」单指标：召回多可能只是话题热。
        要求同时满足 强度 + 召回次数 + 跨会话出现，才算真正重要的记忆。
        """
        promoted = 0
        for memory in self._store.list_long_memories():
            if (
                memory.recall_count >= self._config.promote_min_recall
                and memory.strength >= self._config.promote_min_strength
                and memory.cross_chat_count >= self._config.promote_min_cross_chat
            ):
                self._store.promote(memory.id)
                promoted += 1
        return promoted

    # ── 合并 ──────────────────────────────────────────────

    async def _merge(self) -> dict:
        """规则聚类筛出候选，再交给 LLM 合并。

        合并前先按「强度够弱 + 足够陈旧」过滤，把 LLM 的接触面压到最小，
        既省成本，也把幻觉可能影响的范围限制在少数弱记忆上。
        """
        stats = {"merge_clusters": 0, "merged_removed": 0, "merged_created": 0}
        if not getattr(self._extractor, "available", False):
            return stats

        now = time.time()
        candidates = [
            memory
            for memory in self._store.list_long_memories()
            if memory.strength <= self._config.merge_max_strength
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

    def _cluster(self, memories: list[Memory]) -> list[list[Memory]]:
        """按「标签 ∪ 关键词」的 Jaccard 相似度做贪心聚类。

        聚类中心用并集（会随成员增长而变宽），这是刻意接受的设计：
        候选已经被「弱 + 陈旧」限制住，宽中心能让同主题的碎记忆聚到一起。
        """
        clusters: list[tuple[set[str], list[Memory]]] = []
        for memory in sorted(memories, key=lambda m: -m.strength):
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
        strength = min(
            sum(memory.strength for memory in cluster) * _MERGE_STRENGTH_RATIO,
            _MAX_STRENGTH,
        )
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
                    strength=strength / max(len(merged_items), 1),
                    tier=TIER_LONG,
                    merged_from=merged_from,
                )
                created += 1
            for memory in cluster:
                self._store.delete(memory.id)
        logger.info(
            "[什亭之枢] 合并 %d 条弱记忆为 %d 条 (strength=%.2f)",
            len(cluster),
            created,
            strength,
        )
        return created
