"""记忆页面的只读接口。

页面放在 `pages/memory/`：AstrBot 会扫描插件下的 `views/` 与 `pages/` 目录，
把每个含 `index.html` 的子目录认成一个页面（标题默认取目录名，可用
`.astrbot-plugin/i18n/<locale>.json` 覆盖）。

前端通过宿主注入的 `window.AstrBotPluginPage` 桥调用接口，该桥传递**相对**路径
（如 `overview`），宿主自动补成 `/api/plug/astrbot_plugin_shittim/overview`。

本模块只提供读取，不包含任何写入操作：记忆数据归插件所有，页面仅负责展示。
"""

from __future__ import annotations

import json
from typing import Any

from astrbot.api import logger

try:  # AstrBot 4.27+ 的新版插件 Web API
    from astrbot.api.web import error_response, json_response, request

    _HAS_WEB_API = True
except ImportError:  # pragma: no cover - 旧版回退 quart
    from quart import jsonify, request  # type: ignore

    _HAS_WEB_API = False

# 分页上限：单个请求最多返回的条数，避免一次性加载整库数据
_MAX_LIMIT = 200
_DEFAULT_LIMIT = 50

# 记忆页面只认这两层；短期账本走另一个接口
_VALID_TIERS = ("long", "permanent")


def _ok(data: Any = None, message: str = "") -> Any:
    payload = {"status": "ok", "message": message, "data": data}
    return json_response(payload) if _HAS_WEB_API else jsonify(payload)


def _err(message: str, status_code: int = 400) -> Any:
    if _HAS_WEB_API:
        return error_response(message, status_code=status_code)
    resp = jsonify({"status": "error", "message": message, "data": {}})
    resp.status_code = status_code
    return resp


def _query_raw(name: str) -> str | None:
    """读一个查询参数，兼容新版 request.query 与旧版 quart request.args。

    取不到（旧版字段名不同、或压根没有请求上下文）时返回 None，
    让调用方走默认值，而不是把异常抛到页面上。
    """
    for accessor in ("query", "args"):
        try:
            holder = getattr(request, accessor)
        except Exception:
            continue
        if holder is None:
            continue
        try:
            value = holder.get(name)
        except Exception:
            continue
        if value is not None:
            return str(value)
    return None


def _query_int(name: str, default: int) -> int:
    try:
        return int(str(_query_raw(name)))
    except (TypeError, ValueError):
        return default


def _paging(limit: int, offset: int) -> tuple[int, int]:
    """收敛分页参数：limit 落在 [1, _MAX_LIMIT]，offset 不为负。"""
    return max(1, min(int(limit), _MAX_LIMIT)), max(0, int(offset))


def _short_term_item(row) -> dict:
    try:
        caption = str(row["image_caption"] or "")
    except (KeyError, IndexError, TypeError):
        caption = ""
    try:
        staged = bool(row["staged"])
    except (KeyError, IndexError, TypeError):
        staged = False
    return {
        "id": int(row["id"]),
        "chat_id": str(row["chat_id"] or ""),
        "chat_label": str(row["chat_label"] or ""),
        "chat_kind": str(row["chat_kind"] or ""),
        "sender_id": str(row["sender_id"] or ""),
        "sender_name": str(row["sender_name"] or ""),
        "role": str(row["role"] or "user"),
        "content": str(row["content"] or ""),
        # 图片转述文本：页面据此把「[图片]」替换为可读内容
        "image_caption": caption,
        "created_at": float(row["created_at"] or 0.0),
        "gated": bool(row["gated"]),
        "source": str(row["source"] or "live"),
        "staged": staged,
    }


def _summary_item(row) -> dict:
    try:
        covered_items = json.loads(str(row["covered_items"] or "[]"))
    except (TypeError, ValueError):
        covered_items = []
    return {
        "id": int(row["id"]),
        "content": str(row["content"] or ""),
        "created_at": float(row["created_at"] or 0.0),
        "covered_at": float(row["covered_at"] or 0.0),
        "covered_count": int(row["covered_count"] or 0),
        "covered_items": [int(i) for i in covered_items if str(i).lstrip("-").isdigit()],
    }


# 等级编号 → 页面文案。等级由名次推导（见 store.curation_level），不落库。
_LEVEL_LABELS = {0: "完整", 1: "精简", 2: "关键词"}


def _memory_item(memory) -> dict:
    return {
        "id": memory.id,
        "tier": memory.tier,
        "content": memory.content,
        "source_quote": memory.source_quote,
        "tags": memory.tags,
        "keywords": memory.keywords,
        "chat_label": memory.chat_label or "",
        "user_name": memory.user_name or "",
        # 位置榜：名次即记忆排序依据，等级由名次推导
        "rank": memory.rank,
        "rank_total": memory.rank_total,
        "level": memory.level,
        "level_label": _LEVEL_LABELS.get(memory.level, "完整"),
        "promoted_at": memory.promoted_at,
        "recall_count": memory.recall_count,
        "cross_chat_count": memory.cross_chat_count,
        "manual": bool(memory.manual),
        "created_at": memory.created_at,
        "last_recalled_at": memory.last_recalled_at,
    }


class MemoryPageAPI:
    """记忆页面的四个只读视图。"""

    def __init__(self, plugin) -> None:
        self._plugin = plugin

    async def overview(self) -> Any:
        """总览：各层条数、会话数、关注中的会话数、当前作息。"""
        short_term = self._plugin.short_term
        tiers = self._plugin.memory_store.count_by_tier()
        chats = short_term.list_known_chats(500)
        try:
            focused = len(self._plugin.focus.list_focused(500))
        except Exception:
            focused = 0
        try:
            sleep = self._plugin.sleep_snapshot()
        except Exception:
            # 仅用于展示：作息数据不可用时不应导致整个总览接口失败
            sleep = None
        return _ok(
            {
                "short_term": short_term.count(),
                "summaries": len(short_term.list_summaries(500)),
                "chats": len(chats),
                "long": tiers.get("long", 0),
                "permanent": tiers.get("permanent", 0),
                "focused": focused,
                "sleep": sleep,
            }
        )

    async def chats(self) -> Any:
        """会话列表：短期账本中出现过的会话。"""
        rows = self._plugin.short_term.list_known_chats(500)
        return _ok(
            [
                {
                    "chat_id": str(row["chat_id"] or ""),
                    "chat_label": str(row["chat_label"] or ""),
                    "chat_kind": str(row["chat_kind"] or ""),
                    "message_count": int(row["message_count"] or 0),
                    "last_at": float(row["last_at"] or 0.0),
                }
                for row in rows
            ]
        )

    async def short_term(self) -> Any:
        """短期账本分页；传入 chat_id 时仅返回该会话。"""
        chat_id = (_query_raw("chat_id") or "").strip() or None
        limit, offset = _paging(
            _query_int("limit", _DEFAULT_LIMIT), _query_int("offset", 0)
        )
        rows, total = self._plugin.short_term.page(chat_id, limit, offset)
        return _ok(
            {
                "total": total,
                "limit": limit,
                "offset": offset,
                "items": [_short_term_item(row) for row in rows],
            }
        )

    async def summaries(self) -> Any:
        """短期层中已压缩的段落：与时间线中的摘要块同源。

        摘要不进入消息分页（缺少 sender 等消息属性），因此单独提供一个视图，
        页面可将「原始消息 + 摘要」还原为上下文实际读取的顺序。
        """
        limit, _ = _paging(_query_int("limit", _DEFAULT_LIMIT), 0)
        rows = self._plugin.short_term.list_summaries(limit)
        return _ok({"items": [_summary_item(row) for row in rows]})

    async def memories(self) -> Any:
        """长期 / 永久记忆分页，按位置榜倒序（近期被想起的排在前面）。"""
        tier = (_query_raw("tier") or "long").strip()
        if tier not in _VALID_TIERS:
            return _err("tier 只能是 long 或 permanent")
        limit, offset = _paging(
            _query_int("limit", _DEFAULT_LIMIT), _query_int("offset", 0)
        )
        items, total = self._plugin.memory_store.list_page(tier, limit, offset)
        return _ok(
            {
                "tier": tier,
                "total": total,
                "limit": limit,
                "offset": offset,
                "items": [_memory_item(memory) for memory in items],
            }
        )


def register_all_routes(plugin) -> None:
    """把页面路由挂到宿主；宿主不支持插件 Web API 时静默跳过。"""
    register = getattr(plugin.context, "register_web_api", None)
    if not callable(register):
        logger.warning("[什亭之枢] 当前 AstrBot 不支持插件 Web API，记忆页面不可用")
        return

    api = MemoryPageAPI(plugin)
    routes = (
        ("overview", api.overview, "记忆总览"),
        ("chats", api.chats, "会话列表"),
        ("short_term", api.short_term, "短期账本"),
        ("summaries", api.summaries, "短期摘要"),
        ("memories", api.memories, "长期与永久记忆"),
    )
    for path, handler, description in routes:
        # 重复注册同一路由时宿主会替换，热重载不会累积
        register(
            f"/astrbot_plugin_shittim/{path}",
            handler,
            ["GET"],
            description,
        )
    logger.info("[什亭之枢] 记忆页面接口已注册（%d 个路由）", len(routes))
