"""记忆页面的只读接口。

页面放在 `pages/memory/`：AstrBot 会扫描插件下的 `views/` 与 `pages/` 目录，
把每个含 `index.html` 的子目录认成一个页面（标题默认取目录名，可用
`.astrbot-plugin/i18n/<locale>.json` 覆盖）。

前端拿到的是宿主注入的 `window.AstrBotPluginPage` 桥，它传**相对**路径
（如 `overview`），宿主自动补成 `/api/plug/astrbot_plugin_shittim/overview`。

这里只有读，没有任何写：记忆是她的，页面只负责让主人看得见。
"""

from __future__ import annotations

from typing import Any

try:
    from astrbot.api import logger
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

try:  # AstrBot 4.27+ 的新版插件 Web API
    from astrbot.api.web import error_response, json_response, request

    _HAS_WEB_API = True
except ImportError:  # pragma: no cover - 旧版回退 quart
    from quart import jsonify, request  # type: ignore

    _HAS_WEB_API = False

# 分页上限：页面一次最多看这么多条，避免把整库拉进浏览器
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
    return {
        "id": int(row["id"]),
        "chat_id": str(row["chat_id"] or ""),
        "chat_label": str(row["chat_label"] or ""),
        "chat_kind": str(row["chat_kind"] or ""),
        "sender_id": str(row["sender_id"] or ""),
        "sender_name": str(row["sender_name"] or ""),
        "role": str(row["role"] or "user"),
        "content": str(row["content"] or ""),
        # 图片转述出来的文字：页面据此把「[图片]」补成看得见的内容
        "image_caption": caption,
        "created_at": float(row["created_at"] or 0.0),
        "gated": bool(row["gated"]),
        "source": str(row["source"] or "live"),
        "extracted": bool(row["extracted"]),
    }


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
        "strength": round(float(memory.strength), 2),
        "recall_count": memory.recall_count,
        "useful_count": memory.useful_count,
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
        """总览：各层条数、会话数、她在关注的会话数。"""
        short_term = self._plugin.short_term
        tiers = self._plugin.memory_store.count_by_tier()
        chats = short_term.list_known_chats(500)
        try:
            focused = len(self._plugin.focus.list_focused(500))
        except Exception:
            focused = 0
        return _ok(
            {
                "short_term": short_term.count(),
                "chats": len(chats),
                "long": tiers.get("long", 0),
                "permanent": tiers.get("permanent", 0),
                "focused": focused,
            }
        )

    async def chats(self) -> Any:
        """会话列表：她的短期账本里出现过哪些会话。"""
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
        """短期账本分页；带 chat_id 就只看那个会话。"""
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

    async def memories(self) -> Any:
        """长期 / 永久记忆分页，按强度倒序。"""
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
