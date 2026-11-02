"""记忆页面的后端接口。

页面放在 `pages/memory/`：AstrBot 会扫描插件下的 `views/` 与 `pages/` 目录，
把每个含 `index.html` 的子目录认成一个页面（标题默认取目录名，可用
`.astrbot-plugin/i18n/<locale>.json` 覆盖）。

前端通过宿主注入的 `window.AstrBotPluginPage` 桥调用接口，该桥传递**相对**路径
（如 `overview`），宿主自动补成 `/api/plug/astrbot_plugin_shittim/overview`。

读接口提供记忆、短期账本、用户画像、巡检运行时状态与主脑上下文快照；写接口
只开放长期 / 永久记忆的**手工编辑与删除**——短期账本是原始记录，改动它会让
上下文与判定失去同一份事实来源。
"""

from __future__ import annotations

import json
import time
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

# 手工编辑的单条记忆长度上限：长期记忆是短句，超长输入多半是误粘贴
_MAX_MEMORY_CHARS = 4000

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


async def _json_body() -> dict:
    """读取请求体 JSON，兼容新版 astrbot.api.web 与旧版 quart。

    非 JSON、空体或读取失败一律返回空字典，让调用方按「缺参数」处理，
    而不是把异常抛成 500。
    """
    try:
        if _HAS_WEB_API:
            data = await request.json(default=None)
        else:
            data = await request.json
    except Exception:
        data = None
    return data if isinstance(data, dict) else {}


def _body_int(body: dict, name: str) -> int | None:
    """从请求体取一个正整数 id；非法或缺省返回 None。"""
    try:
        value = int(str(body.get(name)).strip())
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _short_term_item(row) -> dict:
    try:
        caption = str(row["image_caption"] or "")
    except (KeyError, IndexError, TypeError):
        caption = ""
    try:
        forward_content = str(row["forward_content"] or "")
    except (KeyError, IndexError, TypeError):
        forward_content = ""
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
        # 合并转发的聊天记录：页面据此展开「[转发消息]」
        "forward_content": forward_content,
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


def _profile_item(profile) -> dict:
    """用户画像 → 页面数据。名字列表来自 user_ledger，画像只提供印象。"""
    return {
        "platform": profile.platform,
        "user_id": profile.user_id,
        "display_name": profile.display_name,
        # 同一个人在私聊 / 各群里用过的名字
        "aliases": profile.aliases,
        "summary": profile.summary,
        "keywords": profile.keywords,
        "message_count": profile.message_count,
        "first_seen": profile.first_seen,
        "last_seen": profile.last_seen,
        "profiled_at": profile.profiled_at,
        "created_at": profile.created_at,
        "updated_at": profile.updated_at,
    }


class MemoryPageAPI:
    """记忆页面的视图与记忆手工维护接口。"""

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
                "profiles": self._plugin.profiles.count(),
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

    async def profiles(self) -> Any:
        """用户画像分页，按最近活跃倒序。

        画像以「平台 + QQ 号」为身份：同一行里同时给出他在不同会话用过的
        名字，页面据此确认「这是同一个人」。
        """
        limit, offset = _paging(
            _query_int("limit", _DEFAULT_LIMIT), _query_int("offset", 0)
        )
        items, total = self._plugin.profiles.list_page(limit, offset)
        return _ok(
            {
                "total": total,
                "limit": limit,
                "offset": offset,
                "items": [_profile_item(profile) for profile in items],
            }
        )

    async def runtime(self) -> Any:
        """巡检参数、当前窗口 / 判定 / 冷却，以及会话焦点状态。

        参数与运行时状态放在同一个视图里：只看配置无法判断「为什么这条消息
        没被放行」，两者对照才能看清是窗口还没到点、正在判定，还是判为不回复
        后的冷却期。焦点列表包含已离开与已超时的记录——排障时最需要的恰好
        是这些「看起来不该在」的会话。
        """
        plugin = self._plugin
        decision = plugin.decision_config
        focus_cfg = plugin.focus_config
        labels: dict[str, str] = {}

        def label(chat_id: str) -> str:
            if chat_id not in labels:
                try:
                    labels[chat_id] = plugin.short_term.label_for_chat(chat_id)
                except Exception:
                    labels[chat_id] = ""
            return labels[chat_id]

        try:
            patrol = plugin.patrol.snapshot()
        except Exception:
            patrol = {"windows": [], "judging": [], "cooldowns": []}
        patrol.setdefault("last_decisions", [])
        for item in patrol["windows"]:
            item["chat_label"] = label(str(item.get("chat_id") or ""))
        for item in patrol["cooldowns"]:
            item["chat_label"] = label(str(item.get("chat_id") or ""))
        for item in patrol["last_decisions"]:
            item["chat_label"] = label(str(item.get("chat_id") or ""))
        patrol["judging"] = [
            {"chat_id": chat_id, "chat_label": label(chat_id)}
            for chat_id in patrol["judging"]
        ]

        now = time.time()
        timeout = plugin.focus.timeout_seconds
        focus_items: list[dict] = []
        try:
            rows = plugin.focus.list_all(200)
        except Exception:
            rows = []
        for row in rows:
            chat_id = str(row["chat_id"] or "")
            focused_at = float(row["focused_at"] or 0.0)
            idle = max(now - focused_at, 0.0)
            if int(row["left"] or 0) == 1:
                status = "left"
            elif idle > timeout:
                status = "expired"
            else:
                status = "focused"
            focus_items.append(
                {
                    "chat_id": chat_id,
                    "chat_label": str(row["chat_label"] or "") or label(chat_id),
                    "focused_at": focused_at,
                    "idle_seconds": idle,
                    "status": status,
                }
            )

        def _active(method_name: str) -> bool:
            try:
                return bool(getattr(plugin, method_name)())
            except Exception:
                return False

        try:
            sleep = plugin.sleep_snapshot()
        except Exception:
            sleep = None

        return _ok(
            {
                "gate_active": _active("_gate_active"),
                "memory_active": _active("_memory_active"),
                "decision": {
                    "enabled": bool(decision.enabled),
                    "patrol_enabled": bool(decision.patrol_enabled),
                    "patrol_window_seconds": float(decision.patrol_window_seconds),
                    "patrol_cooldown_seconds": float(
                        decision.patrol_cooldown_seconds
                    ),
                    "patrol_noise_filter": bool(decision.patrol_noise_filter),
                    "private_debounce_seconds": float(
                        decision.private_debounce_seconds
                    ),
                    "private_debounce_max_seconds": float(
                        decision.private_debounce_max_seconds
                    ),
                    "context_messages": int(decision.context_messages),
                    "always_reply_private": bool(decision.always_reply_private),
                    "always_reply_when_mentioned": bool(
                        decision.always_reply_when_mentioned
                    ),
                },
                "focus_config": {
                    "timeout_minutes": float(focus_cfg.timeout_minutes),
                    "timeout_seconds": float(timeout),
                    "require_focus_for_say": bool(focus_cfg.require_focus_for_say),
                },
                "patrol": patrol,
                "focus_items": focus_items,
                "sleep": sleep,
            }
        )

    async def context(self) -> Any:
        """主脑上下文：不带 chat_id 时列出可查看的会话，带上则返回完整快照。"""
        chat_id = (_query_raw("chat_id") or "").strip()
        if not chat_id:
            return _ok({"items": self._plugin.context_sessions()})
        snapshot = self._plugin.context_snapshot(chat_id)
        if snapshot is None:
            return _err("该会话还没有捕获到主脑请求，先等它被唤醒一次", 404)
        return _ok(snapshot)

    async def update_memory(self) -> Any:
        """手工改写一条长期 / 永久记忆的正文。"""
        body = await _json_body()
        memory_id = _body_int(body, "id")
        if memory_id is None:
            return _err("缺少有效的记忆 id")
        content = str(body.get("content") or "").strip()
        if not content:
            return _err("记忆内容不能为空")
        if len(content) > _MAX_MEMORY_CHARS:
            return _err(f"记忆内容过长（最多 {_MAX_MEMORY_CHARS} 字）")

        store = self._plugin.memory_store
        if store.get(memory_id) is None:
            return _err("记忆不存在或已被删除", 404)
        if not store.update_content(memory_id, content):
            return _err("记忆更新失败")
        return _ok({"id": memory_id})

    async def delete_memory(self) -> Any:
        """删除一条长期 / 永久记忆，并阻止它被下一次压缩重新写回。

        只删除还不够：压缩模型看到同一段对话时会再次产出同一条记忆。
        因此先写入抑制表（与 `forget` 工具同一机制），再物理删除。
        """
        body = await _json_body()
        memory_id = _body_int(body, "id")
        if memory_id is None:
            return _err("缺少有效的记忆 id")

        store = self._plugin.memory_store
        memory = store.get(memory_id)
        if memory is None:
            return _err("记忆不存在或已被删除", 404)
        store.suppress(memory.content)
        store.delete(memory_id)
        return _ok({"id": memory_id})


def register_all_routes(plugin) -> None:
    """把页面路由挂到宿主；宿主不支持插件 Web API 时静默跳过。"""
    register = getattr(plugin.context, "register_web_api", None)
    if not callable(register):
        logger.warning("[什亭之枢] 当前 AstrBot 不支持插件 Web API，记忆页面不可用")
        return

    api = MemoryPageAPI(plugin)
    routes = (
        ("overview", api.overview, ["GET"], "记忆总览"),
        ("chats", api.chats, ["GET"], "会话列表"),
        ("short_term", api.short_term, ["GET"], "短期账本"),
        ("summaries", api.summaries, ["GET"], "短期摘要"),
        ("memories", api.memories, ["GET"], "长期与永久记忆"),
        ("profiles", api.profiles, ["GET"], "用户画像"),
        ("memories/update", api.update_memory, ["POST"], "编辑长期或永久记忆"),
        ("memories/delete", api.delete_memory, ["POST"], "删除长期或永久记忆"),
        ("runtime", api.runtime, ["GET"], "巡检参数与焦点状态"),
        ("context", api.context, ["GET"], "主脑当前上下文"),
    )
    for path, handler, methods, description in routes:
        # 重复注册同一路由时宿主会替换，热重载不会累积
        register(
            f"/astrbot_plugin_shittim/{path}",
            handler,
            methods,
            description,
        )
    logger.info("[什亭之枢] 记忆页面接口已注册（%d 个路由）", len(routes))
