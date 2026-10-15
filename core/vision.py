"""图片转文字：让她真的「看得见」群里发的图。

## 为什么自己做

AstrBot 自带的图片转述只发生在**主脑的请求链路**上：它把图交给视觉模型，
描述只存在于那一次请求里。而记账、决断、跨会话语境块、记忆页面用的都是
**我们这份账本**——账本里图片永远是 `[图片]` 这个占位。于是：

- 判定时她看到的是「[图片]」，无从判断该不该接话
- 记忆页面上也只能看到一个占位，主人看不到图里是什么

所以在消息到达时自己转述一次，把描述写回账本，后面所有环节都自动受益。

## 用哪个模型

默认复用 AstrBot 全局设置里的「图片转述模型」
（`provider_settings.default_image_caption_provider_id`），
也就是 WebUI 里配好的那个；插件配置里填了 provider 则以插件为准。

## 拿不到图怎么办

上游的图片链接会过期、本地临时文件会被清理。这里只在**消息到达的那一刻**
处理：当场把本地文件读成 data URL 交给模型，读完即走，不长期保存图片。
读不到就如实放弃，账本里继续留占位——宁可没有描述，也不要编一段出来。
"""

from __future__ import annotations

import base64
import hashlib
import os
import time
from urllib.parse import unquote, urlparse

try:
    from astrbot.api import logger
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

# 读图上限：单张图超过这个大小就不转述了，避免把大图塞进请求
_MAX_IMAGE_BYTES = 10 * 1024 * 1024
# 描述长度上限：一句话足矣，长了会污染上下文
_MAX_CAPTION_CHARS = 200

_DEFAULT_PROMPT = (
    "这是一张聊天里发的图片。请用一句中文客观描述画面内容，"
    "重点说清楚画面里有什么、写了什么字或表达了什么，"
    "不要寒暄，不要加「这张图片」之类的开场，直接给描述。"
)


def _sniff_mime(raw: bytes) -> str:
    """按文件魔数猜图片类型，猜不出按 jpeg 处理。"""
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"GIF8"):
        return "image/gif"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    if raw.startswith(b"BM"):
        return "image/bmp"
    return "image/jpeg"


def _local_path(ref: str) -> str:
    """把 file:// 与裸路径统一成本地路径，不是本地引用则返回空串。"""
    if ref.startswith("file://"):
        parsed = urlparse(ref)
        path = unquote(parsed.path or "")
        # Windows 下 file:///E:/x 会解析成 /E:/x，去掉前导斜杠
        if os.name == "nt" and len(path) > 2 and path[0] == "/" and path[2] == ":":
            path = path[1:]
        return path
    if ref.startswith("http://") or ref.startswith("https://"):
        return ""
    if ref.startswith("data:"):
        return ""
    if os.path.isabs(ref):
        return ref
    return ""


def _read_local(path: str) -> bytes:
    try:
        if not path or not os.path.isfile(path):
            return b""
        if os.path.getsize(path) > _MAX_IMAGE_BYTES:
            logger.warning("[什亭之枢] 图片过大，跳过转述: %s", path)
            return b""
        with open(path, "rb") as handle:
            return handle.read()
    except OSError as error:
        logger.debug("[什亭之枢] 读取图片失败: %s | %s", path, error)
        return b""


def to_model_ref(ref: str) -> str:
    """把一条图片引用整理成模型能直接吃的形式。

    - 本地文件 → data URL（上游只认 base64 与 http(s)，本地路径会被原样透传）
    - `base64://` → data URL
    - http(s) 与 data URL → 原样
    拿不到内容时返回空串，调用方据此跳过。
    """
    ref = (ref or "").strip()
    if not ref:
        return ""
    if ref.startswith("data:"):
        return ref
    if ref.startswith("base64://"):
        payload = ref[len("base64://") :]
        try:
            raw = base64.b64decode(payload, validate=False)
        except Exception:
            return ""
        return f"data:{_sniff_mime(raw)};base64,{payload}"
    if ref.startswith("http://") or ref.startswith("https://"):
        return ref
    raw = _read_local(_local_path(ref))
    if not raw:
        return ""
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{_sniff_mime(raw)};base64,{encoded}"


class ImageCaptioner:
    """把消息里的图片转成一句话，写回账本。"""

    def __init__(self, db, config, astr_context) -> None:
        self._db = db
        self._config = config
        self._astr_context = astr_context

    def update_config(self, config) -> None:
        self._config = config

    # ── 取图 ──────────────────────────────────────────────

    @staticmethod
    def collect_refs(event, limit: int) -> list[str]:
        """从事件的消息链里取图片引用，最多取 limit 条。

        优先 `path`（AstrBot 归一化后的本地路径），再退到 `file` / `url`。
        用类名而不是 isinstance 判断，免得绑死某个 AstrBot 版本的导入路径。
        """
        try:
            components = event.get_messages() or []
        except Exception:
            return []
        refs: list[str] = []
        for component in components:
            if type(component).__name__ != "Image":
                continue
            for attribute in ("path", "file", "url"):
                value = getattr(component, attribute, None)
                if isinstance(value, str) and value.strip():
                    refs.append(value.strip())
                    break
            if len(refs) >= max(int(limit), 1):
                break
        return refs

    # ── 转述 ──────────────────────────────────────────────

    async def describe(self, refs: list[str]) -> str:
        """给一批图片生成一句描述（取第一张成功的）。失败返回空串。"""
        provider = self._resolve_provider()
        if provider is None:
            logger.warning(
                "[什亭之枢] 没有可用的图片转述模型（插件配置与 AstrBot 的"
                "「图片转述模型」都为空），图片只会留下「[图片]」占位"
            )
            return ""
        if not self._supports_image(provider):
            logger.warning(
                "[什亭之枢] 图片转述模型不支持图片输入，已跳过（请换一个视觉模型）"
            )
            return ""

        prompt = str(getattr(self._config, "vision_prompt", "") or "").strip()
        if not prompt:
            prompt = self._global_prompt() or _DEFAULT_PROMPT

        for ref in refs:
            cached = self._cached(ref)
            if cached:
                logger.debug("[什亭之枢] 图片转述命中缓存")
                return cached
            model_ref = to_model_ref(ref)
            if not model_ref:
                continue
            try:
                response = await provider.text_chat(
                    prompt=prompt, image_urls=[model_ref]
                )
            except Exception as error:
                logger.warning("[什亭之枢] 图片转述失败: %s", error)
                continue
            caption = (getattr(response, "completion_text", "") or "").strip()
            caption = " ".join(caption.split())[:_MAX_CAPTION_CHARS]
            if not caption:
                continue
            self._store(ref, caption)
            return caption
        return ""

    def _resolve_provider(self):
        provider_id = str(getattr(self._config, "vision_provider_id", "") or "").strip()
        if not provider_id:
            provider_id = self._global_caption_provider_id()
        if not provider_id or self._astr_context is None:
            return None
        try:
            return self._astr_context.get_provider_by_id(provider_id)
        except Exception:
            return None

    def _global_config(self):
        """AstrBot 全局配置；拿不到就返回空 dict。"""
        try:
            config = self._astr_context.get_config()
        except Exception:
            return {}
        return config if isinstance(config, dict) else {}

    def _global_caption_provider_id(self) -> str:
        """WebUI 里「图片转述模型」的配置值。"""
        config = self._global_config()
        settings = config.get("provider_settings")
        if not isinstance(settings, dict):
            return ""
        return str(settings.get("default_image_caption_provider_id") or "").strip()

    def _global_prompt(self) -> str:
        config = self._global_config()
        settings = config.get("provider_settings")
        if not isinstance(settings, dict):
            return ""
        return str(settings.get("image_caption_prompt") or "").strip()

    @staticmethod
    def _supports_image(provider) -> bool:
        """模型是否声明支持图片。

        未声明时按支持处理：有些适配器不填 modalities，一刀切拦掉会让功能
        直接失效；真发不过去，调用处还有异常兜底。
        """
        try:
            modalities = provider.provider_config.get("modalities")
        except Exception:
            return True
        if not isinstance(modalities, list) or not modalities:
            return True
        return "image" in [str(item).lower() for item in modalities]

    # ── 缓存 ──────────────────────────────────────────────

    @staticmethod
    def _ref_key(ref: str) -> str:
        return hashlib.sha1((ref or "").encode("utf-8", "ignore")).hexdigest()

    def _cached(self, ref: str) -> str:
        row = self._db.query_one(
            "SELECT caption FROM image_captions WHERE ref_hash = ?",
            (self._ref_key(ref),),
        )
        return str(row["caption"]) if row else ""

    def _store(self, ref: str, caption: str) -> None:
        self._db.execute(
            """
            INSERT INTO image_captions (ref_hash, caption, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(ref_hash) DO UPDATE SET caption = excluded.caption
            """,
            (self._ref_key(ref), caption, time.time()),
        )
