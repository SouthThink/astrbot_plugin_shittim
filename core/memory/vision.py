"""图片转述：把消息中的图片转换为文本描述并写回账本。

## 独立实现的原因

AstrBot 内置的图片转述只作用于**主脑请求链路**：描述仅存在于该次请求内。
而记账、决断、跨会话语境块与记忆页面读取的都是**本插件的账本**，
账本中图片始终为 `[图片]` 占位符，因此：

- 决断层只能看到占位符，无法据此判断是否参与对话
- 记忆页面同样只显示占位符，无法呈现图片内容

因此在消息到达时单独转述一次并写回账本，后续所有环节均可直接使用。

## 模型选择

默认复用 AstrBot 全局设置中的图片转述模型
（`provider_settings.default_image_caption_provider_id`）；
插件配置中指定了 provider 时以插件配置为准。

## 图片引用失效处理

上游链接可能过期，本地临时文件可能被清理，因此仅在**消息到达时**处理：
当场把本地文件读取为 data URL 交给模型，不长期保存图片。
读取失败时保留占位符，不生成推测性描述。
"""

from __future__ import annotations

import base64
import hashlib
import os
import time
from urllib.parse import unquote, urlparse

from astrbot.api import logger

# 单张图片大小上限：超出则不转述，避免大图占用请求体积
_MAX_IMAGE_BYTES = 10 * 1024 * 1024
# 描述文本长度上限：过长会占用上下文预算
_MAX_CAPTION_CHARS = 200

_DEFAULT_PROMPT = (
    "这是一张聊天里发的图片。请用一句中文客观描述画面内容，"
    "重点说清楚画面里有什么、写了什么字或表达了什么，"
    "不要寒暄，不要加「这张图片」之类的开场，直接给描述。"
)


def _sniff_mime(raw: bytes) -> str:
    """按文件魔数推断图片类型，无法识别时按 jpeg 处理。"""
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
    """把一条图片引用转换为模型可直接使用的形式。

    - 本地文件 → data URL（上游仅接受 base64 与 http(s)，本地路径会被原样透传）
    - `base64://` → data URL
    - http(s) 与 data URL → 原样返回
    内容不可用时返回空串，由调用方跳过。
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
    """把消息中的图片转成一句描述，写回账本。"""

    def __init__(self, db, config, astr_context) -> None:
        self._db = db
        self._config = config
        self._astr_context = astr_context

    def update_config(self, config) -> None:
        self._config = config

    # ── 取图 ──────────────────────────────────────────────

    @staticmethod
    def collect_refs(event, limit: int) -> list[str]:
        """从事件的消息链中提取图片引用，最多 limit 条。

        优先 `path`（AstrBot 归一化后的本地路径），其次 `file` / `url`。
        按类名而非 isinstance 判断，避免绑定特定 AstrBot 版本的导入路径。
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
        """为一批图片生成一句描述（取第一张成功的）。失败返回空串。"""
        provider = self._resolve_provider()
        if provider is None:
            logger.warning(
                "[什亭之枢] 没有可用的图片转述模型（插件配置与 AstrBot 的"
                "「图片转述模型」均为空），图片只能保留「[图片]」占位"
            )
            return ""
        if not self._supports_image(provider):
            logger.warning(
                "[什亭之枢] 图片转述模型不支持图片输入，已跳过（请改用视觉模型）"
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
        """AstrBot 全局配置；不可用时返回空 dict。"""
        try:
            config = self._astr_context.get_config()
        except Exception:
            return {}
        return config if isinstance(config, dict) else {}

    def _global_caption_provider_id(self) -> str:
        """WebUI 中「图片转述模型」的配置值。"""
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

        未声明时按支持处理：部分适配器不填 modalities，一律拦截会导致功能
        直接失效；实际调用失败时由调用处的异常处理兜底。
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
