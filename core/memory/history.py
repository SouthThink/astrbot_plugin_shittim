"""图片引用的可用性检查与失效引用摘除。

本模块只负责一件事：在请求发出前去除**确定不可用**的图片引用。

必要性：单个失效图片会导致**整条请求**被上游以 400 拒绝，AstrBot 随后会将
本轮请求切换到备用模型，造成语气突变、工具调用大概率失效。其代价远大于
「该图片未传递」。摘除后失效图片只影响自身。

（本模块此前还承担「从 AstrBot 官方会话历史回填短期记忆」。上下文改为全局
时间线整体重建后，官方历史不再进入上下文，回填逻辑一并移除。）
"""

from __future__ import annotations

import os

# 上游仅接受这几种图片引用。其余（本地路径）由 AstrBot 在 provider 层转为 base64；
# 转换失败的引用会被**原样透传**，严格的上游会直接返回 400 并导致整条请求失败。
_REMOTE_IMAGE_PREFIXES = ("data:", "http://", "https://")
_FILE_PREFIX = "file://"

# 整条消息只剩失效图片时写入的文本，保留「此处原本有图片」这一信息
BROKEN_IMAGE_PLACEHOLDER = "[图片已失效]"


def is_usable_image_ref(ref: str, cache_path: str = "") -> bool:
    """判断该图片引用能否安全地交给上游。

    远端引用与 data URL 一律放行；本地路径要求文件确实存在且非空，
    否则会被原样透传并触发 400。
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
    """摘除不可用的图片引用，返回 (处理后的 contexts, 被摘除的引用)。"""
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
        # 整条消息只剩失效图片时补一个文本占位，避免消息内容变为空
        new_entry["content"] = kept or [
            {"type": "text", "text": BROKEN_IMAGE_PLACEHOLDER}
        ]
        result.append(new_entry)

    return result, dropped


def strip_invalid_image_urls(urls: list) -> tuple[list, list[str]]:
    """与 strip_invalid_images 相同，但作用于 req.image_urls 这类字符串列表。"""
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
