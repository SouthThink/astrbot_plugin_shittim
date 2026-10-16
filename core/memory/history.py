"""图片引用的可用性检查与坏图摘除。

这里只做一件事：在请求发出前，把**确定读不到**的图片引用摘掉。

为什么必须做：一张失效的图片会让**整条请求**被上游 400 拒绝，而 AstrBot
随后会把这一轮切给备用模型——语气突变、工具调用多半失效。代价远大于
「这张图没传过去」。摘掉后坏图只影响它自己。

（本模块过去还承担「从 AstrBot 官方会话历史回填短期记忆」。上下文改为
全局时间线整体重建之后，官方历史不再进她的视野，回填一并退役。）
"""

from __future__ import annotations

import os

# 上游只接受这几种图片引用。其余（本地路径）由 AstrBot 在 provider 层转 base64；
# 转换失败的会**原样透传**，严格的上游就会直接 400 并让整条请求失败。
_REMOTE_IMAGE_PREFIXES = ("data:", "http://", "https://")
_FILE_PREFIX = "file://"

# 整条消息只剩坏图时留下的文本，让「这里本来有张图」这件事不至于彻底消失
BROKEN_IMAGE_PLACEHOLDER = "[图片已失效]"


def is_usable_image_ref(ref: str, cache_path: str = "") -> bool:
    """这个图片引用能不能安全地交给上游。

    远端引用与 data URL 一律放行；本地路径要求文件确实存在且非空，
    否则它一定会被原样透传出去触发 400。
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
    """摘掉读不到的图片引用，返回 (处理后的 contexts, 被摘掉的引用)。"""
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
        # 整条消息只剩坏图时补一个文本占位，别让这条消息变成空内容
        new_entry["content"] = kept or [
            {"type": "text", "text": BROKEN_IMAGE_PLACEHOLDER}
        ]
        result.append(new_entry)

    return result, dropped


def strip_invalid_image_urls(urls: list) -> tuple[list, list[str]]:
    """同上，但作用于 req.image_urls 这种「字符串列表」形式。"""
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
