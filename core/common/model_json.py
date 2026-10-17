"""从模型输出中解析 JSON 的共用工具。

决断层与记忆压缩都需要解析模型返回的 JSON，而模型常附带代码块围栏或前后的
解释性文字，因此将容错逻辑集中于此，避免两处各自实现。
"""

from __future__ import annotations

import json
import re


def parse_model_json(raw: str):
    """从模型输出中健壮地提取 JSON，兼容代码块围栏与前后附加文本。"""
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        pass
    # 兜底：截取最外层的数组或对象
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except (TypeError, ValueError):
                continue
    return None
