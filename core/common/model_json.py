"""从模型输出里抠 JSON 的共用工具。

决断层与记忆压缩都要解析模型返回的 JSON，模型又常带上代码块围栏或前后的
解释性文字，所以容错逻辑集中在这里，避免两处各写一份。
"""

from __future__ import annotations

import json
import re


def parse_model_json(raw: str):
    """从模型输出里健壮地抠出 JSON，容忍代码块围栏与前后赘述。"""
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
