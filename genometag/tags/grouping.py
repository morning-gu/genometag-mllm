"""Tag 语义分组：1,128 tag → ~20 组 + content/preference 类型 + 一句话定义。

分组是一次性离线产物（tag_groups.json，入 git）。跨批一致性：
后续批次的 prompt 携带已出现的组名，引导模型复用而非发明新组。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

VALID_TAG_TYPES = ("content", "preference")

_GROUPING_SYSTEM = "你是电影标签体系专家，负责为 MovieLens genome tag 建立语义分组。"


def build_grouping_messages(tag_batch: list[str], known_groups: list[str]) -> list[dict]:
    hint = ""
    if known_groups:
        hint = "已有组名（优先复用，不要新建语义重复的组）：" + "、".join(known_groups) + "\n"
    user = (
        hint
        + "将以下电影标签按语义分组（目标约 20 组，如：视觉风格、情绪氛围、叙事主题、"
        "角色特征、制作要素、受众偏好……），并为每个标签标注类型"
        "（content = 描述电影本身内容特征；preference = 表达观众偏好/评价）"
        "与一句话定义。\n\n标签：\n"
        + "\n".join(f"- {t}" for t in tag_batch)
        + '\n\n只输出 JSON：{"标签名": {"group": "组名", "tag_type": "content|preference", '
          '"definition": "一句话定义"}, ...}'
    )
    return [{"role": "system", "content": _GROUPING_SYSTEM},
            {"role": "user", "content": user}]


def parse_grouping_response(text: str, expected: list[str]) -> dict[str, dict] | None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # 与 parse_scores 一致的降级：模型可能用 ```json 围栏或前后缀文本包裹
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group())
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    out: dict[str, dict] = {}
    for tag in expected:
        entry = data.get(tag)
        if not isinstance(entry, dict):
            return None
        group, ttype, definition = entry.get("group"), entry.get("tag_type"), entry.get("definition")
        if not isinstance(group, str) or not group or ttype not in VALID_TAG_TYPES \
                or not isinstance(definition, str) or not definition:
            return None
        out[tag] = {"group": group, "tag_type": ttype, "definition": definition}
    return out


def validate_coverage(parsed: dict[str, dict], expected: list[str]) -> list[str]:
    return [t for t in expected if t not in parsed]


def group_tags_via_llm(tags: list[str], client, cache, model: str,
                       batch_size: int = 150) -> pd.DataFrame:
    """分批调用 LLM 分组；每批失败重试一次，仍失败抛 RuntimeError（不静默丢 tag）。"""
    from genometag.annotate.cache import cache_key

    rows: list[dict] = []
    known_groups: list[str] = []
    for start in range(0, len(tags), batch_size):
        batch = tags[start:start + batch_size]
        messages = build_grouping_messages(batch, known_groups)
        parsed = None
        for attempt in range(2):
            run_id = "grouping" if attempt == 0 else "grouping#r1"
            key = cache_key(messages, model, {}, run_id)
            raw = cache.get(key) if cache else None
            if raw is None:
                resp = client.complete(messages, run_id=run_id)
                raw = resp.raw
                if cache:
                    cache.put(key, raw)
                content = resp.content
            else:
                content = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
            parsed = parse_grouping_response(content, batch)
            if parsed is not None:
                break
        if parsed is None:
            raise RuntimeError(f"分组批次失败（tags 起始: {batch[0]}，重试后仍无法解析）")
        missing = validate_coverage(parsed, batch)
        if missing:
            raise RuntimeError(f"分组批次缺失标签: {missing[:5]}")
        for tag, entry in parsed.items():
            rows.append({"tag": tag, **entry})
            if entry["group"] not in known_groups:
                known_groups.append(entry["group"])
    return pd.DataFrame(rows)


def load_tag_groups(path: Path) -> pd.DataFrame:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return pd.DataFrame(data)[["tag_id", "tag", "group", "tag_type", "definition"]]
