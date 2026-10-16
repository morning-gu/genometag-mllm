"""Tag 分组：响应解析、覆盖校验、跨批合并、tag_groups 加载。"""
import json

import pandas as pd
import pytest

from genometag.tags.grouping import (
    build_grouping_messages, group_tags_via_llm, load_tag_groups,
    parse_grouping_response, validate_coverage,
)

GOOD = '{"dark": {"group": "视觉风格", "tag_type": "content", "definition": "色调偏暗"}, '\
       '"overrated": {"group": "受众偏好", "tag_type": "preference", "definition": "名过其实"}}'


def test_parse_grouping_response():
    parsed = parse_grouping_response(GOOD, ["dark", "overrated"])
    assert parsed["dark"]["tag_type"] == "content"
    assert parsed["overrated"]["group"] == "受众偏好"


def test_parse_missing_tag_returns_none():
    assert parse_grouping_response(GOOD, ["dark", "missing"]) is None


def test_parse_bad_type_returns_none():
    bad = json.dumps({"dark": {"group": "g", "tag_type": "weird", "definition": "d"}})
    assert parse_grouping_response(bad, ["dark"]) is None


def test_validate_coverage():
    parsed = json.loads(GOOD)
    assert validate_coverage(parsed, ["dark", "overrated", "tense"]) == ["tense"]


class _ScriptedClient:
    """按 prompt 中出现的 tag 顺序返回预置分组结果。"""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls = 0

    def complete(self, messages, run_id):
        self.calls += 1
        return _Resp(self.responses.pop(0))


class _Resp:
    def __init__(self, content):
        self.content, self.usage, self.raw, self.ok = content, {}, {}, True


def test_group_tags_via_llm_batches_merge():
    tags = ["dark", "overrated", "tense", "neon visuals"]
    r1 = json.dumps({t: {"group": f"G{i}", "tag_type": "content", "definition": "d"}
                     for i, t in enumerate(tags[:2])})
    r2 = json.dumps({t: {"group": f"G{i}", "tag_type": "preference", "definition": "d"}
                     for i, t in enumerate(tags[2:])})
    client = _ScriptedClient([r1, r2])
    df = group_tags_via_llm(tags, client, cache=None, model="m", batch_size=2)
    assert len(df) == 4 and client.calls == 2
    assert set(df.columns) == {"tag", "group", "tag_type", "definition"}


def test_load_tag_groups(tmp_path):
    data = [{"tag_id": 1, "tag": "dark", "group": "视觉风格",
             "tag_type": "content", "definition": "色调偏暗"}]
    p = tmp_path / "tag_groups.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    df = load_tag_groups(p)
    assert list(df.columns) == ["tag_id", "tag", "group", "tag_type", "definition"]


def test_group_tags_via_llm_recovers_missing_tags():
    """宽松接受 + 缺失补漏：第一批漏 1 个 tag → 补漏批返回 → 4/4 全部到位。"""
    def _entry(group="G1", ttype="content"):
        return {"group": group, "tag_type": ttype, "definition": "d"}

    # 第一批（4 tags）只返回 3 个——合法 JSON、finish=stop，但漏 "d"（服务器实测场景）
    partial = json.dumps({t: _entry() for t in ["a", "b", "c"]})
    gapfill = json.dumps({"d": _entry()})
    client = _ScriptedClient([partial, gapfill])
    df = group_tags_via_llm(["a", "b", "c", "d"], client, cache=None,
                            model="m", batch_size=4)
    assert len(df) == 4 and set(df["tag"]) == {"a", "b", "c", "d"}
    assert client.calls == 2                       # 原批 + 补漏批


def test_group_tags_via_llm_raises_after_gapfill_exhausted():
    """补漏 3 次尝试后仍缺 → RuntimeError（不静默丢 tag 的底线不变）。"""
    import pytest

    def _entry():
        return {"group": "G1", "tag_type": "content", "definition": "d"}

    # 每次响应都漏 "d"
    partial = json.dumps({t: _entry() for t in ["a", "b", "c"]})
    client = _ScriptedClient([partial, partial, partial])
    with pytest.raises(RuntimeError, match="缺失"):
        group_tags_via_llm(["a", "b", "c", "d"], client, cache=None,
                           model="m", batch_size=4)
