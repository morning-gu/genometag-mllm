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
