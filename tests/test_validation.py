"""Review Focus #2：MLLM 输出解析的全分支。"""
from genometag.annotate.prompts import parse_scores

EXPECTED = ["dark palette", "overrated"]


def test_valid_json():
    assert parse_scores('{"dark palette": 0.8, "overrated": 0.1}', EXPECTED) == {
        "dark palette": 0.8, "overrated": 0.1}


def test_json_with_surrounding_text():
    text = '好的，以下是评分：\n{"dark palette": 0.5, "overrated": 0.2}\n以上。'
    assert parse_scores(text, EXPECTED) == {"dark palette": 0.5, "overrated": 0.2}


def test_missing_tag_returns_none():
    assert parse_scores('{"dark palette": 0.8}', EXPECTED) is None


def test_out_of_range_returns_none():
    assert parse_scores('{"dark palette": 1.5, "overrated": 0.2}', EXPECTED) is None
    assert parse_scores('{"dark palette": -0.1, "overrated": 0.2}', EXPECTED) is None


def test_non_numeric_returns_none():
    assert parse_scores('{"dark palette": "high", "overrated": 0.2}', EXPECTED) is None


def test_garbage_returns_none():
    assert parse_scores("完全不是 JSON", EXPECTED) is None
    assert parse_scores("[]", EXPECTED) is None
