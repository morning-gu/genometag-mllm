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


def test_parse_scores_lenient_partial_and_string_numbers():
    """宽松解析：漏 tag 跳过、字符串数值可转、越界/垃圾跳过——不全有或全无。"""
    from genometag.annotate.prompts import parse_scores_lenient

    text = '{"a": 0.5, "b": "0.75", "c": 1.5, "d": "junk", "e": true}'
    assert parse_scores_lenient(text, ["a", "b", "c", "d", "e", "missing"]) == {
        "a": 0.5, "b": 0.75}
    assert parse_scores_lenient("完全不是 JSON", ["a"]) == {}
    # 围栏包裹
    assert parse_scores_lenient('```json\n{"a": 0.2}\n```', ["a"]) == {"a": 0.2}
