"""缓存：键稳定性、run_id/参数维度、落盘往返。"""
from genometag.annotate.cache import ResponseCache, cache_key

MSGS = [{"role": "user", "content": "hi"}]


def test_key_stable():
    a = cache_key(MSGS, "gpt-4o", {"temperature": 0.7}, "run-0")
    b = cache_key(MSGS, "gpt-4o", {"temperature": 0.7}, "run-0")
    assert a == b and len(a) == 64


def test_key_differs_by_run_id():
    """Review Focus #1：同 prompt 不同轮次必须不同键，否则多轮取平均失效。"""
    a = cache_key(MSGS, "gpt-4o", {}, "run-0")
    b = cache_key(MSGS, "gpt-4o", {}, "run-1")
    assert a != b


def test_key_changes_with_params():
    a = cache_key(MSGS, "gpt-4o", {"temperature": 0.0}, "run-0")
    b = cache_key(MSGS, "gpt-4o", {"temperature": 1.0}, "run-0")
    assert a != b


def test_put_get_roundtrip(tmp_path):
    cache = ResponseCache(tmp_path)
    assert cache.get("nonexistent") is None
    resp = {"choices": [{"message": {"content": "{\"a\": 1}"}}], "usage": {"prompt_tokens": 3}}
    cache.put("k1", resp)
    assert cache.get("k1") == resp
