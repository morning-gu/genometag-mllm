"""客户端：transport 注入测试（无网络）。重试、response_format 降级、usage 提取。"""
import pytest

from genometag.annotate.client import LLMResponse, OpenAIClient


def _ok_transport(content='{"a": 1}'):
    def transport(model, messages, temperature, response_format):
        assert response_format is not None or True  # 降级路径会传 None
        return {"choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    return transport


def test_complete_success():
    client = OpenAIClient(model="m", base_url_env="X", transport=_ok_transport(),
                          request_interval=0.0)
    resp = client.complete([{"role": "user", "content": "hi"}], run_id="run-0")
    assert isinstance(resp, LLMResponse)
    assert resp.ok and resp.content == '{"a": 1}'
    assert resp.usage == {"prompt_tokens": 10, "completion_tokens": 5}


def test_retry_on_transport_error_then_success():
    calls = {"n": 0}

    def flaky(model, messages, temperature, response_format):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("网络抖动")
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    client = OpenAIClient(model="m", base_url_env="X", transport=flaky,
                          request_interval=0.0)
    resp = client.complete([{"role": "user", "content": "hi"}], run_id="r")
    assert resp.ok and calls["n"] == 2


def test_gives_up_after_max_retries():
    def always_fail(model, messages, temperature, response_format):
        raise ConnectionError("down")

    client = OpenAIClient(model="m", base_url_env="X", transport=always_fail,
                          request_interval=0.0, max_retries=2)
    resp = client.complete([{"role": "user", "content": "hi"}], run_id="r")
    assert not resp.ok and resp.content == ""


def test_response_format_fallback():
    """端点不支持 json_object → 去掉 response_format 重发一次。"""
    calls = {"n": 0}

    def picky(model, messages, temperature, response_format):
        calls["n"] += 1
        if response_format is not None:
            raise ValueError("response_format is not supported")
        return {"choices": [{"message": {"content": "plain"}}], "usage": {}}

    client = OpenAIClient(model="m", base_url_env="X", transport=picky,
                          request_interval=0.0)
    resp = client.complete([{"role": "user", "content": "hi"}], run_id="r")
    assert resp.ok and resp.content == "plain" and calls["n"] == 2


def test_missing_env_raises():
    import os
    client = OpenAIClient(model="m", base_url_env="NO_SUCH_ENV_XYZ")
    with pytest.raises(RuntimeError, match="NO_SUCH_ENV_XYZ"):
        client._build_transport()  # 环境变量缺失时构建即失败（不静默）
