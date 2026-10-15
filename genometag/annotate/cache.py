"""LLM 调用磁盘缓存：成本控制、断点续跑、可复现的公共基础。

键 = SHA256(messages + model + 采样参数 + run_id)。run_id 必须入键：
多轮取平均/一致性分析会对同一 prompt 发多次请求。
值 = 完整原始响应（审计与换解析逻辑重算不需要重新付费）。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def cache_key(messages: list[dict], model: str, params: dict, run_id: str) -> str:
    payload = json.dumps(
        {"messages": messages, "model": model, "params": params, "run_id": run_id},
        ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


class ResponseCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> dict | None:
        p = self._path(key)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def put(self, key: str, response: dict) -> None:
        self._path(key).write_text(
            json.dumps(response, ensure_ascii=False, indent=1), encoding="utf-8")
