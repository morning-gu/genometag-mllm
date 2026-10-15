"""Mock 客户端：确定性、与合成 ground truth 的相关性、轮次噪声。"""
import json

import numpy as np

from genometag.annotate.mock import MockClient
from genometag.data.sample import SYNTHETIC_MOVIES, SYNTHETIC_TAGS, base_score

PROMPT = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": (
        "电影：《Crimson Horizon》（2018）\n\n"
        "## 剧本节选\nINT. ...\n\n"
        "## 需要打分的标签\n"
        "- neon visuals：画面以霓虹灯光为主\n"
        "- dark palette：整体色调偏暗\n"
        "- overrated：名过其实\n\n"
        "请输出 JSON。")},
]


def test_complete_returns_json_for_all_tags():
    client = MockClient()
    resp = client.complete(PROMPT, run_id="run-0")
    scores = json.loads(resp.content)
    assert set(scores) == {"neon visuals", "dark palette", "overrated"}
    assert all(0.0 <= v <= 1.0 for v in scores.values())
    assert resp.ok


def test_deterministic_same_run():
    a = MockClient().complete(PROMPT, run_id="run-0")
    b = MockClient().complete(PROMPT, run_id="run-0")
    assert a.content == b.content


def test_run_noise_varies_but_small():
    a = json.loads(MockClient().complete(PROMPT, "run-0").content)
    b = json.loads(MockClient().complete(PROMPT, "run-1").content)
    for tag in a:
        assert abs(a[tag] - b[tag]) <= 0.1  # correlation_noise=0.05 → |diff| ≤ 0.1


def test_correlation_with_synthetic_ground_truth():
    """合成 ground truth 与 mock 预测的 Pearson 相关 > 0.5（spec 的可控相关性）。"""
    client = MockClient()
    truth, pred = [], []
    for m in SYNTHETIC_MOVIES:
        tag_lines = "\n".join(f'- {t["tag"]}：{t["definition"]}' for t in SYNTHETIC_TAGS)
        prompt = [{"role": "user", "content": (
            f'电影：《{m["title"]}》（{m["year"]}）\n\n'
            f'## 需要打分的标签\n{tag_lines}\n\n请输出 JSON。')}]
        scores = json.loads(client.complete(prompt, "run-0").content)
        for t in SYNTHETIC_TAGS:
            truth.append(base_score(m["title"], t["tag"]))
            pred.append(scores[t["tag"]])
    r = np.corrcoef(truth, pred)[0, 1]
    assert r > 0.5
