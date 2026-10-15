"""Prompt 组装：模态裁剪、帧采样、截断、mock 兼容契约、few-shot。"""
from pathlib import Path

from genometag.annotate.prompts import (
    FewShot, MOVIE_TITLE_LINE, TagDef, approx_tokens, build_messages,
    sample_keyframes, truncate_to_budget,
)
from genometag.config import ModelConfig
from genometag.data.movienet import MovieMultimodal


def _movie(tmp_path, *, script="S" * 2000, subs="D" * 1000):
    frames = []
    for i in range(10):
        p = tmp_path / f"f{i}.jpg"
        p.write_bytes(b"\xff\xd8fake")
        frames.append(p)
    return MovieMultimodal("mn1", "Test Film", 2020, "tt1", frames, script, subs)


TAGS = [TagDef(1, "dark palette", "色调偏暗"), TagDef(2, "overrated", "名过其实")]


def test_modality_filtering(tmp_path):
    movie = _movie(tmp_path)
    cfg = ModelConfig(max_frames=3)
    msgs = build_messages(movie, TAGS, ["script"], [], cfg)
    content = msgs[1]["content"]
    assert isinstance(content, list)
    assert len(content) == 1                       # 纯文本，无图像
    assert "剧本节选" in content[0]["text"]
    msgs_v = build_messages(movie, TAGS, ["video"], [], cfg)
    assert len(msgs_v[1]["content"]) == 1 + 3      # 文本 + 3 帧（10 帧采样到 3）


def test_sample_keyframes_even():
    frames = [Path(f"f{i}") for i in range(10)]
    assert sample_keyframes(frames, 3) == [Path("f0"), Path("f3"), Path("f6")]
    assert sample_keyframes(frames, 10) == frames   # 不超过则原样
    assert sample_keyframes(frames, 20) == frames


def test_truncate_to_budget():
    text = "a" * 1000
    assert truncate_to_budget(text, 100) == "a" * 400   # 4 字符/token
    assert truncate_to_budget("short", 10000) == "short"


def test_title_line_contract_with_mock():
    """Review Focus 契约：MOVIE_TITLE_LINE 生成的行必须能被 mock 的正则解析。"""
    import re
    line = MOVIE_TITLE_LINE.format(title="Crimson Horizon", year=2018)
    assert re.search(r"电影：《(.+?)》（", line).group(1) == "Crimson Horizon"


def test_few_shot_rendered(tmp_path):
    movie = _movie(tmp_path)
    fs = [FewShot("Old Film", 2015, {"dark palette": 0.9})]
    msgs = build_messages(movie, TAGS, ["script"], fs, ModelConfig(max_frames=2))
    text = msgs[1]["content"][0]["text"]
    assert "Old Film" in text and "dark palette: 0.9" in text
