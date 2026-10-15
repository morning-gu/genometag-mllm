"""Prompt 组装（模态裁剪、预算截断、few-shot）与输出解析。

文本块格式是 MockClient（annotate/mock.py）的解析契约：
- 电影标题行：MOVIE_TITLE_LINE
- 标签块：TAGS_BLOCK_START 后每行 "- {tag}：{definition}"
"""
from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path

from genometag.config import ModelConfig
from genometag.data.movienet import MovieMultimodal

SYSTEM_PROMPT = "你是电影内容分析专家。根据提供的电影内容为标签打关联度分，输出严格的 JSON 对象。"
MOVIE_TITLE_LINE = "电影：《{title}》（{year}）"
TAGS_BLOCK_START = "## 需要打分的标签"
_INSTRUCTIONS = (
    "对下面每一个标签，给出该电影与之的关联度评分（0.0-1.0，保留两位小数）。\n"
    "只输出 JSON 对象，键为标签名、值为分数，不要输出其他内容。"
)


@dataclass
class TagDef:
    tag_id: int
    tag: str
    definition: str


@dataclass
class FewShot:
    title: str
    year: int
    scores: dict[str, float]


def approx_tokens(text: str) -> int:
    """近似 token 数（英文约 4 字符/token；预算截断用，不追求精确）。"""
    return max(1, len(text) // 4)


def truncate_to_budget(text: str, token_budget: int) -> str:
    if approx_tokens(text) <= token_budget:
        return text
    return text[: token_budget * 4]


def sample_keyframes(frames: list[Path], max_frames: int) -> list[Path]:
    if len(frames) <= max_frames:
        return list(frames)
    step = len(frames) / max_frames
    return [frames[int(i * step)] for i in range(max_frames)]


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def build_messages(movie: MovieMultimodal, tag_defs: list[TagDef],
                   modalities: list[str], few_shots: list[FewShot],
                   model_cfg: ModelConfig) -> list[dict]:
    parts: list[str] = [
        MOVIE_TITLE_LINE.format(title=movie.title, year=movie.year if movie.year else "未知年份"),
        _INSTRUCTIONS,
    ]
    if few_shots:
        lines = ["## 示例（已知关联度的电影）"]
        for fs in few_shots:
            lines.append(f"《{fs.title}》（{fs.year}）：" + ", ".join(
                f"{t}: {s}" for t, s in fs.scores.items()))
        parts.append("\n".join(lines))

    sampled: list[Path] = []
    if "video" in modalities and movie.keyframes:
        sampled = sample_keyframes(movie.keyframes, model_cfg.max_frames)
        parts.append(f"## 关键帧（{len(sampled)} 张，见后附图像）")
    if "script" in modalities and movie.script:
        parts.append("## 剧本节选\n" + truncate_to_budget(movie.script, model_cfg.script_token_budget))
    if "dialogue" in modalities and movie.subtitles:
        parts.append("## 台词/字幕\n" + truncate_to_budget(movie.subtitles, model_cfg.dialogue_token_budget))

    tag_lines = [f"- {t.tag}：{t.definition}" for t in tag_defs]
    parts.append(TAGS_BLOCK_START + "\n" + "\n".join(tag_lines))

    content: list[dict] = [{"type": "text", "text": "\n\n".join(parts)}]
    for frame in sampled:
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encode_image(frame)}"}})
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content}]


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_scores(text: str, expected_tags: list[str]) -> dict[str, float] | None:
    """解析模型输出。任一 tag 缺失/非数值/越界/不可解析 → None（调用方决定重试）。"""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_RE.search(text)
        if not m:
            return None
        try:
            data = json.loads(m.group())
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    scores: dict[str, float] = {}
    for tag in expected_tags:
        if tag not in data:
            return None
        v = data[tag]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        if not 0.0 <= float(v) <= 1.0:
            return None
        scores[tag] = float(v)
    return scores
