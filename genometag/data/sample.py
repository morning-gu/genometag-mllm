"""合成小样本：结构镜像真实原始数据格式，用于开发、测试与冒烟。

产出三块：
- raw/movienet/   （镜像 MovieNet 假定格式，见 docs/data-format.md）
- raw/movielens/  （镜像 MovieLens ml-latest 格式）
- processed/tag_groups.json（合成样本的分组是已知的，直接生成，不走 LLM）

ground truth 生成规则：relevance = 0.85 * base_score(title, tag) + 0.15 * 独立噪声。
MockClient（genometag/annotate/mock.py）预测 base_score + 小幅轮次噪声，
两者天然相关——这是 mock"可控相关性"的实现基础。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

SYNTHETIC_MOVIES = [
    # movie_id(ML), movienet_id, title, year, genres, imdb_id(不带 tt 前缀的数字)
    {"movie_id": 1, "movienet_id": "mn01", "title": "Crimson Horizon", "year": 2018,
     "genres": ["Action", "Thriller"], "imdb_id": 1},
    {"movie_id": 2, "movienet_id": "mn02", "title": "Neon Alley", "year": 2019,
     "genres": ["Crime", "Drama"], "imdb_id": 2},
    {"movie_id": 3, "movienet_id": "mn03", "title": "Quiet Harbor", "year": 2020,
     "genres": ["Drama", "Romance"], "imdb_id": 3},
    {"movie_id": 4, "movienet_id": "mn04", "title": "Last Ember", "year": 2021,
     "genres": ["Sci-Fi", "Action"], "imdb_id": 4},
    {"movie_id": 5, "movienet_id": "mn05", "title": "Paper Skies", "year": 2023,
     "genres": ["Drama", "Mystery"], "imdb_id": 5},
]

SYNTHETIC_TAGS = [
    # group: 视觉风格 / 情绪氛围 / 叙事主题 / 受众偏好
    {"tag_id": 1,  "tag": "neon visuals",    "group": "视觉风格", "tag_type": "content",
     "definition": "画面以霓虹灯光与高饱和夜景为主"},
    {"tag_id": 2,  "tag": "dark palette",    "group": "视觉风格", "tag_type": "content",
     "definition": "整体色调偏暗、低照明场景占主导"},
    {"tag_id": 3,  "tag": "vivid colors",    "group": "视觉风格", "tag_type": "content",
     "definition": "色彩鲜艳明亮、对比强烈"},
    {"tag_id": 4,  "tag": "grainy texture",  "group": "视觉风格", "tag_type": "content",
     "definition": "胶片颗粒感或做旧质感"},
    {"tag_id": 5,  "tag": "static camera",   "group": "视觉风格", "tag_type": "content",
     "definition": "大量固定机位长镜头"},
    {"tag_id": 6,  "tag": "slow pans",       "group": "视觉风格", "tag_type": "content",
     "definition": "缓慢的横摇/横移镜头"},
    {"tag_id": 7,  "tag": "melancholic",     "group": "情绪氛围", "tag_type": "content",
     "definition": "整体情绪忧郁、低落"},
    {"tag_id": 8,  "tag": "tense",           "group": "情绪氛围", "tag_type": "content",
     "definition": "持续紧张、压迫感"},
    {"tag_id": 9,  "tag": "hopeful",         "group": "情绪氛围", "tag_type": "content",
     "definition": "充满希望与治愈感"},
    {"tag_id": 10, "tag": "bleak",           "group": "情绪氛围", "tag_type": "content",
     "definition": "荒凉、绝望的世界观"},
    {"tag_id": 11, "tag": "dreamlike",       "group": "情绪氛围", "tag_type": "content",
     "definition": "梦幻、超现实的氛围"},
    {"tag_id": 12, "tag": "nostalgic",       "group": "情绪氛围", "tag_type": "content",
     "definition": "怀旧、追忆过去的情绪"},
    {"tag_id": 13, "tag": "revenge plot",    "group": "叙事主题", "tag_type": "content",
     "definition": "以复仇为核心驱动力"},
    {"tag_id": 14, "tag": "family bonds",    "group": "叙事主题", "tag_type": "content",
     "definition": "家庭关系与亲情羁绊"},
    {"tag_id": 15, "tag": "identity crisis", "group": "叙事主题", "tag_type": "content",
     "definition": "主角经历身份认同危机"},
    {"tag_id": 16, "tag": "heist plan",      "group": "叙事主题", "tag_type": "content",
     "definition": "围绕精心策划的抢劫/盗窃行动"},
    {"tag_id": 17, "tag": "lost love",       "group": "叙事主题", "tag_type": "content",
     "definition": "失去的爱情或错过的恋人"},
    {"tag_id": 18, "tag": "survival",        "group": "叙事主题", "tag_type": "content",
     "definition": "绝境求生主线"},
    {"tag_id": 19, "tag": "overrated",       "group": "受众偏好", "tag_type": "preference",
     "definition": "名过其实、评价高于实际质量"},
    {"tag_id": 20, "tag": "cult classic",    "group": "受众偏好", "tag_type": "preference",
     "definition": "小众狂热追捧的经典"},
    {"tag_id": 21, "tag": "feel-good",       "group": "受众偏好", "tag_type": "preference",
     "definition": "看完心情变好的轻松作品"},
    {"tag_id": 22, "tag": "slow burn",       "group": "受众偏好", "tag_type": "preference",
     "definition": "节奏缓慢、渐入佳境"},
    {"tag_id": 23, "tag": "rewatchable",     "group": "受众偏好", "tag_type": "preference",
     "definition": "值得反复观看"},
    {"tag_id": 24, "tag": "niche appeal",    "group": "受众偏好", "tag_type": "preference",
     "definition": "只对特定人群有吸引力"},
]


def _hash_unit(key: str) -> float:
    """哈希 → 确定性 [0,1) 均匀值。"""
    h = hashlib.sha256(key.encode()).hexdigest()
    return int(h[:8], 16) / 0x1_0000_0000


def base_score(title: str, tag: str) -> float:
    """电影×标签 的确定性基准分。ground truth 与 MockClient 共用（相关性的来源）。"""
    return _hash_unit(f"base|{title}|{tag}")


def _truth(title: str, tag: str) -> float:
    """ground truth = 0.85*base + 0.15*独立噪声，clip 到 [0,1]。"""
    noise = _hash_unit(f"truth|{title}|{tag}")
    return min(1.0, max(0.0, 0.85 * base_score(title, tag) + 0.15 * noise))


def _make_frame(path: Path, color: tuple, index: int) -> None:
    """生成一张渐变关键帧（尺寸小，冒烟要快）。"""
    img = Image.new("RGB", (64, 36), color)
    draw = ImageDraw.Draw(img)
    shade = int(255 * index / 8)
    draw.rectangle([0, 0, 63, 18], fill=(min(255, color[0] + shade), color[1], color[2]))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=60)


_MOVIE_COLORS = [(120, 20, 20), (20, 20, 120), (20, 120, 60), (80, 80, 20), (60, 20, 80)]


def _script_text(title: str) -> str:
    return (
        f"INT. {title.upper()} - OPENING - NIGHT\n\n"
        "The city hums outside. Rain streaks the window.\n\n"
        f"Our protagonist studies a photograph. The story of {title} begins here.\n\n"
        "EXT. STREET - CONTINUOUS\n\nTraffic. Neon reflections in puddles.\n"
    )


def _srt_text(title: str) -> str:
    return (
        "1\n00:00:01,000 --> 00:00:03,500\n"
        f"They said {title} would change everything.\n\n"
        "2\n00:00:04,000 --> 00:00:06,000\n"
        "Nothing changes. We just keep moving.\n\n"
        "3\n00:00:07,000 --> 00:00:09,500\n"
        "Then move faster.\n"
    )


def make_synthetic_dataset(out_root: Path) -> None:
    """生成完整合成数据集。out_root 即 data/synthetic/。"""
    out_root = Path(out_root)
    mn = out_root / "raw" / "movienet"
    ml = out_root / "raw" / "movielens"
    mn.mkdir(parents=True, exist_ok=True)
    ml.mkdir(parents=True, exist_ok=True)

    # MovieNet 侧（write_text 不会自动建目录，先建 scripts/ 与 subtitles/）
    (mn / "scripts").mkdir(parents=True, exist_ok=True)
    (mn / "subtitles").mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(SYNTHETIC_MOVIES):
        for f in range(8):
            _make_frame(mn / "keyframes" / m["movienet_id"] / f"frame_{f:03d}.jpg",
                        _MOVIE_COLORS[i], f)
        (mn / "scripts" / f'{m["movienet_id"]}.txt').write_text(
            _script_text(m["title"]), encoding="utf-8")
        (mn / "subtitles" / f'{m["movienet_id"]}.srt').write_text(
            _srt_text(m["title"]), encoding="utf-8")
    pd_rows = [{"movie_id": m["movienet_id"], "title": m["title"], "year": m["year"],
                "imdb_id": f'tt{m["imdb_id"]:07d}'} for m in SYNTHETIC_MOVIES]
    import pandas as pd
    pd.DataFrame(pd_rows).to_csv(mn / "movies.csv", index=False)

    # MovieLens 侧（movies.csv 镜像真实三列格式，年份由 load_movies 从标题解析）
    movies_rows = [{"movieId": m["movie_id"],
                    "title": f'{m["title"]} ({m["year"]})',
                    "genres": "|".join(m["genres"])}
                   for m in SYNTHETIC_MOVIES]
    pd.DataFrame(movies_rows).to_csv(ml / "movies.csv", index=False)
    pd.DataFrame([{"movieId": m["movie_id"], "imdbId": m["imdb_id"], "tmdbId": m["movie_id"]}
                  for m in SYNTHETIC_MOVIES]).to_csv(ml / "links.csv", index=False)
    pd.DataFrame([{"tagId": t["tag_id"], "tag": t["tag"]}
                  for t in SYNTHETIC_TAGS]).to_csv(ml / "genome-tags.csv", index=False)
    score_rows = [{"movieId": m["movie_id"], "tagId": t["tag_id"],
                   "relevance": round(_truth(m["title"], t["tag"]), 4)}
                  for m in SYNTHETIC_MOVIES for t in SYNTHETIC_TAGS]
    pd.DataFrame(score_rows).to_csv(ml / "genome-scores.csv", index=False)
    # 用户标签：电影 1-3 各 2 条（4-5 冷启动）
    user_rows = [{"userId": u, "movieId": mid, "tag": "fun", "timestamp": 1700000000}
                 for mid in (1, 2, 3) for u in (1, 2)]
    pd.DataFrame(user_rows).to_csv(ml / "tags.csv", index=False)

    # processed：tag_groups.json（合成分组已知，直接生成）
    processed = out_root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    (processed / "tag_groups.json").write_text(
        json.dumps(SYNTHETIC_TAGS, ensure_ascii=False, indent=1), encoding="utf-8")
