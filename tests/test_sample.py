"""合成样本生成：结构、形状、冷启动分布、年份分割。"""
import pandas as pd

from genometag.data.sample import (
    SYNTHETIC_MOVIES, SYNTHETIC_TAGS, base_score, make_synthetic_dataset,
)


def test_constants_shape():
    assert len(SYNTHETIC_TAGS) == 24
    assert len({t["group"] for t in SYNTHETIC_TAGS}) == 4
    assert len(SYNTHETIC_MOVIES) == 5
    # tag_type 只有 content / preference 两种
    assert {t["tag_type"] for t in SYNTHETIC_TAGS} == {"content", "preference"}


def test_base_score_deterministic_and_bounded():
    assert base_score("Crimson Horizon", "neon visuals") == base_score("Crimson Horizon", "neon visuals")
    assert 0.0 <= base_score("A", "x") <= 1.0
    assert base_score("A", "x") != base_score("B", "x")  # 不同电影不同分


def test_make_synthetic_dataset_structure(tmp_path):
    make_synthetic_dataset(tmp_path)
    # MovieLens 侧
    ml = tmp_path / "raw" / "movielens"
    scores = pd.read_csv(ml / "genome-scores.csv")
    tags = pd.read_csv(ml / "genome-tags.csv")
    movies = pd.read_csv(ml / "movies.csv")
    links = pd.read_csv(ml / "links.csv")
    utags = pd.read_csv(ml / "tags.csv")
    assert len(scores) == 5 * 24
    assert len(tags) == 24
    assert len(movies) == 5
    assert len(links) == 5
    # 冷启动：仅 movie_id 1-3 有用户标签，4-5 无
    tagged = set(utags["movieId"].unique())
    assert tagged == {1, 2, 3}
    # 年份分割（2022）：4 部 < 2022，1 部 ≥ 2022（年份编码在标题内，同真实格式）
    titles = movies["title"].tolist()
    assert sum(1 for t in titles if "(2023)" in t) == 1
    assert sum(1 for t in titles if any(y in t for y in
               ("(2018)", "(2019)", "(2020)", "(2021)"))) == 4
    # MovieNet 侧
    mn = tmp_path / "raw" / "movienet"
    mn_movies = pd.read_csv(mn / "movies.csv")
    assert len(mn_movies) == 5
    frames = sorted((mn / "keyframes" / "mn01").glob("*.jpg"))
    assert len(frames) == 8  # 每部 8 帧
    assert (mn / "subtitles" / "mn01.srt").exists()
    # processed：tag_groups.json 已随样本生成
    import json
    tg = json.loads((tmp_path / "processed" / "tag_groups.json").read_text(encoding="utf-8"))
    assert len(tg) == 24
    assert {"tag_id", "tag", "group", "tag_type", "definition"} <= set(tg[0].keys())
