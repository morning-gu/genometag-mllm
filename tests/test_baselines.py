"""基线：确定性、防泄露、手算值。"""
import importlib
import importlib.util

import pandas as pd
import pytest

from genometag.baselines.genre_baseline import genre_baseline
from genometag.baselines.mean_baseline import mean_baseline
from genometag.baselines.random_baseline import random_baseline


def test_random_deterministic_and_bounded():
    df1 = random_baseline([1, 2], [10, 11], seed=7)
    df2 = random_baseline([1, 2], [10, 11], seed=7)
    pd.testing.assert_frame_equal(df1, df2)
    assert len(df1) == 4                     # 2 电影 × 2 tag
    assert df1["score"].between(0, 1).all()
    assert list(df1.columns) == ["movie_id", "tag_id", "score", "method"]


def test_mean_baseline_hand_computed():
    train = pd.DataFrame({"movie_id": [1, 2, 1, 2], "tag_id": [10, 10, 11, 11],
                          "relevance": [0.8, 0.4, 0.2, 0.2]})
    df = mean_baseline(train, movie_ids=[5])
    # tag 10 → 0.6；tag 11 → 0.2
    got = dict(zip(df["tag_id"], df["score"]))
    assert got[10] == pytest.approx(0.6)
    assert got[11] == pytest.approx(0.2)
    assert (df["method"] == "mean").all()


def test_mean_baseline_leakage_guard():
    """拟合集含测试电影 → 抛错（防泄露自检）。"""
    train = pd.DataFrame({"movie_id": [1, 2], "tag_id": [10, 10], "relevance": [0.8, 0.4]})
    with pytest.raises(ValueError, match="泄露|重叠"):
        mean_baseline(train, movie_ids=[1], test_ids={1})


def test_genre_baseline_hand_computed():
    dataset = pd.DataFrame({"movie_id": [1, 2, 3, 4, 5, 6],
                            "genres": [["A"], ["A", "B"], ["B"], ["A"], ["A", "B"], []]})
    train = pd.DataFrame({  # 电影 1-3 为训练参考集
        "movie_id": [1, 2, 3], "tag_id": [10, 10, 10], "relevance": [0.8, 0.6, 0.2]})
    df = genre_baseline(train, dataset, movie_ids=[4, 5, 6])
    got = dict(zip(df["movie_id"], df["score"]))
    # v_A = mean(0.8, 0.6) = 0.7; v_B = mean(0.6, 0.2) = 0.4; 全局 = 1.6/3
    assert got[4] == pytest.approx(0.7)                 # [A]
    assert got[5] == pytest.approx(0.55)                # [A, B] → (0.7+0.4)/2
    assert got[6] == pytest.approx(1.6 / 3)             # 无 genre → 全局均值


@pytest.mark.skipif(importlib.util.find_spec("torch") is not None,
                    reason="torch 存在时（无论能否加载）'缺 torch 优雅跳过'路径无法测试")
def test_bert_branch_skips_gracefully_without_torch(tmp_path):
    """回归：懒导入下模块自身可导入，缺 torch 在调用点抛 ImportError——
    05 的 bert 分支须捕获并优雅跳过（report["bert"]=="skipped"），
    其余基线照常产出，整个 run_baselines 不抛异常。"""
    from genometag.config import AppConfig, Paths
    from genometag.data.align import build_dataset
    from genometag.data.sample import make_synthetic_dataset

    make_synthetic_dataset(tmp_path)
    paths = Paths(  # 同 tests/test_pipeline.py 的 _paths_for 模式
        root=tmp_path,
        movienet=tmp_path / "raw" / "movienet",
        movielens=tmp_path / "raw" / "movielens",
        interim=tmp_path / "interim", processed=tmp_path / "processed",
        cache=tmp_path / ".cache", results=tmp_path / "results" / "smoke",
        tag_groups=tmp_path / "processed" / "tag_groups.json",
        dataset=tmp_path / "processed" / "dataset.parquet",
        genome_scores=tmp_path / "processed" / "genome_scores.parquet",
        genome_tags=tmp_path / "processed" / "genome_tags.parquet",
        aligned=tmp_path / "interim" / "aligned.parquet")
    build_dataset(paths)                      # 产出 dataset/genome_scores
    cfg = AppConfig()
    cfg.baselines.run = ["random", "mean", "genre", "bert"]  # 默认已含 bert，显式声明意图

    script_05 = importlib.import_module("scripts.05_run_baselines")
    df, report = script_05.run_baselines(cfg, paths)  # 本 venv 无 torch → 真实走跳过路径

    assert report["bert"] == "skipped"
    assert set(df["method"]) == {"random", "mean", "genre"}   # 其余基线照常产出
    assert len(df) > 0
