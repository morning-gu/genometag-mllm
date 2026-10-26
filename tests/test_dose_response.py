"""逐片 NDCG + 剂量-反应统计：与官方报告同语义、先平均再算指标、主检验斜率 vs 1。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genometag.config import load_config  # noqa: E402
from genometag.eval.per_movie import build_delta_table, dose_stats, per_movie_ndcg  # noqa: E402
from genometag.eval.report import metrics_table  # noqa: E402


def _pred(rows: list[tuple[int, int, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["movie_id", "tag_id", "run_id", "score"])


def _truth(rows: list[tuple[int, int, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["movie_id", "tag_id", "relevance"])


def _groups() -> pd.DataFrame:
    return pd.DataFrame({"tag_id": [10, 11, 12, 13],
                         "tag_type": ["content", "content", "preference", "preference"]})


CFG = load_config("configs/smoke.yaml").protocol


def test_per_movie_matches_report():
    pred = _pred([(1, 10, "r1", 0.8), (1, 11, "r1", 0.5), (1, 12, "r1", 0.3), (1, 13, "r1", 0.1),
                  (1, 10, "r2", 0.7), (1, 11, "r2", 0.6), (1, 12, "r2", 0.3), (1, 13, "r2", 0.05),
                  (2, 10, "r1", 0.4), (2, 11, "r1", 0.6), (2, 12, "r1", 0.7), (2, 13, "r1", 0.2),
                  (2, 10, "r2", 0.4), (2, 11, "r2", 0.5), (2, 12, "r2", 0.9), (2, 13, "r2", 0.1)])
    truth = _truth([(1, t, r) for t, r in [(10, 0.9), (11, 0.6), (12, 0.2), (13, 0.0)]]
                   + [(2, t, r) for t, r in [(10, 0.3), (11, 0.7), (12, 0.8), (13, 0.1)]])
    pm = per_movie_ndcg(pred, truth, _groups(), CFG.top_k, CFG.relevance_threshold)
    assert list(pm.columns) == ["movie_id", "ndcg_all", "ndcg_content", "ndcg_preference"]
    pred_mean = pred.groupby(["movie_id", "tag_id"], as_index=False)["score"].mean()
    expected = metrics_table({"m": pred_mean}, truth,
                             pd.DataFrame({"movie_id": []}), CFG, _groups())
    assert pm["ndcg_all"].mean() == pytest.approx(expected["ndcg@k"].iloc[0])


def test_per_movie_matches_report_tied_scores():
    """并列分数（LLM 输出 0.1 网格常态）：逐片口径必须与 06 官方评估逐位一致——
    官方路径喂给 ndcg_at_k 的是 tag_id 升序向量（pivot 列序），非分数降序。"""
    rng = np.random.default_rng(0)
    tags = list(range(10, 40))                      # 30 个标签
    groups = pd.DataFrame({"tag_id": tags,
                           "tag_type": ["content" if t % 2 else "preference" for t in tags]})
    rows_p, rows_t = [], []
    for movie in range(1, 21):
        for t in tags:
            score = float(rng.choice([0.0, 0.05, 0.1, 0.15]))   # 重度并列
            for r in range(2):
                rows_p.append((movie, t, f"r{r}", score))
            rows_t.append((movie, t, float(rng.uniform(0, 1))))
    pred, truth = _pred(rows_p), _truth(rows_t)
    pm = per_movie_ndcg(pred, truth, groups, CFG.top_k, CFG.relevance_threshold)
    pred_mean = pred.groupby(["movie_id", "tag_id"], as_index=False)["score"].mean()
    expected = metrics_table({"m": pred_mean}, truth,
                             pd.DataFrame({"movie_id": []}), CFG, groups)
    assert pm["ndcg_all"].mean() == pytest.approx(expected["ndcg@k"].iloc[0])


def test_all_nan_runs_dropped_like_report():
    """某标签全部轮次失败（score 全 NaN）：必须与 06 一样先剔除，
    不得让 NaN 伪装成最低分进入 top-k 污染该片指标。"""
    rows = [(1, 10, "r1", 0.5), (1, 10, "r2", 0.5),            # 有效标签
            (1, 11, "r1", np.nan), (1, 11, "r2", np.nan)]      # 全失败标签
    pred, truth = _pred(rows), _truth([(1, 10, 0.0), (1, 11, 1.0)])
    pm = per_movie_ndcg(pred, truth, _groups(), top_k=2, threshold=0.5)
    # 06 语义：NaN 行剔除 → 只剩 tag10 → dcg=0（rel 0.0）、idcg=min(2,1)=1 → 0.0
    assert pm.loc[0, "ndcg_all"] == pytest.approx(0.0)


def test_nan_truth_tag_dropped_movie_kept():
    """movie 1 的 tag 11 无真值 → 剔除该标签，movie 1 仍有分数（content 上算）。"""
    pred = _pred([(1, 10, "r1", 0.8), (1, 11, "r1", 0.2)])
    truth = _truth([(1, 10, 0.9)])          # tag 11 真值缺失
    pm = per_movie_ndcg(pred, truth, _groups(), top_k=1, threshold=0.5)
    assert list(pm["movie_id"]) == [1]
    assert pm.loc[0, "ndcg_content"] == pytest.approx(1.0)
    assert np.isnan(pm.loc[0, "ndcg_preference"])
    assert pm.loc[0, "ndcg_all"] == pytest.approx(1.0)


def test_run_mean_first():
    """3 轮分数先平均再排序（run-mean 后 tag10 第一）；逐轮平均会得到 0.667。"""
    pred = _pred([(1, 10, "r1", 0.9), (1, 11, "r1", 0.1),
                  (1, 10, "r2", 0.1), (1, 11, "r2", 0.9),
                  (1, 10, "r3", 0.9), (1, 11, "r3", 0.1)])
    truth = _truth([(1, 10, 1.0), (1, 11, 0.0)])
    pm = per_movie_ndcg(pred, truth, _groups(), top_k=1, threshold=0.5)
    assert pm.loc[0, "ndcg_all"] == pytest.approx(1.0)
    assert pm.loc[0, "ndcg_all"] != pytest.approx(2 / 3)


def test_build_delta_table():
    pm_full = pd.DataFrame({"movie_id": [1, 2], "ndcg_all": [0.5, 0.7]})
    pm_title = pd.DataFrame({"movie_id": [1, 2], "ndcg_all": [0.6, 0.6]})
    d = build_delta_table(pm_full, pm_title)
    assert list(d.columns) == ["movie_id", "title_prior", "full", "delta"]
    assert list(d["movie_id"]) == [1, 2]
    assert d["delta"].tolist() == pytest.approx([-0.1, 0.1])


def test_slope_recovers():
    """full = 0.9 − 0.3·title + 小噪声：斜率 ≈ −0.3（<1，冗余零假设被拒），
    分箱 Δ 随先验升而降。"""
    rng = np.random.default_rng(42)
    n = 120
    title = rng.uniform(0.2, 0.9, n)
    full = 0.9 - 0.3 * title + rng.normal(0, 0.02, n)
    df = pd.DataFrame({"movie_id": np.arange(n), "title_prior": title,
                       "full": full, "delta": full - title})
    s = dose_stats(df, n_boot=2000, seed=42)
    assert s["n"] == n
    assert {"mean_delta", "ci_low", "ci_high", "p_delta", "slope",
            "slope_ci_low", "slope_ci_high", "slope_p_vs_1",
            "spearman_rho", "bins"} <= set(s)
    assert s["slope"] < 1
    assert s["slope_p_vs_1"] < 0.001
    assert s["p_delta"] < 0.01
    bins = s["bins"]
    assert [b["quartile"] for b in bins] == ["Q1", "Q2", "Q3", "Q4"]
    means = [b["mean_delta"] for b in bins]
    assert all(a > b for a, b in zip(means, means[1:]))


def test_dose_stats_tied_prior_bins():
    """逐片 NDCG 大量并列（真实数据常态）：qcut 重复边界不得崩溃；
    唯一值少于 4 时按实际档位输出 Q1..Qk。"""
    n = 40
    x = np.repeat([0.2, 0.4, 0.6, 0.9], 10)
    full = 0.9 - 0.3 * x
    df = pd.DataFrame({"movie_id": np.arange(n), "title_prior": x,
                       "full": full, "delta": full - x})
    s = dose_stats(df, n_boot=200, seed=42)
    assert [b["quartile"] for b in s["bins"]] == ["Q1", "Q2", "Q3", "Q4"]

    x2 = np.repeat([0.3, 0.8], 10)
    full2 = 0.9 - 0.3 * x2
    df2 = pd.DataFrame({"movie_id": np.arange(20), "title_prior": x2,
                        "full": full2, "delta": full2 - x2})
    s2 = dose_stats(df2, n_boot=200, seed=42)
    assert [b["quartile"] for b in s2["bins"]] == ["Q1", "Q2"]


def test_dose_stats_bootstrap_slope_finite():
    """并列先验下零方差自抽样 → NaN/inf 斜率必须剔除：
    p 值落在 [0,1]，CI 有限；x 恒定时 slope/p 为 NaN 而非崩溃。"""
    x = np.repeat([0.2, 0.9], 50)
    full = 0.9 - 0.3 * x + np.repeat([0.01, -0.01], 50)
    df = pd.DataFrame({"movie_id": np.arange(100), "title_prior": x,
                       "full": full, "delta": full - x})
    s = dose_stats(df, n_boot=500, seed=42)
    assert 0.0 <= s["slope_p_vs_1"] <= 1.0
    assert np.isfinite(s["slope_ci_low"]) and np.isfinite(s["slope_ci_high"])

    xc = np.full(20, 0.5)
    dfc = pd.DataFrame({"movie_id": np.arange(20), "title_prior": xc,
                        "full": 0.7 + np.arange(20) * 0.001, "delta": 0.2})
    sc = dose_stats(dfc, n_boot=100, seed=42)
    assert np.isnan(sc["slope_p_vs_1"])
    assert np.isnan(sc["slope_ci_low"])
