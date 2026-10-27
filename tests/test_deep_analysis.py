"""深化分析统计核心：win-rate（精确二项 + Wilson CI）、类型聚合、头部案例。"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genometag.eval.deep_analysis import (  # noqa: E402
    genre_aggregate, per_type_deltas, top_movers, win_rate)


def test_win_rate_exact_binomial_with_ties():
    d = pd.Series([1, 1, 1, 1, 1, 1, 1, -1, -1, 0.0])
    w = win_rate(d)
    assert w["n"] == 10 and w["n_win"] == 7 and w["n_loss"] == 2 and w["n_tie"] == 1
    assert w["rate"] == pytest.approx(7 / 9)
    # X~Binom(9,0.5)：P(X>=7)=P(X<=2)=46/512 → 双侧 p=92/512
    assert w["p"] == pytest.approx(92 / 512)
    assert w["ci_low"] < w["rate"] < w["ci_high"]


def test_win_rate_degenerate():
    assert win_rate(pd.Series([0.0, 0.0]))["rate"] is None or math.isnan(
        win_rate(pd.Series([0.0, 0.0]))["rate"])
    w9 = win_rate(pd.Series([1.0] * 9))
    assert w9["rate"] == pytest.approx(1.0) and w9["ci_low"] > 0.7
    assert math.isnan(win_rate(pd.Series(dtype=float))["rate"])


def _meta() -> pd.DataFrame:
    return pd.DataFrame({
        "movie_id": [1, 2, 3, 4],
        "title": ["a", "b", "c", "d"],
        "year": [2015, 2016, 1990, 2015],
        "genres": [["Action", "Sci-Fi"], ["Action"], ["Drama"], ["Sci-Fi"]],
    })


def _deltas() -> pd.DataFrame:
    return pd.DataFrame({
        "movie_id": [1, 2, 3, 4],
        "title_prior": [0.5, 0.6, 0.7, 0.8],
        "full": [0.6, 0.5, 0.6, 0.7],
        "delta": [0.1, -0.1, -0.1, -0.1],
    })


def test_genre_aggregate():
    g = genre_aggregate(_deltas(), _meta(), min_n=2)
    assert list(g.columns) == ["genre", "n", "mean_delta", "win_rate"]
    row = g[g["genre"] == "Action"].iloc[0]
    assert row["n"] == 2 and row["mean_delta"] == pytest.approx(0.0)
    assert row["win_rate"] == pytest.approx(0.5)
    sci = g[g["genre"] == "Sci-Fi"].iloc[0]
    assert sci["n"] == 2 and sci["mean_delta"] == pytest.approx(0.0)
    assert g["mean_delta"].is_monotonic_decreasing          # 降序
    assert "Drama" not in set(g["genre"])                    # min_n=2 过滤


def test_top_movers():
    m = top_movers(_deltas(), _meta(), k=2)
    assert list(m["gainers"]["movie_id"]) == [1]
    assert list(m["losers"]["movie_id"]) == [2, 3]           # 并列按 movie_id 稳定
    assert set(["title", "year", "genres"]) <= set(m["gainers"].columns)
    assert m["gainers"]["delta"].is_monotonic_decreasing


def test_per_type_deltas():
    pm_full = pd.DataFrame({"movie_id": [1, 2],
                            "ndcg_all": [0.6, 0.5],
                            "ndcg_content": [0.7, 0.4],
                            "ndcg_preference": [0.5, 0.6]})
    pm_title = pd.DataFrame({"movie_id": [1, 2],
                             "ndcg_all": [0.5, 0.6],
                             "ndcg_content": [0.6, 0.5],
                             "ndcg_preference": [0.4, 0.7]})
    d = per_type_deltas(pm_full, pm_title)
    assert list(d.columns) == ["movie_id", "delta_all", "delta_content",
                               "delta_preference"]
    assert d["delta_all"].tolist() == pytest.approx([0.1, -0.1])
    assert d["delta_content"].tolist() == pytest.approx([0.1, -0.1])
    assert d["delta_preference"].tolist() == pytest.approx([0.1, -0.1])
