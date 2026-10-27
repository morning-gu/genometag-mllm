"""深化分析统计核心：win-rate（精确二项检验 + Wilson CI）、genre 聚合、头部案例。

约定：Δ = full − title-only（逐片 NDCG@k，见 per_movie.py）；并列（Δ=0）不计入
win-rate 分母；精确二项零假设为 win=loss 各 0.5。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

_Z95 = 1.959963984540054


def _wilson(wins: int, m: int) -> tuple[float, float]:
    p = wins / m
    denom = 1.0 + _Z95 ** 2 / m
    center = (p + _Z95 ** 2 / (2 * m)) / denom
    half = _Z95 * math.sqrt(p * (1 - p) / m + _Z95 ** 2 / (4 * m * m)) / denom
    return center - half, center + half


def win_rate(delta: pd.Series) -> dict:
    d = pd.Series(delta).dropna()
    n_win = int((d > 0).sum())
    n_loss = int((d < 0).sum())
    n_tie = int((d == 0).sum())
    m = n_win + n_loss
    out = {"n": int(len(d)), "n_win": n_win, "n_loss": n_loss, "n_tie": n_tie}
    if m == 0:
        return {**out, "rate": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "p": float("nan")}
    lo, hi = _wilson(n_win, m)
    tail_hi = max(n_win, n_loss)
    tail = sum(math.comb(m, i) for i in range(tail_hi, m + 1)) / 2 ** m
    return {**out, "rate": n_win / m, "ci_low": lo, "ci_high": hi,
            "p": float(min(1.0, 2 * tail))}


def genre_aggregate(delta_df: pd.DataFrame, meta: pd.DataFrame,
                    min_n: int = 3) -> pd.DataFrame:
    g = delta_df.merge(meta[["movie_id", "genres"]], on="movie_id", how="inner")
    g = g.explode("genres").rename(columns={"genres": "genre"})
    rows = []
    for genre, sub in g.groupby("genre"):
        w = win_rate(sub["delta"])
        rows.append({"genre": genre, "n": int(len(sub)),
                     "mean_delta": float(sub["delta"].mean()),
                     "win_rate": w["rate"]})
    out = pd.DataFrame(rows)
    return out[out["n"] >= min_n].sort_values(
        "mean_delta", ascending=False).reset_index(drop=True)


def top_movers(delta_df: pd.DataFrame, meta: pd.DataFrame,
               k: int = 10) -> dict[str, pd.DataFrame]:
    cols = ["movie_id", "title", "year", "genres",
            "title_prior", "full", "delta"]
    j = delta_df.merge(meta[["movie_id", "title", "year", "genres"]],
                       on="movie_id", how="inner")[cols]
    kk = min(k, len(j))
    gainers = j[j["delta"] > 0].sort_values(
        ["delta", "movie_id"], ascending=[False, True]).head(kk)
    losers = j[j["delta"] < 0].sort_values(["delta", "movie_id"]).head(kk)
    return {"gainers": gainers.reset_index(drop=True),
            "losers": losers.reset_index(drop=True)}


def per_type_deltas(pm_full: pd.DataFrame,
                    pm_title: pd.DataFrame) -> pd.DataFrame:
    j = pm_full.merge(pm_title, on="movie_id", suffixes=("_f", "_t"))
    return pd.DataFrame({
        "movie_id": j["movie_id"],
        "delta_all": j["ndcg_all_f"] - j["ndcg_all_t"],
        "delta_content": j["ndcg_content_f"] - j["ndcg_content_t"],
        "delta_preference": j["ndcg_preference_f"] - j["ndcg_preference_t"],
    })
