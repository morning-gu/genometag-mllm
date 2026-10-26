"""逐片 NDCG（镜像 report._evaluate_method 语义：先 run 平均、与 truth inner-join）
与剂量-反应统计（主检验 = full~title 斜率 vs 1，防部分-整体相关的机械负斜率陷阱）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from genometag.eval.metrics import ndcg_at_k


def per_movie_ndcg(pred_df: pd.DataFrame, truth_df: pd.DataFrame,
                   tag_groups_df: pd.DataFrame,
                   top_k: int, threshold: float) -> pd.DataFrame:
    """列 movie_id, ndcg_all, ndcg_content, ndcg_preference；某类型无（有真值的）
    标签时该列为 NaN；truth 缺失的标签经 inner-join 自然剔除。"""
    pred = pred_df.dropna(subset=["score"]).groupby(
        ["movie_id", "tag_id"], as_index=False)["score"].mean()
    joined = pred.merge(truth_df, on=["movie_id", "tag_id"])
    type_of = dict(zip(tag_groups_df["tag_id"], tag_groups_df["tag_type"]))
    joined["tag_type"] = joined["tag_id"].map(type_of)
    rows = []
    for movie, g in joined.groupby("movie_id"):
        row: dict[str, float | int] = {"movie_id": int(movie)}
        for name, gg in (("ndcg_all", g),
                         ("ndcg_content", g[g["tag_type"] == "content"]),
                         ("ndcg_preference", g[g["tag_type"] == "preference"])):
            if len(gg) == 0:
                row[name] = np.nan
                continue
            # 与 report._evaluate_method 同序：pivot 列序 = tag_id 升序。
            # 按分数排序会在并列分数下与官方路径的不稳定 argsort 产生不同 tie-break
            o = gg.sort_values("tag_id")
            row[name] = ndcg_at_k(o["score"].to_numpy(), o["relevance"].to_numpy(),
                                  top_k, threshold)
        rows.append(row)
    return pd.DataFrame(rows, columns=["movie_id", "ndcg_all",
                                       "ndcg_content", "ndcg_preference"])


def build_delta_table(pm_full: pd.DataFrame, pm_title: pd.DataFrame) -> pd.DataFrame:
    d = pm_full[["movie_id", "ndcg_all"]].rename(
        columns={"ndcg_all": "full"}).merge(
        pm_title[["movie_id", "ndcg_all"]].rename(columns={"ndcg_all": "title_prior"}),
        on="movie_id", how="inner")
    d["delta"] = d["full"] - d["title_prior"]
    return d[["movie_id", "title_prior", "full", "delta"]].sort_values(
        "movie_id").reset_index(drop=True)


def _boot_slope(x: np.ndarray, y: np.ndarray, idx: np.ndarray) -> np.ndarray:
    xm = x[idx] - x[idx].mean(axis=1, keepdims=True)
    ym = y[idx] - y[idx].mean(axis=1, keepdims=True)
    num = (xm * ym).mean(axis=1)
    den = (xm ** 2).mean(axis=1)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def dose_stats(df: pd.DataFrame, n_boot: int = 10000,
               seed: int = 42) -> dict:
    """headline：mean Δ + 配对 bootstrap CI 与单侧 p；
    主检验：full~title OLS 斜率 vs 1（冗余零假设）的 bootstrap 双侧 p；
    描述项：Spearman ρ(Δ, title)（注意部分-整体相关的机械负偏，不作推断）；
    title 四分位分箱 mean Δ。"""
    x = df["title_prior"].to_numpy(dtype=float)
    y = df["full"].to_numpy(dtype=float)
    d = df["delta"].to_numpy(dtype=float)
    n = len(df)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))

    boot_mean = d[idx].mean(axis=1)
    ci_low, ci_high = np.percentile(boot_mean, [2.5, 97.5])
    p_delta = float((boot_mean <= 0).mean())

    x_const = bool(np.ptp(x) == 0) or len(x) < 2   # 先验无变异：斜率/相关/分箱均无定义
    slope = float("nan") if x_const else float(
        np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))

    rho = float("nan") if x_const else float(
        pd.Series(d).corr(pd.Series(x), method="spearman"))
    # 逐片 NDCG 并列值多：重复分位边界降级为更少档位（Q1..Qk）
    if x_const:
        k = 0
    else:
        codes = pd.qcut(x, 4, labels=False, duplicates="drop").astype(int)
        k = int(codes.max()) + 1
    bins = [{"quartile": f"Q{i + 1}", "n": int((codes == i).sum()),
             "mean_delta": float(d[codes == i].mean())}
            for i in range(k)]

    boot_b = _boot_slope(x, y, idx)
    finite = boot_b[np.isfinite(boot_b)]   # 零方差自抽样（并列先验）产生 NaN/±inf，剔除
    if len(finite) == 0:                   # x 恒定：斜率检验无定义
        sb_low = sb_high = p_b1 = float("nan")
    else:
        sb_low, sb_high = np.percentile(finite, [2.5, 97.5])
        p_b1 = float(2 * min((finite >= 1).mean(), (finite <= 1).mean()))

    return {"n": n, "mean_delta": float(d.mean()),
            "ci_low": float(ci_low), "ci_high": float(ci_high),
            "p_delta": p_delta, "slope": slope,
            "slope_ci_low": float(sb_low), "slope_ci_high": float(sb_high),
            "slope_p_vs_1": p_b1, "spearman_rho": rho, "bins": bins}
