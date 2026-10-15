"""评估报告：指标表（CSV+markdown）、模态×标签类别热力图。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from genometag.config import ProtocolConfig
from genometag.eval.metrics import (
    consistency_std, icc_2_1, mae, ndcg_at_k, precision_at_k, rmse,
)
from genometag.eval.protocol import cold_start_subset, tag_type_breakdown, temporal_split


def _wide(df: pd.DataFrame) -> pd.DataFrame:
    """(movie_id, tag_id, score) → movie × tag 宽矩阵。"""
    return df.pivot(index="movie_id", columns="tag_id", values="score")


def _evaluate_method(pred: pd.DataFrame, truth: pd.DataFrame,
                     cfg: ProtocolConfig) -> dict[str, float]:
    # pred 的 score 先重命名为 score_p：truth 侧列名为 relevance，合并无列冲突时
    # pandas 不会追加后缀，直接引用 score_p 会 KeyError
    joined = pred.rename(columns={"score": "score_p"}).merge(
        truth, on=["movie_id", "tag_id"])
    p, t = joined["score_p"].to_numpy(), joined["relevance"].to_numpy()
    pw, tw = _wide(joined.rename(columns={"score_p": "score"})), _wide(
        joined.rename(columns={"relevance": "score"}))
    pk, nk = [], []
    for movie in pw.index:
        if movie in tw.index:
            pk.append(precision_at_k(pw.loc[movie].to_numpy(), tw.loc[movie].to_numpy(),
                                     cfg.top_k, cfg.relevance_threshold))
            nk.append(ndcg_at_k(pw.loc[movie].to_numpy(), tw.loc[movie].to_numpy(),
                                cfg.top_k, cfg.relevance_threshold))
    return {"rmse": rmse(p, t), "mae": mae(p, t),
            "precision@k": float(np.mean(pk)) if pk else 0.0,
            "ndcg@k": float(np.mean(nk)) if nk else 0.0}


def metrics_table(predictions: dict[str, pd.DataFrame], truth: pd.DataFrame,
                  dataset: pd.DataFrame, cfg: ProtocolConfig,
                  tag_groups: pd.DataFrame | None = None,
                  movie_ids: list[int] | None = None) -> pd.DataFrame:
    if movie_ids is not None:
        keep = set(movie_ids)
        truth = truth[truth["movie_id"].isin(keep)]
    rows = []
    for method, pred in predictions.items():
        if movie_ids is not None:
            pred = pred[pred["movie_id"].isin(set(movie_ids))]
        rows.append({"method": method, **_evaluate_method(pred, truth, cfg)})
    return pd.DataFrame(rows)


def modality_heatmap(rows: pd.DataFrame, out_png: Path) -> None:
    """模态 × 标签类别的 NDCG 热力图。实现前已按 dataviz 技能规范设计：

    - 幅值 → 单色顺序色带（Blues），vmin=0/vmax=1 定标，跨实验可比；
    - 每格直接标注数值（relief 通道，避免仅靠颜色编码），文字黑/白按填充亮度选择；
    - colorbar 作为标尺图例，metrics.csv 承担表格视图兜底。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pivot = rows.pivot(index="modality_set", columns="tag_type", values="ndcg")
    # 中文轴标签需要 CJK 字体（rc_context 作用域内生效，不污染全局状态）
    with matplotlib.rc_context({
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                                "WenQuanYi Micro Hei", "PingFang SC"]
            + matplotlib.rcParams["font.sans-serif"],
            "axes.unicode_minus": False}):
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(pivot.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)), pivot.columns)
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        ax.set_xlabel("标签类别")
        ax.set_ylabel("模态组合")
        ax.set_title("NDCG：模态 × 标签类别")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.iloc[i, j]
                if pd.isna(v):
                    continue
                # dataviz 规范：格内文字按填充亮度选黑/白，保证对比度可读
                rgba = im.cmap(im.norm(v))[:3]
                lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="black" if lum > 0.45 else "white")
        fig.colorbar(im, ax=ax, label="NDCG")
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=150)
        plt.close(fig)


def write_report(exp_dirs: list[Path], baseline_dir: Path | None, out_dir: Path,
                 cfg: ProtocolConfig, dataset: pd.DataFrame,
                 truth: pd.DataFrame, tag_groups: pd.DataFrame | None,
                 user_tags: pd.DataFrame | None) -> Path:
    """汇总多实验 + 基线 → evaluation/ 目录（metrics.csv、report.md、
    consistency.json、heatmap.png）。

    metrics.csv 的 subset 视图：
    - 全部 / 时序训练(<year) / 时序测试(≥year)：推理（训练集）vs 回忆（测试集）
      的对比是论文核心论证（论文 §"推理vs回忆"）；
    - 冷启动（有 user_tags 时）；
    - tag:<类型>（有 tag_groups 时）：内容型 vs 偏好型标签的逐类型对比，
      单实验也产出（不依赖热力图路径）。
    """
    predictions: dict[str, pd.DataFrame] = {}
    for exp in exp_dirs:
        ann = pd.read_parquet(exp / "annotations.parquet")
        agg = (ann.dropna(subset=["score"])
               .groupby(["movie_id", "tag_id"], as_index=False)["score"].mean())
        modality = ann["modality_set"].iloc[0]
        predictions[f"MLLM[{modality}]"] = agg
    if baseline_dir and (baseline_dir / "baselines.parquet").exists():
        base = pd.read_parquet(baseline_dir / "baselines.parquet")
        for method, df in base.groupby("method"):
            predictions[str(method)] = df[["movie_id", "tag_id", "score"]]

    out_dir.mkdir(parents=True, exist_ok=True)
    # 评估电影集合 = 第一个实验实际标注到的电影（基线在同一集合上对比才公平）
    first_ann = pd.read_parquet(exp_dirs[0] / "annotations.parquet")
    movie_ids = sorted(first_ann["movie_id"].unique().tolist())

    sections: list[pd.DataFrame] = [metrics_table(
        predictions, truth, dataset, cfg, tag_groups, movie_ids).assign(subset="全部")]
    # 时序训练/测试子集：train 上近记忆式复述、test 上才见真泛化
    train_ids, test_ids = temporal_split(dataset, cfg.temporal_split_year)
    sections.append(metrics_table(predictions, truth, dataset, cfg, tag_groups,
                                  [m for m in movie_ids if m in set(train_ids)]
                                  ).assign(subset=f"时序训练(<{cfg.temporal_split_year})"))
    sections.append(metrics_table(predictions, truth, dataset, cfg, tag_groups,
                                  [m for m in movie_ids if m in set(test_ids)]
                                  ).assign(subset=f"时序测试(≥{cfg.temporal_split_year})"))
    # 冷启动子集
    if user_tags is not None:
        cold = cold_start_subset(dataset, user_tags, cfg.cold_start_max_tags)
        sections.append(metrics_table(predictions, truth, dataset, cfg, tag_groups,
                                      [m for m in movie_ids if m in set(cold)]
                                      ).assign(subset="冷启动"))
    # 逐 tag 类型子集：内容型 vs 偏好型（单实验也产出，不依赖热力图路径）
    if tag_groups is not None:
        for tag_type, tids in tag_type_breakdown(tag_groups).items():
            keep_tags = set(tids)
            sub_preds = {name: p[p["tag_id"].isin(keep_tags)]
                         for name, p in predictions.items()}
            sections.append(metrics_table(
                sub_preds, truth, dataset, cfg, tag_groups, movie_ids
            ).assign(subset=f"tag:{tag_type}"))

    table = pd.concat(sections, ignore_index=True)
    table.to_csv(out_dir / "metrics.csv", index=False)
    md = ["# 评估报告\n", table.to_markdown(index=False), ""]

    # 一致性分析（spec §5.1）：轮次 > 1 时计算 ICC(2,1) 与平均轮次标准差
    n_runs = first_ann["run_id"].nunique()
    if n_runs > 1:
        wide_runs = first_ann.dropna(subset=["score"]).pivot_table(
            index=["movie_id", "tag_id"], columns="run_id", values="score")
        if wide_runs.shape[1] > 1 and len(wide_runs) > 1:
            matrix = wide_runs.to_numpy()
            consistency = {
                "n_runs": int(n_runs),
                "icc_2_1": icc_2_1(matrix),
                "mean_std": float(np.mean(consistency_std(matrix))),
            }
            (out_dir / "consistency.json").write_text(
                json.dumps(consistency, ensure_ascii=False, indent=1), encoding="utf-8")
            md.append(f"\n## 一致性（{n_runs} 轮调用）\n"
                      f"- ICC(2,1): {consistency['icc_2_1']:.3f}\n"
                      f"- 平均轮次标准差: {consistency['mean_std']:.3f}\n")
    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")

    # 热力图：多模态实验 + tag 类型信息时生成
    if len(exp_dirs) > 1 and tag_groups is not None:
        rows = []
        for exp in exp_dirs:
            ann = pd.read_parquet(exp / "annotations.parquet")
            agg = (ann.dropna(subset=["score"])
                   .groupby(["movie_id", "tag_id"], as_index=False)["score"].mean())
            for tag_type, tids in tag_type_breakdown(tag_groups).items():
                sub = agg[agg["tag_id"].isin(tids)]
                m = metrics_table({"m": sub}, truth, dataset, cfg, None, movie_ids)
                rows.append({"modality_set": ann["modality_set"].iloc[0],
                             "tag_type": tag_type, "ndcg": float(m["ndcg@k"].iloc[0])})
        modality_heatmap(pd.DataFrame(rows), out_dir / "heatmap.png")
    return out_dir
