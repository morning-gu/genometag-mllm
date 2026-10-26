"""10 · 剂量-反应分析：逐片 title-only NDCG（先验强度）vs 内容净增量。

统计核心在 genometag/eval/per_movie.py（主检验 = full~title 斜率 vs 1，
防 Δ~title 的部分-整体机械负相关）；本脚本只做读取、汇总与绘图。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# 仓库根加入 sys.path，使脚本在未安装包的情况下也能直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from genometag.config import load_config, paths_for  # noqa: E402
from genometag.eval.per_movie import build_delta_table, dose_stats, per_movie_ndcg  # noqa: E402
from genometag.tags.grouping import load_tag_groups  # noqa: E402

# dataviz 默认分类色板（light）：slot1 蓝=新片人口，slot2 橙=老片对照
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e8e7e4"

_CJK_FONTS = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
              "Source Han Sans SC", "WenQuanYi Zen Hei"]


def _cjk_font_available() -> bool:
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CJK_FONTS:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


def _per_movie(paths, exp: str, groups: pd.DataFrame,
               top_k: int, threshold: float) -> pd.DataFrame:
    ann = pd.read_parquet(paths.root / "results" / exp / "annotations.parquet")
    truth = pd.read_parquet(paths.genome_scores)
    return per_movie_ndcg(ann, truth, groups, top_k, threshold)


def _plot(populations: list[tuple[pd.DataFrame, str, str]],
          fit: tuple[float, float, float, float], out_png: Path) -> None:
    """fit = (slope_delta, intercept_delta, x_min, x_max)（新片人口的 Δ~title 拟合线）。"""
    zh = _cjk_font_available()
    xlabel = "title-only NDCG@k（先验强度）" if zh else "title-only NDCG@k (prior strength)"
    ylabel = "Δ NDCG@k（full − title-only）" if zh else "Δ NDCG@k (full − title-only)"
    fit_label = "线性拟合（新片）" if zh else "Linear fit (new films)"
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.axhline(0, color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=2)

    for d, color, label in populations:
        ax.scatter(d["title_prior"], d["delta"], s=30, color=color, alpha=0.65,
                   edgecolors=SURFACE, linewidths=0.8, label=label, zorder=3)
    b, c, x0, x1 = fit
    ax.plot([x0, x1], [b * x0 + c, b * x1 + c], color=INK, lw=2, zorder=4,
            label=fit_label)

    ax.set_xlabel(xlabel, color=INK, fontsize=10)
    ax.set_ylabel(ylabel, color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, facecolor=SURFACE)
    plt.close(fig)


def main(config_path: str, full_exp: str, title_exp: str,
         pooled_old: str | None = None, n_boot: int = 10000,
         seed: int = 42) -> int:
    config = load_config(config_path)
    paths = paths_for(config)
    groups = load_tag_groups(paths.tag_groups)
    proto = config.protocol

    d_new = build_delta_table(
        _per_movie(paths, full_exp, groups, proto.top_k, proto.relevance_threshold),
        _per_movie(paths, title_exp, groups, proto.top_k, proto.relevance_threshold))
    pops = [(d_new, BLUE, f"{full_exp} − {title_exp}")]
    summary = {"newfilm": dose_stats(d_new, n_boot=n_boot, seed=seed)}

    if pooled_old:
        old_full, old_title = (s.strip() for s in pooled_old.split(","))
        d_old = build_delta_table(
            _per_movie(paths, old_full, groups, proto.top_k, proto.relevance_threshold),
            _per_movie(paths, old_title, groups, proto.top_k, proto.relevance_threshold))
        pops.append((d_old, ORANGE, f"{old_full} − {old_title}"))
        summary["old"] = dose_stats(d_old, n_boot=n_boot, seed=seed)

    out_dir = paths.root / "results" / full_exp / "dose_response"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = [df.assign(population=label) for df, _, label in pops]
    pd.concat(frames, ignore_index=True)[["movie_id", "title_prior", "full",
                                          "delta", "population"]].to_csv(
        out_dir / "delta_per_movie.csv", index=False)
    (out_dir / "dose_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    s = summary["newfilm"]
    b_full = s["slope"]
    b_delta = b_full - 1.0
    intercept = s["mean_delta"] - b_delta * d_new["title_prior"].mean()
    _plot(pops, (b_delta, intercept,
                 float(d_new["title_prior"].min()), float(d_new["title_prior"].max())),
          out_dir / "dose_response.png")

    print(f"[10] 剂量-反应 → {out_dir}")
    print(f"[10] 新片 n={s['n']}  meanΔ={s['mean_delta']:.4f} "
          f"[{s['ci_low']:.4f},{s['ci_high']:.4f}] p(Δ≤0)={s['p_delta']:.4f}")
    print(f"[10] 主检验 full~title 斜率 b={s['slope']:.4f} "
          f"[{s['slope_ci_low']:.4f},{s['slope_ci_high']:.4f}] "
          f"p(b=1)={s['slope_p_vs_1']:.4f}（冗余零假设）")
    print("[10] 注意：Spearman ρ(Δ,title) 仅描述——Δ 含 title 项，存在部分-整体机械负相关，不作推断")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--full", required=True, help="内容臂 exp_name（如 newfilm-full）")
    parser.add_argument("--title", required=True, help="title-only 臂 exp_name")
    parser.add_argument("--pooled-old", default=None,
                        help="可选 '老片full,老片title'（如 v3-full,v3-title-only）并入对照人口")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    raise SystemExit(main(args.config, args.full, args.title,
                          args.pooled_old, args.n_boot, args.seed))
