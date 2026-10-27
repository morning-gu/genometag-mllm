"""11 · 深化分析：win-rate（按标签类型）、genre 聚合、头部案例、逐片分布图。

统计核心在 genometag/eval/deep_analysis.py（TDD 覆盖）；本脚本做数据装配与绘图。
用法（服务器项目根）：
  .venv/bin/python scripts/11_deep_analysis.py --config configs/newfilm/full.yaml \
      --pair newfilm-full,newfilm-title,newfilm --pair v3-full,v3-title-only,old
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from genometag.config import load_config, paths_for  # noqa: E402
from genometag.eval.deep_analysis import (  # noqa: E402
    genre_aggregate, per_type_deltas, top_movers, win_rate)
from genometag.eval.per_movie import per_movie_ndcg  # noqa: E402
from genometag.tags.grouping import load_tag_groups  # noqa: E402

# dataviz 默认分类色板（light）：slot1 蓝 / slot2 橙
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e8e7e4"
_CJK_FONTS = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
              "Source Han Sans SC", "WenQuanYi Zen Hei"]


def _cjk() -> bool:
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CJK_FONTS:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


def _per_movie(paths, exp: str, groups, top_k: int, threshold: float):
    ann = pd.read_parquet(paths.root / "results" / exp / "annotations.parquet")
    truth = pd.read_parquet(paths.genome_scores)
    return per_movie_ndcg(ann, truth, groups, top_k, threshold)


def _style(ax):
    fig = ax.figure
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)


def _ecdf_plot(pairs: dict[str, pd.DataFrame], zh: bool, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=150)
    _style(ax)
    for (label, color) in (("newfilm", BLUE), ("old", ORANGE)):
        if label not in pairs:
            continue
        d = np.sort(pairs[label]["delta_all"].to_numpy())
        ax.plot(d, np.arange(1, len(d) + 1) / len(d), color=color, lw=2,
                label=label, zorder=3)
    ax.axvline(0, color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=2)
    ax.set_xlabel("Δ NDCG@k（full − title-only）" if zh else "Δ NDCG@k (full − title-only)",
                  color=INK, fontsize=10)
    ax.set_ylabel("累积比例（逐片 ECDF）" if zh else "ECDF", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_png, facecolor=SURFACE)
    plt.close(fig)


def _winrate_plot(rows: list[dict], zh: bool, out_png: Path) -> None:
    rows = [r for r in rows if np.isfinite(r["rate"])]
    fig, ax = plt.subplots(figsize=(7.2, 0.6 * len(rows) + 1.6), dpi=150)
    _style(ax)
    ys = np.arange(len(rows))[::-1]
    for y, r in zip(ys, rows):
        color = BLUE if r["population"] == "newfilm" else ORANGE
        star = "*" if r["p"] < 0.05 else ""
        ax.errorbar(r["rate"], y,
                    xerr=[[r["rate"] - r["ci_low"]], [r["ci_high"] - r["rate"]]],
                    fmt="o", ms=7, color=color, capsize=3, lw=2, zorder=3)
        ax.annotate(f"{r['rate']:.2f}{star} (n={r['n_win'] + r['n_loss']})",
                    (r["ci_high"], y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK2)
    ax.axvline(0.5, color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=2)
    ax.set_yticks(ys, [r["row_label"] for r in rows], fontsize=9)
    ax.set_xlim(0, 1)
    xlabel = "full 优于 title-only 的逐片比例（95% Wilson CI，* p<0.05）" if zh else \
        "per-movie win rate (95% Wilson CI, * p<0.05)"
    ax.set_xlabel(xlabel, color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, facecolor=SURFACE)
    plt.close(fig)


def _genre_plot(g: pd.DataFrame, pop: str, zh: bool, out_png: Path) -> None:
    g = g.sort_values("mean_delta")
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(g) + 1.6), dpi=150)
    _style(ax)
    colors = [BLUE if v > 0 else ORANGE for v in g["mean_delta"]]
    ax.barh(np.arange(len(g)), g["mean_delta"], color=colors, height=0.62, zorder=3)
    ax.set_yticks(np.arange(len(g)),
                  [f"{ge} (n={n})" for ge, n in zip(g["genre"], g["n"])], fontsize=9)
    ax.axvline(0, color=INK2, lw=0.8, zorder=2)
    xlabel = f"mean Δ NDCG@k（{pop}，n≥3 的类型）" if zh else f"mean Δ NDCG@k ({pop})"
    ax.set_xlabel(xlabel, color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, facecolor=SURFACE)
    plt.close(fig)


def main(config_path: str, pairs: list[str], k: int = 10,
         min_genre_n: int = 3) -> int:
    config = load_config(config_path)
    paths = paths_for(config)
    groups = load_tag_groups(paths.tag_groups)
    proto = config.protocol
    meta = pd.read_parquet(paths.dataset)[["movie_id", "title", "year", "genres"]]
    zh = _cjk()

    ecdf_frames: dict[str, pd.DataFrame] = {}
    winrate_rows: list[dict] = []
    summary: dict[str, dict] = {}
    out_root = paths.root / "results" / pairs[0].split(",")[0].strip() / "deep_analysis"
    out_root.mkdir(parents=True, exist_ok=True)
    for spec in pairs:
        full_exp, title_exp, pop = (s.strip() for s in spec.split(","))
        pm_f = _per_movie(paths, full_exp, groups, proto.top_k, proto.relevance_threshold)
        pm_t = _per_movie(paths, title_exp, groups, proto.top_k, proto.relevance_threshold)
        pt = per_type_deltas(pm_f, pm_t)
        ecdf_frames[pop] = pt
        levels = pm_f[["movie_id", "ndcg_all"]].merge(
            pm_t[["movie_id", "ndcg_all"]], on="movie_id",
            suffixes=("_f", "_t")).rename(
            columns={"ndcg_all_f": "full", "ndcg_all_t": "title_prior"})
        movers_src = pt.merge(levels, on="movie_id").rename(
            columns={"delta_all": "delta"})

        summary[pop] = {"mean_delta": {c: float(pt[f"delta_{c}"].mean())
                                       for c in ("all", "content", "preference")}}
        wr = {}
        for c in ("all", "content", "preference"):
            w = win_rate(pt[f"delta_{c}"])
            wr[c] = w
            winrate_rows.append({**w, "population": pop, "type": c,
                                 "row_label": f"{pop} · {c}"})
        summary[pop]["win_rate"] = wr
        summary[pop]["by_decade"] = [
            {"decade": f"{int(d)}s", "n": int(len(sub)),
             "mean_delta": float(sub["delta_all"].mean())}
            for d, sub in pt.merge(meta, on="movie_id").assign(
                decade=lambda x: x["year"] // 10 * 10).groupby("decade")]

        movers = top_movers(
            movers_src[["movie_id", "title_prior", "full", "delta"]], meta, k=k)
        pd.concat([movers["gainers"].assign(direction="gain"),
                   movers["losers"].assign(direction="loss")]).to_csv(
            out_root / f"movers_{pop}.csv", index=False, encoding="utf-8-sig")
        g = genre_aggregate(pt.rename(columns={"delta_all": "delta"})[
            ["movie_id", "delta"]], meta, min_n=min_genre_n)
        g.to_csv(out_root / f"delta_by_genre_{pop}.csv", index=False,
                 encoding="utf-8-sig")
        if pop == "newfilm":
            _genre_plot(g, pop, zh, out_root / "genre_delta_newfilm.png")

    _ecdf_plot(ecdf_frames, zh, out_root / "delta_ecdf.png")
    _winrate_plot(winrate_rows, zh, out_root / "winrate_by_type.png")
    (out_root / "deep_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[11] 深化分析 → {out_root}")
    for pop, s in summary.items():
        md, wr = s["mean_delta"], s["win_rate"]
        print(f"[11] {pop}: meanΔ all={md['all']:.4f} content={md['content']:.4f} "
              f"pref={md['preference']:.4f}")
        for c in ("all", "content", "preference"):
            w = wr[c]
            print(f"[11]   win[{c}] {w['n_win']}W/{w['n_loss']}L/{w['n_tie']}T "
                  f"rate={w['rate']:.3f} p={w['p']:.4f}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/newfilm/full.yaml")
    parser.add_argument("--pair", action="append", required=True,
                        help="full_exp,title_exp,pop_label（可多次）")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--min-genre-n", type=int, default=3)
    args = parser.parse_args()
    raise SystemExit(main(args.config, args.pair, args.k, args.min_genre_n))
