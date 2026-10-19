"""记忆探针 2：错配内容（spec 2026-10-16 修订）。

因果检验"MLLM 的预测是否由内容驱动"：prompt = 电影 A 的标题/年份 + 电影 B 的
内容（关键帧/剧本/台词），同类型/跨类型双对照采样。

解读：
- 预测钉住 A 的真值（corr_A 高、shift = corr_B − corr_A ≈ 0）→ 模型忽略内容（记忆 A）
- 预测移向 B 的真值（corr_B 上升 / corr_A 下降）→ 内容被真实使用
同类型对照控制"B 内容恰好通用"的混淆；跨类型条件下内容差异最大，迁移信号应最强。

单轮调用（探针无需多轮平均，成本 ~n_pairs × 组数）。
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from genometag.annotate.cache import cache_key
from genometag.annotate.client import sampling_params
from genometag.annotate.pipeline import _to_multimodal, select_few_shots
from genometag.annotate.prompts import TagDef, build_messages, parse_scores
from genometag.data.movienet import MovieMultimodal
from genometag.eval.protocol import fitting_pool_ids, load_eval_movie_ids
from genometag.tags.grouping import load_tag_groups


@dataclass(frozen=True)
class MismatchPair:
    pair_id: int
    movie_a: int
    movie_b: int
    condition: str          # same_genre | cross_genre


def _eligible_as_b(dataset: pd.DataFrame, modalities: list[str]) -> set[int]:
    """B 必须在所选模态上内容可用（dataset 字段即 build_dataset 写入时的真值）。"""
    ok = set(dataset["movie_id"])
    if "video" in modalities:
        ok &= set(dataset[dataset["keyframes"].map(len) > 0]["movie_id"])
    if "script" in modalities:
        ok &= set(dataset[dataset["script_path"].notna()]["movie_id"])
    if "dialogue" in modalities:
        ok &= set(dataset[dataset["subtitles_path"].notna()]["movie_id"])
    return ok


def sample_pairs(dataset: pd.DataFrame, n_pairs: int, seed: int,
                 modalities: list[str]) -> list[MismatchPair]:
    """同类型/跨类型各半采样；类型判定按 genres 集合的交集。"""
    rng = random.Random(seed)
    eligible = _eligible_as_b(dataset, modalities)
    rows = {int(r.movie_id): r for r in dataset.itertuples()}
    genres = {mid: set(rows[mid].genres) for mid in rows}
    all_ids = sorted(rows)

    def _find_b(a: int, want_share: bool) -> int | None:
        cands = [b for b in sorted(eligible)
                 if b != a and bool(genres[a] & genres[b]) == want_share]
        return rng.choice(cands) if cands else None

    pairs: list[MismatchPair] = []
    per_cond = max(1, n_pairs // 2)
    for condition, want_share in (("same_genre", True), ("cross_genre", False)):
        made, attempts = 0, 0
        while made < per_cond and attempts < per_cond * 20:
            attempts += 1
            a = rng.choice(all_ids)
            b = _find_b(a, want_share)
            if b is None:
                continue
            pairs.append(MismatchPair(len(pairs), a, b, condition))
            made += 1
    return pairs


def _hybrid(mm_a: MovieMultimodal, mm_b: MovieMultimodal) -> MovieMultimodal:
    """A 的身份（标题/年份）+ B 的内容（关键帧/剧本/台词）。"""
    return MovieMultimodal(movienet_id=mm_b.movienet_id, title=mm_a.title,
                           year=mm_a.year, imdb_id=None,
                           keyframes=mm_b.keyframes, script=mm_b.script,
                           subtitles=mm_b.subtitles)


def run_mismatch_probe(config, paths, client, cache) -> Path:
    dataset = pd.read_parquet(paths.dataset)
    genome_scores = pd.read_parquet(paths.genome_scores)
    tag_groups = load_tag_groups(paths.tag_groups)
    tag_names = dict(zip(tag_groups["tag_id"], tag_groups["tag"]))
    eval_ids = load_eval_movie_ids(config, paths)
    fit_pool = fitting_pool_ids(config, dataset, eval_ids)
    few_shots = select_few_shots(
        dataset, genome_scores, config.annotation.few_shot_examples,
        fit_pool, config.protocol.top_k, tag_names)

    pairs = sample_pairs(dataset, config.probes.n_pairs,
                         config.data.seed, config.annotation.modalities)
    groups_df = {g: df for g, df in tag_groups.groupby("group")}
    rows = {int(r["movie_id"]): r for _, r in dataset.iterrows()}  # Series 行：_to_multimodal 需要

    paths.results.mkdir(parents=True, exist_ok=True)
    pred_rows: list[dict] = []
    for pair in pairs:
        mm_a = _to_multimodal(rows[pair.movie_a], paths.movienet)
        mm_b = _to_multimodal(rows[pair.movie_b], paths.movienet)
        hybrid = _hybrid(mm_a, mm_b)
        for group, gdf in groups_df.items():
            tag_defs = [TagDef(int(r.tag_id), r.tag, r.definition)
                        for r in gdf.itertuples()]
            messages = build_messages(hybrid, tag_defs,
                                      config.annotation.modalities,
                                      few_shots, config.model)
            run_id = f"mismatch-{pair.pair_id}"
            content = None
            if cache is not None and not config.mock:
                key = cache_key(messages, config.model.model,
                                sampling_params(config.model), run_id)
                raw = cache.get(key)
                if raw is not None:          # 缓存命中：从原始响应信封提取
                    content = raw.get("choices", [{}])[0].get(
                        "message", {}).get("content", "")
            if content is None:              # 新调用：直接取 resp.content（mock 同）
                resp = client.complete(messages, run_id=run_id)
                if cache is not None and not config.mock:
                    cache.put(key, resp.raw)
                content = resp.content
            scores = parse_scores(content, [t.tag for t in tag_defs])
            if scores is None:
                continue
            for td in tag_defs:
                pred_rows.append({"pair_id": pair.pair_id,
                                  "movie_a": pair.movie_a, "movie_b": pair.movie_b,
                                  "condition": pair.condition, "tag_id": td.tag_id,
                                  "score": scores[td.tag]})

    out = paths.results / "mismatch.parquet"
    pd.DataFrame(pred_rows, columns=["pair_id", "movie_a", "movie_b",
                                     "condition", "tag_id", "score"]
                 ).to_parquet(out, index=False)

    # 评估：每对的 pred 向量 vs truth_A / truth_B（Pearson + RMSE）
    truth = {m: dict(zip(g.tag_id, g.relevance))
             for m, g in genome_scores.groupby("movie_id")}
    per_pair = []
    for pair in pairs:
        sub = [r for r in pred_rows if r["pair_id"] == pair.pair_id]
        if not sub:
            continue
        tags = [r["tag_id"] for r in sub]
        pred = np.array([r["score"] for r in sub])
        ta = np.array([truth[pair.movie_a].get(t, np.nan) for t in tags])
        tb = np.array([truth[pair.movie_b].get(t, np.nan) for t in tags])
        def _corr(x: np.ndarray) -> float:
            mask = ~(np.isnan(pred) | np.isnan(x))
            p, xv = pred[mask], x[mask]
            return float(np.corrcoef(p, xv)[0, 1]) if p.std() > 0 and xv.std() > 0 else 0.0
        per_pair.append({"pair_id": pair.pair_id, "condition": pair.condition,
                         "corr_a": _corr(ta), "corr_b": _corr(tb),
                         "rmse_a": float(np.sqrt(np.nanmean((pred - ta) ** 2))),
                         "rmse_b": float(np.sqrt(np.nanmean((pred - tb) ** 2)))})
    pp = pd.DataFrame(per_pair)
    summary = {}
    for cond, g in pp.groupby("condition"):
        summary[cond] = {
            "n_pairs": int(len(g)),
            "mean_corr_a": float(g.corr_a.mean()), "mean_corr_b": float(g.corr_b.mean()),
            "mean_shift": float((g.corr_b - g.corr_a).mean()),
        }
    summary["interpretation"] = (
        "corr_A 高且 shift≈0 → 记忆主导（预测钉住 A 的真值标签）；"
        "corr_B 上升/corr_A 下降 → 内容被真实使用（预测随内容迁移）")
    (paths.results / "mismatch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[07] 错配探针完成:", json.dumps(summary, ensure_ascii=False))
    return out
