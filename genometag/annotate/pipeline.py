"""标注编排：电影 × tag组 × 轮次 任务矩阵，断点续跑，解析重试降级，可选并行。

annotations.jsonl 是续跑的事实来源（append-only，每任务一行，行级 flush——
看门狗以 jsonl 增长判活）；结束时压缩为 annotations.parquet 供评估使用。
任务相互独立，workers>1 时线程池并行与串行结果等价（有测试钉死）。
mock 模式不读不写缓存（确定性 + 不污染真实缓存）。
"""
from __future__ import annotations

import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from genometag.annotate.cache import ResponseCache, cache_key
from genometag.annotate.client import sampling_params
from genometag.annotate.prompts import (
    FewShot, TagDef, build_messages, parse_scores_lenient,
)
from genometag.config import AppConfig, paths_for
from genometag.data.movienet import MovieMultimodal
from genometag.tags.grouping import load_tag_groups


@dataclass(frozen=True)
class TaskSpec:
    movie_id: int
    group: str
    run_id: str


def build_task_matrix(movie_ids: list[int], tag_groups: pd.DataFrame,
                      n_runs: int) -> list[TaskSpec]:
    groups = sorted(tag_groups["group"].unique())
    return [TaskSpec(m, g, f"run-{r}")
            for m in movie_ids for g in groups for r in range(n_runs)]


def select_few_shots(dataset: pd.DataFrame, genome_scores: pd.DataFrame,
                     n: int, split_year: int, top_k: int,
                     tag_names: dict[int, str]) -> list[FewShot]:
    """从 year < split_year 的电影中选 top-k 分数和最高的 n 部（防泄露约束）。

    tag_names: tag_id → tag 文本（来自 tag_groups；few-shot 展示 tag 名而非 id）。
    """
    if n <= 0:
        return []
    train = dataset[dataset["year"] < split_year]
    scores = genome_scores[genome_scores["movie_id"].isin(set(train["movie_id"]))]
    if scores.empty:
        return []
    top_sums = (scores.groupby("movie_id")["relevance"]
                .apply(lambda s: s.nlargest(top_k).sum()))
    best = top_sums.sort_values(ascending=False).head(n).index.tolist()
    title_map = dict(zip(dataset["movie_id"], dataset["title"]))
    year_map = dict(zip(dataset["movie_id"], dataset["year"]))
    shots = []
    for movie_id in best:
        rows = genome_scores[genome_scores["movie_id"] == movie_id].nlargest(top_k, "relevance")
        shots.append(FewShot(
            title=title_map[movie_id], year=int(year_map[movie_id]),
            scores={tag_names[int(t)]: round(float(r), 2)
                    for t, r in zip(rows["tag_id"], rows["relevance"])}))
    return shots


def _read_jsonl_lines(jsonl_path: Path) -> tuple[list[dict], int]:
    """逐行解析 jsonl，容忍中断残留的截断/损坏行（跳过并计数）。"""
    good: list[dict] = []
    bad = 0
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            good.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    if bad:
        print(f"[04] 跳过损坏的 jsonl 行: {bad}（上次运行中断残留，相关任务将重跑）")
    return good, bad


def run_annotation(config: AppConfig, paths, client, cache: ResponseCache | None,
                   dry_run: bool = False) -> Path | None:
    dataset = pd.read_parquet(paths.dataset)
    genome_scores = pd.read_parquet(paths.genome_scores)
    tag_groups = load_tag_groups(paths.tag_groups)

    movie_ids = sorted(dataset["movie_id"].tolist())
    if config.data.movie_limit is not None and config.data.movie_limit < len(movie_ids):
        movie_ids = random.Random(config.data.seed).sample(movie_ids, config.data.movie_limit)

    tag_names = dict(zip(tag_groups["tag_id"], tag_groups["tag"]))
    few_shots = select_few_shots(
        dataset, genome_scores, config.annotation.few_shot_examples,
        config.protocol.temporal_split_year, config.protocol.top_k, tag_names)

    tasks = build_task_matrix(movie_ids, tag_groups, config.annotation.n_runs)
    # 记忆探针 1（spec 2026-10-16 修订）：modalities=[] 是显式的 title-only 臂
    modality_set = "|".join(config.annotation.modalities) or "title-only"
    groups_df = {g: df for g, df in tag_groups.groupby("group")}
    movie_rows = {int(row["movie_id"]): row for _, row in dataset.iterrows()}

    if dry_run:
        # spec §4.5 的 --dry-run：打印前 3 个任务的 prompt 文本，不调用、不写文件
        for task in tasks[:3]:
            mm = _to_multimodal(movie_rows[task.movie_id], paths.movienet)
            tag_defs = [TagDef(int(r.tag_id), r.tag, r.definition)
                        for r in groups_df[task.group].itertuples()]
            messages = build_messages(
                mm, tag_defs,
                _available_modalities(mm, config.annotation.modalities),
                few_shots, config.model)
            texts = []
            for m in messages:
                c = m["content"]
                if isinstance(c, str):
                    texts.append(c)
                else:
                    texts.extend(p.get("text", "") for p in c if p.get("type") == "text")
            print(f"[dry-run] 任务 {task.movie_id}|{task.group}|{task.run_id}\n"
                  + "\n".join(texts)[:1500] + "\n")
        return None

    paths.results.mkdir(parents=True, exist_ok=True)
    jsonl_path = paths.results / "annotations.jsonl"
    done_keys: set[str] = set()
    if jsonl_path.exists():
        records, _ = _read_jsonl_lines(jsonl_path)
        done_keys = {r["task_key"] for r in records}
    todo = [t for t in tasks if f"{t.movie_id}|{t.group}|{t.run_id}" not in done_keys]
    total = len(todo)
    n_workers = max(1, config.annotation.workers)
    if total:
        print(f"[04] 待跑 {total} 个任务（续跑跳过 {len(done_keys)}），"
              f"workers={n_workers}", flush=True)

    cost = {"total_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    counts = {"done": 0, "parse_failed": 0, "no_content": 0, "null_scores": 0}
    lock = threading.Lock()

    with jsonl_path.open("a", encoding="utf-8") as sink:
        def process(task: TaskSpec) -> None:
            """单任务：构建 prompt → 调用（缓存优先）→ 宽松解析；落盘在锁内。"""
            task_key = f"{task.movie_id}|{task.group}|{task.run_id}"
            mm = _to_multimodal(movie_rows[task.movie_id], paths.movienet)
            group_tags = groups_df[task.group]
            available = _available_modalities(mm, config.annotation.modalities)
            call_cost = {"total_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
            # 仅当"请求了内容模态但均不可得"才跳过；modalities=[] 是 title-only 探针的显式意图
            if not available and config.annotation.modalities:
                status, nulls, usage = "no_content", 0, {}
                line = json.dumps({"task_key": task_key, "status": status,
                                   "rows": [], "usage": usage})
            else:
                tag_defs = [TagDef(int(r.tag_id), r.tag, r.definition)
                            for r in group_tags.itertuples()]
                messages = build_messages(mm, tag_defs, available, few_shots, config.model)
                expected = [t.tag for t in tag_defs]

                # 宽松解析（服务器实测：完整 JSON 偶尔漏 1-2 tag 或输出字符串数值——
                # 全有或全无会报废 58/60 的好结果；缺失记 null 并计数，仅全空才重试）
                scores: dict[str, float] = {}
                usage: dict = {}
                content = ""
                for attempt in range(3):
                    eff_run = task.run_id if attempt == 0 else f"{task.run_id}#r{attempt}"
                    raw = None
                    if cache is not None and not config.mock:
                        key = cache_key(messages, config.model.model,
                                        sampling_params(config.model), eff_run)
                        raw = cache.get(key)
                    if raw is None:
                        resp = client.complete(messages, run_id=eff_run)
                        raw = resp.raw
                        content = resp.content
                        usage = resp.usage
                        call_cost["total_calls"] += 1
                        call_cost["prompt_tokens"] += usage.get("prompt_tokens", 0)
                        call_cost["completion_tokens"] += usage.get("completion_tokens", 0)
                        if cache is not None and not config.mock:
                            cache.put(key, raw)
                    else:
                        # 缓存命中：存的是完整原始响应信封，从信封还原文本
                        content = raw.get("choices", [{}])[0].get(
                            "message", {}).get("content", "")
                    scores = parse_scores_lenient(content, expected)
                    if scores:
                        break

                status = "done" if scores else "parse_failed"
                nulls = len(expected) - len(scores)
                rows = [{"movie_id": task.movie_id, "tag_id": int(td.tag_id),
                         "group": task.group, "modality_set": modality_set,
                         "run_id": task.run_id,
                         "score": scores.get(td.tag)}
                        for td in tag_defs]
                line = json.dumps({"task_key": task_key, "status": status,
                                   "rows": rows, "usage": usage}, ensure_ascii=False)

            with lock:
                sink.write(line + "\n")
                sink.flush()   # 行级落盘：看门狗以 jsonl 增长判活（并行长跑必需）
                counts[status] += 1
                counts["null_scores"] += nulls
                for k in call_cost:
                    cost[k] += call_cost[k]
                processed = counts["done"] + counts["parse_failed"] + counts["no_content"]
                if processed % 25 == 0 or processed == total:
                    print(f"[04] 进度 {processed}/{total}"
                          f"（done={counts['done']} failed={counts['parse_failed']}）",
                          flush=True)

        if n_workers == 1:
            for task in todo:
                process(task)
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = [pool.submit(process, t) for t in todo]
                for f in futures:
                    f.result()

    # 压缩 jsonl → parquet（含历史行，防丢；损坏行跳过，不阻塞续跑）
    records, _ = _read_jsonl_lines(jsonl_path)
    all_rows: list[dict] = []
    for r in records:
        all_rows.extend(r["rows"])
    parquet_path = paths.results / "annotations.parquet"
    pd.DataFrame(all_rows,
                 columns=["movie_id", "tag_id", "group", "modality_set", "run_id", "score"]
                 ).to_parquet(parquet_path, index=False)

    (paths.results / "run_manifest.json").write_text(
        json.dumps({"tasks": counts, **cost}, ensure_ascii=False, indent=1), encoding="utf-8")
    (paths.results / "cost_report.json").write_text(
        json.dumps(cost, ensure_ascii=False, indent=1), encoding="utf-8")
    return parquet_path


def _available_modalities(mm: MovieMultimodal, wanted: list[str]) -> list[str]:
    avail = []
    if "video" in wanted and mm.keyframes:
        avail.append("video")
    if "script" in wanted and mm.script:
        avail.append("script")
    if "dialogue" in wanted and mm.subtitles:
        avail.append("dialogue")
    return avail


def _to_multimodal(row, movienet_root: Path) -> MovieMultimodal:
    # 注意：混合 str/缺失列经 pandas3 string dtype + parquet 往返后缺失为 NaN（float）
    # —— NaN 为真值，必须用 pd.notna 守卫（真实数据 479/1035 有剧本的混合列实测触发）
    frames = [movienet_root / k for k in row["keyframes"]]
    script = None
    if pd.notna(row["script_path"]) and row["script_path"]:
        p = movienet_root / row["script_path"]
        script = p.read_text(encoding="utf-8", errors="replace") if p.exists() else None
    subtitles = None
    if pd.notna(row["subtitles_path"]) and row["subtitles_path"]:
        p = movienet_root / row["subtitles_path"]
        subtitles = p.read_text(encoding="utf-8", errors="replace") if p.exists() else None
    return MovieMultimodal(str(row["movienet_id"]), str(row["title"]),
                           int(row["year"]) if pd.notna(row["year"]) else None,
                           None, frames, script, subtitles)
