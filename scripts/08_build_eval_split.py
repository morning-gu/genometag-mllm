"""08 · 生成协议 v3 评估集清单（三模态宇宙 + 年代分层随机留出）。

宇宙 = keyframes + 剧本 + 字幕三模态全有的电影；
测试集 = 宇宙内按年代（十年段）比例分配 + 组内随机（seed 固定）；
拟合池 = 运行时按"评估集补集"计算（eval.protocol.fitting_pool_ids），无需落盘。
产物 data/processed/eval_test.txt 入 git（协议可审计、可复现）。
幂等：已存在则跳过（删除可重新生成）。
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

# 仓库根加入 sys.path，使脚本在未安装包的情况下也能直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genometag.config import load_config, paths_for  # noqa: E402


def main(config_path: str = "configs/default.yaml", test_frac: float = 0.2,
         seed: int = 42) -> int:
    config = load_config(config_path)
    paths = paths_for(config)
    out = paths.processed / "eval_test.txt"
    if out.exists():
        print(f"[08] 已存在，跳过: {out}（删除可重新生成）")
        return 0

    ds = pd.read_parquet(paths.dataset)
    all3 = ((ds["keyframes"].apply(len) > 0) & ds["script_path"].notna()
            & ds["subtitles_path"].notna())
    universe = ds[all3]
    print(f"[08] 三模态宇宙: {len(universe)}/{len(ds)} 部")

    decades = (universe["year"] // 10 * 10).astype(int)
    counts = decades.value_counts().sort_index()
    n_test = round(len(universe) * test_frac)
    raw = counts * test_frac
    alloc = raw.astype(int)                       # 比例分配：floor
    remain = n_test - int(alloc.sum())
    if remain > 0:                                # 最大余数法补齐
        for d in (raw - alloc).sort_values(ascending=False).index[:remain]:
            alloc[d] += 1

    rng = random.Random(seed)
    picked: list[int] = []
    for d in sorted(alloc.index):
        pool = sorted(universe.loc[decades == d, "movie_id"].tolist())
        k = min(int(alloc[d]), len(pool))
        picked.extend(rng.sample(pool, k))
    picked.sort()

    paths.processed.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(str(m) for m in picked) + "\n", encoding="utf-8")
    print(f"[08] 测试集: {len(picked)} 部（{test_frac:.0%}，seed={seed}）→ {out}")
    print("     年代配额:", {f"{d}s": int(alloc[d]) for d in sorted(alloc.index)})

    # 诊断：few-shot 预览（拟合池 = 补集，与 04/05 运行时语义一致）
    from genometag.annotate.pipeline import select_few_shots
    from genometag.tags.grouping import load_tag_groups
    genome = pd.read_parquet(paths.genome_scores)
    groups = load_tag_groups(paths.tag_groups)
    tag_names = dict(zip(groups["tag_id"], groups["tag"]))
    pool_ids = sorted(set(ds["movie_id"]) - set(picked))
    shots = select_few_shots(ds, genome, 2, pool_ids, 10, tag_names)
    print("     few-shot（拟合池 top-2）:", [f"{s.title} ({s.year})" for s in shots])
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    raise SystemExit(main(args.config, args.test_frac, args.seed))
