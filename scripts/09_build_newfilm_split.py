"""09 · 生成新片评估集清单（≥year-min 帧+字幕可用，剧本不要求；强制排除既有评估集防泄露）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# 仓库根加入 sys.path，使脚本在未安装包的情况下也能直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genometag.config import load_config, paths_for  # noqa: E402
from genometag.eval.protocol import select_newfilm_ids, write_movie_list  # noqa: E402


def main(config_path: str = "configs/default.yaml", year_min: int = 2015) -> int:
    config = load_config(config_path)
    paths = paths_for(config)
    ds = pd.read_parquet(paths.dataset)
    genome = pd.read_parquet(paths.genome_scores)
    eval_file = paths.processed / "eval_test.txt"
    if not eval_file.exists():
        raise ValueError(f"评估集清单不存在（防泄露排除依赖它）: {eval_file}")
    eval_ids = [int(x) for x in eval_file.read_text(encoding="utf-8").split() if x.strip()]

    ids = select_newfilm_ids(ds, genome, eval_ids, year_min=year_min)
    out = paths.processed / "newfilm2015.txt"
    status = write_movie_list(out, ids)
    sub = ds[ds["movie_id"].isin(ids)]
    print(f"[09] 新片清单（{status}）: {len(ids)} 部（year ≥ {year_min}，帧+字幕）"
          f" 年份 {int(sub['year'].min())}-{int(sub['year'].max())} → {out}")
    print(f"[09] 与 eval_test 交集: {len(set(ids) & set(eval_ids))}（必须为 0）")

    if status == "written" and ids:
        from genometag.annotate.pipeline import select_few_shots
        from genometag.tags.grouping import load_tag_groups

        groups = load_tag_groups(paths.tag_groups)
        tag_names = dict(zip(groups["tag_id"], groups["tag"]))
        pool_ids = sorted(set(ds["movie_id"]) - set(ids))
        shots = select_few_shots(ds, genome, 2, pool_ids, 10, tag_names)
        print("     few-shot（拟合池 top-2）:", [f"{s.title} ({s.year})" for s in shots])
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--year-min", type=int, default=2015)
    args = parser.parse_args()
    raise SystemExit(main(args.config, args.year_min))
