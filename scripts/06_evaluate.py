"""06 · 评估：annotations + 基线 → 指标表 / 报告 / 热力图。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# 仓库根加入 sys.path，使脚本在未安装包的情况下也能直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genometag.config import load_config, paths_for  # noqa: E402


def main(config_path: str = "configs/default.yaml",
         extra_exps: list[str] | None = None) -> int:
    config = load_config(config_path)
    paths = paths_for(config)
    from genometag.data.movielens import load_user_tags
    from genometag.eval.report import write_report
    from genometag.tags.grouping import load_tag_groups

    dataset = pd.read_parquet(paths.dataset)
    truth = pd.read_parquet(paths.genome_scores)
    tag_groups = load_tag_groups(paths.tag_groups)
    user_tags_path = paths.movielens / "tags.csv"
    user_tags = load_user_tags(user_tags_path) if user_tags_path.exists() else None

    exps = [paths.results] + [paths.root / "results" / e for e in (extra_exps or [])]
    exp_dirs = [e for e in exps if (e / "annotations.parquet").exists()]
    baseline_dir = paths.root / "results" / "baselines"
    out = write_report(exp_dirs, baseline_dir, paths.results / "evaluation",
                       config.protocol, dataset, truth, tag_groups, user_tags)
    print(f"[06] 评估完成 → {out}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--extra", nargs="*", default=[],
                        help="参与对比的其他实验名（如 ablation/video-only）")
    raise SystemExit(main(parser.parse_args().config, parser.parse_args().extra))
