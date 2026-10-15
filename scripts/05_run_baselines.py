"""05 · 运行基线。mean/genre 只在时序训练参考集上拟合（防泄露）。"""
from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

# 仓库根加入 sys.path，使脚本在未安装包的情况下也能直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genometag.config import load_config, paths_for  # noqa: E402


def run_baselines(config, paths) -> tuple[pd.DataFrame, dict]:
    dataset = pd.read_parquet(paths.dataset)
    scores = pd.read_parquet(paths.genome_scores)
    from genometag.eval.protocol import temporal_split
    train_ids, test_ids = temporal_split(dataset, config.protocol.temporal_split_year)
    train_scores = scores[scores["movie_id"].isin(set(train_ids))]
    tag_ids = sorted(scores["tag_id"].unique())
    movie_ids = sorted(dataset["movie_id"].tolist())
    test_set = set(test_ids)

    frames = []
    report = {"train_movies": len(train_ids), "test_movies": len(test_ids)}
    for name in config.baselines.run:
        if name == "random":
            from genometag.baselines.random_baseline import random_baseline
            frames.append(random_baseline(movie_ids, tag_ids, config.data.seed))
        elif name == "mean":
            from genometag.baselines.mean_baseline import mean_baseline
            frames.append(mean_baseline(train_scores, movie_ids, test_ids=test_set))
        elif name == "genre":
            from genometag.baselines.genre_baseline import genre_baseline
            frames.append(genre_baseline(train_scores, dataset, movie_ids, test_ids=test_set))
        elif name == "bert":
            try:
                mod = importlib.import_module("genometag.baselines.bert_baseline")
                frames.append(mod.bert_baseline(config, paths, dataset, scores,
                                                train_ids, test_ids))
            except ImportError:
                # 懒导入：模块自身无 torch 也能导入，缺依赖在调用点才抛出
                print("[05] torch/transformers 未安装，跳过 BERT 基线"
                      "（pip install -r requirements-baselines.txt）")
                report["bert"] = "skipped"
                continue
        else:
            raise ValueError(f"未知基线: {name}")
        report[name] = "ok"
    out = pd.concat(frames, ignore_index=True)
    return out, report


def main(config_path: str = "configs/default.yaml") -> int:
    config = load_config(config_path)
    paths = paths_for(config)
    df, report = run_baselines(config, paths)
    out_dir = paths.root / "results" / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    # 配置快照随结果落盘（spec §2/§5.4/§6）：结果目录自包含、可复现
    if Path(config_path).exists():
        shutil.copy(config_path, out_dir / "config.yaml")
    df.to_parquet(out_dir / "baselines.parquet", index=False)
    (out_dir / "fit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[05] 基线完成: {len(df)} 行 → {out_dir / 'baselines.parquet'}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    raise SystemExit(main(parser.parse_args().config))
