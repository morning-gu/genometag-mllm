"""端到端冒烟：configs/smoke.yaml（合成样本 + mock）串联 00→06。

用途：日常回归 + pytest 集成测试。幂等：先清空 results/smoke/。
"""
from __future__ import annotations

import importlib
import json
import shutil
import sys
from pathlib import Path

# 仓库根加入 sys.path，使脚本在未安装包的情况下也能直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genometag.config import load_config, paths_for  # noqa: E402

STEPS = ["00_make_synthetic", "01_verify_data", "02_build_dataset", "03_group_tags",
         "04_run_annotation", "05_run_baselines", "06_evaluate"]


def main() -> int:
    cfg_path = "configs/smoke.yaml"
    config = load_config(cfg_path)
    paths = paths_for(config)
    if paths.results.exists():
        shutil.rmtree(paths.results)          # 幂等

    for step in STEPS:
        mod = importlib.import_module(f"scripts.{step}")
        rc = mod.main(cfg_path)
        if rc != 0:
            print(f"[smoke] {step} 失败 (rc={rc})")
            return 1
        print(f"[smoke] {step} ✓")

    # 关键产物断言
    checks = [
        paths.dataset,
        paths.tag_groups,
        paths.results / "annotations.parquet",
        paths.results / "run_manifest.json",
        paths.root / "results" / "baselines" / "baselines.parquet",
        paths.results / "evaluation" / "metrics.csv",
        paths.results / "evaluation" / "consistency.json",   # smoke n_runs=2 → 一致性产出
    ]
    for c in checks:
        if not c.exists():
            print(f"[smoke] 缺少产物: {c}")
            return 1
    manifest = json.loads((paths.results / "run_manifest.json").read_text(encoding="utf-8"))
    print(f"[smoke] PASS — tasks={manifest.get('tasks')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
