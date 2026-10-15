"""04 · MLLM 标注主跑（mock 配置零成本；--dry-run 打印 prompt 不调用）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 仓库根加入 sys.path，使脚本在未安装包的情况下也能直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genometag.annotate.cache import ResponseCache  # noqa: E402
from genometag.annotate.client import make_client  # noqa: E402
from genometag.annotate.pipeline import run_annotation  # noqa: E402
from genometag.config import load_config, paths_for  # noqa: E402


def main(config_path: str = "configs/default.yaml", dry_run: bool = False) -> int:
    config = load_config(config_path)
    paths = paths_for(config)
    client = make_client(config)
    cache = None if config.mock else ResponseCache(paths.cache)
    out = run_annotation(config, paths, client, cache, dry_run=dry_run)
    if out is None:
        print("[04] dry-run 完成（未调用、未写文件）")
    else:
        print(f"[04] 标注完成 → {out}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dry-run", action="store_true", help="打印 prompt，不调用不写文件")
    args = parser.parse_args()
    raise SystemExit(main(args.config, args.dry_run))
