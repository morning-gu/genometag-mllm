"""07 · 记忆探针 2：错配内容——A 标题 + B 内容的因果检验（spec 2026-10-16 修订）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from genometag.annotate.cache import ResponseCache  # noqa: E402
from genometag.annotate.client import make_client  # noqa: E402
from genometag.config import load_config, paths_for  # noqa: E402
from genometag.probes.mismatch import run_mismatch_probe  # noqa: E402


def main(config_path: str = "configs/probes/mismatch.yaml") -> int:
    config = load_config(config_path)
    paths = paths_for(config)
    client = make_client(config)
    cache = None if config.mock else ResponseCache(paths.cache)
    out = run_mismatch_probe(config, paths, client, cache)
    print(f"[07] 错配探针完成 → {out}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/probes/mismatch.yaml")
    raise SystemExit(main(parser.parse_args().config))
