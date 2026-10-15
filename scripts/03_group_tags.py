"""03 · 生成 tag_groups.json（幂等：已存在则跳过；synthetic 源直接用已知分组）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 仓库根加入 sys.path，使脚本在未安装包的情况下也能直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from genometag.config import load_config, paths_for  # noqa: E402


def main(config_path: str = "configs/default.yaml") -> int:
    config = load_config(config_path)
    paths = paths_for(config)
    if paths.tag_groups.exists():
        print(f"[03] 已存在，跳过: {paths.tag_groups}（删除可重新生成）")
        return 0

    genome_tags = pd.read_parquet(paths.genome_tags)
    if config.data.source == "synthetic":
        from genometag.data.sample import SYNTHETIC_TAGS
        rows = SYNTHETIC_TAGS
    else:
        from genometag.annotate.cache import ResponseCache
        from genometag.annotate.client import make_client
        from genometag.tags.grouping import group_tags_via_llm
        df = group_tags_via_llm(
            tags=genome_tags["tag"].tolist(),
            client=make_client(config),
            cache=ResponseCache(paths.cache),
            model=config.model.model)
        id_map = dict(zip(genome_tags["tag"], genome_tags["tag_id"]))
        rows = [{"tag_id": id_map[r["tag"]], **r}
                for _, r in df.iterrows()]
    paths.processed.mkdir(parents=True, exist_ok=True)
    paths.tag_groups.write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                encoding="utf-8")
    print(f"[03] tag 分组完成: {len(rows)} 个 tag, "
          f"{len({r['group'] for r in rows})} 个组 → {paths.tag_groups}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    raise SystemExit(main(parser.parse_args().config))
