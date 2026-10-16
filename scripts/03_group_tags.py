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


def main(config_path: str = "configs/default.yaml", batch_size: int = 150) -> int:
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
        # mock 不读不写缓存：mock:true + 真实源的组合也不得把 mock 信封写进真实缓存
        cache = None if config.mock else ResponseCache(paths.cache)
        # batch_size 受端点上下文约束：每批输出约 batch×35 token（组名+类型+定义），
        # 小上下文端点（如 max_model_len=3072 的 vLLM 部署）需调小（如 60）
        df = group_tags_via_llm(
            tags=genome_tags["tag"].tolist(),
            client=make_client(config),
            cache=cache,
            model=config.model.model,
            batch_size=batch_size)
        id_map = dict(zip(genome_tags["tag"], genome_tags["tag_id"]))
        rows = [{"tag_id": id_map[r["tag"]], **r}
                for _, r in df.iterrows()]
    paths.processed.mkdir(parents=True, exist_ok=True)
    # 大组拆分：组是 prompt 分批单位——组太大时 prompt+输出超出小上下文端点
    # （实测：99-tag 组在 max_model_len=3072 下输出被截断，15/48 任务 parse_failed）
    MAX_GROUP_TAGS = 60
    split_rows, group_seq = [], {}
    for r in rows:
        base = r["group"]
        n = group_seq.get(base, 0)
        sub = n // MAX_GROUP_TAGS
        if sub > 0:
            r = {**r, "group": f"{base}·{sub + 1}"}
        group_seq[base] = n + 1
        split_rows.append(r)
    paths.tag_groups.write_text(json.dumps(split_rows, ensure_ascii=False, indent=1),
                                encoding="utf-8")
    print(f"[03] tag 分组完成: {len(split_rows)} 个 tag, "
          f"{len({r['group'] for r in split_rows})} 个组"
          f"（大组已按 ≤{MAX_GROUP_TAGS} 拆分）→ {paths.tag_groups}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--batch-size", type=int, default=150,
                        help="每批分组标签数（小上下文端点需调小，如 60）")
    raise SystemExit(main(parser.parse_args().config, parser.parse_args().batch_size))
