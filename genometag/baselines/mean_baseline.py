"""全局均值基线：每 tag 在训练参考集（分割年前）上的 relevance 均值。"""
from __future__ import annotations

import pandas as pd

from genometag.eval.protocol import assert_no_leakage


def mean_baseline(train_scores: pd.DataFrame, movie_ids: list[int],
                  test_ids: set[int] | None = None) -> pd.DataFrame:
    if test_ids is not None:  # 防泄露自检：拟合数据不得含测试电影
        assert_no_leakage(set(train_scores["movie_id"]), test_ids)
    per_tag = train_scores.groupby("tag_id")["relevance"].mean()
    rows = [{"movie_id": m, "tag_id": t, "score": float(s), "method": "mean"}
            for m in movie_ids for t, s in per_tag.items()]
    return pd.DataFrame(rows, columns=["movie_id", "tag_id", "score", "method"])
