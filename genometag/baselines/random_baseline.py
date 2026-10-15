"""随机基线：固定种子均匀采样 [0,1)。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def random_baseline(movie_ids: list[int], tag_ids: list[int], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [{"movie_id": m, "tag_id": t, "score": float(rng.uniform()), "method": "random"}
            for m in movie_ids for t in tag_ids]
    return pd.DataFrame(rows, columns=["movie_id", "tag_id", "score", "method"])
