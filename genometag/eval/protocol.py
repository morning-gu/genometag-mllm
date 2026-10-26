"""评估协议：评估集清单、拟合池、时序分割、冷启动子集、tag 类型分解、防泄露断言。

防泄露是全框架约束：所有拟合组件（mean/genre/BERT 基线、few-shot）
必须以 assert_no_leakage 收尾自检。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def load_eval_movie_ids(config, paths) -> list[int] | None:
    """data.movie_list：显式评估集清单（协议 v3 电影级留出）。

    文件每行一个 movie_id，路径相对项目根；未设置返回 None（走时序/抽样语义）。
    """
    if not getattr(config.data, "movie_list", None):
        return None
    p = paths.root / config.data.movie_list
    if not p.exists():
        raise ValueError(f"data.movie_list 文件不存在: {p}")
    ids = [int(line) for line in p.read_text(encoding="utf-8").split() if line.strip()]
    if not ids:
        raise ValueError(f"评估集清单为空: {p}")
    return sorted(set(ids))


def fitting_pool_ids(config, dataset: pd.DataFrame,
                     eval_ids: list[int] | None) -> list[int]:
    """拟合池（防泄露公共语义，04 few-shot 与 05 基线共用）：
    - movie_list 模式（协议 v3）：评估集的补集（含部分模态电影——基线只用统计，模态无要求）
    - 时序模式（默认/遗留）：year < temporal_split_year
    同时校验清单 id ⊆ dataset。"""
    all_ids = set(dataset["movie_id"])
    if eval_ids is not None:
        missing = [i for i in eval_ids if i not in all_ids]
        if missing:
            raise ValueError(f"评估集清单含未知 movie_id: {missing[:5]}（共 {len(missing)} 个）")
        return sorted(all_ids - set(eval_ids))
    train_ids, _ = temporal_split(dataset, config.protocol.temporal_split_year)
    return sorted(train_ids)


def temporal_split(dataset: pd.DataFrame, year: int) -> tuple[pd.Index, pd.Index]:
    train = dataset.loc[dataset["year"] < year, "movie_id"]
    test = dataset.loc[dataset["year"] >= year, "movie_id"]
    return pd.Index(train), pd.Index(test)


def cold_start_subset(dataset: pd.DataFrame, user_tags: pd.DataFrame,
                      max_tags: int) -> pd.Index:
    counts = user_tags.groupby("movie_id").size()
    n = dataset["movie_id"].map(lambda m: counts.get(m, 0))
    return pd.Index(dataset.loc[n <= max_tags, "movie_id"])


def tag_type_breakdown(tag_groups: pd.DataFrame) -> dict[str, list[int]]:
    return {t: g["tag_id"].tolist() for t, g in tag_groups.groupby("tag_type")}


def assert_no_leakage(fit_movie_ids, test_movie_ids) -> None:
    leak = set(fit_movie_ids) & set(test_movie_ids)
    if leak:
        raise ValueError(
            f"拟合池与测试集重叠（数据泄露）: {sorted(leak)[:5]}... "
            "拟合组件只允许使用评估集之外的数据（movie_list 补集或 temporal_split_year 之前）")


def select_newfilm_ids(dataset: pd.DataFrame, genome_scores: pd.DataFrame,
                       eval_ids: list[int] | None,
                       year_min: int = 2015) -> list[int]:
    """新片评估集：year ≥ year_min 且帧+字幕可用（剧本不要求），须有 genome 覆盖，
    并强制排除既有评估集（防泄露）。eval_ids 传 None 视为调用方未加载清单——拒绝。"""
    if eval_ids is None:
        raise ValueError("eval_ids 未提供：防泄露排除需要 eval_test 清单")
    n_frames = dataset["keyframes"].apply(
        lambda k: len(k) if isinstance(k, (list, tuple, np.ndarray)) else 0)
    covered = set(genome_scores["movie_id"])
    mask = ((dataset["year"] >= year_min) & (n_frames > 0)
            & dataset["subtitles_path"].notna()
            & dataset["movie_id"].isin(covered)
            & ~dataset["movie_id"].isin(set(eval_ids)))
    return sorted(set(dataset.loc[mask, "movie_id"].astype(int)))


def write_movie_list(path, ids: list[int]) -> str:
    """清单落盘（升序去重，每行一个 movie_id）。已存在则跳过（幂等），
    返回 "written" / "exists"。"""
    if path.exists():
        return "exists"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = sorted({int(i) for i in ids})
    path.write_text("\n".join(str(i) for i in lines) + "\n", encoding="utf-8")
    return "written"
