"""新片清单（≥2015 帧+字幕）：选择谓词 + 防泄露 + 幂等写入。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genometag.eval.protocol import select_newfilm_ids, write_movie_list  # noqa: E402


def _dataset(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["movie_id", "title", "year", "keyframes",
                                       "script_path", "subtitles_path"])


def _genome(movie_ids: list[int]) -> pd.DataFrame:
    return pd.DataFrame({"movie_id": mid, "tag_id": 1, "relevance": 0.9}
                        for mid in movie_ids)


def _rows() -> list[dict]:
    return [
        # 2014：年份不足，排除
        dict(movie_id=1, title="old", year=2014, keyframes=["k"],
             script_path=None, subtitles_path="s"),
        # 2015-01 边界：含入
        dict(movie_id=2, title="edge", year=2015, keyframes=["k"],
             script_path=None, subtitles_path="s"),
        # 2018：含入
        dict(movie_id=3, title="late", year=2018, keyframes=["k"],
             script_path="scr", subtitles_path="s"),
        # keyframes 空列表：排除
        dict(movie_id=4, title="noframes", year=2016, keyframes=[],
             script_path=None, subtitles_path="s"),
        # keyframes 缺失（None/NaN）：排除
        dict(movie_id=5, title="nullframes", year=2016, keyframes=None,
             script_path=None, subtitles_path="s"),
        # 无 genome 覆盖：排除
        dict(movie_id=6, title="nogenome", year=2016, keyframes=["k"],
             script_path=None, subtitles_path="s"),
        # 满足条件但已在 88 部测试集：排除（防泄露）
        dict(movie_id=7, title="leaked", year=2017, keyframes=["k"],
             script_path="scr", subtitles_path="s"),
        # 2017 三样齐全：含入
        dict(movie_id=8, title="full", year=2017, keyframes=["k"],
             script_path="scr", subtitles_path="s"),
    ]


def test_select_filters_year_modality_genome_and_leakage():
    ds = _dataset(_rows())
    genome = _genome([2, 3, 8])
    ids = select_newfilm_ids(ds, genome, eval_ids=[7])
    assert ids == [2, 3, 8]


def test_select_output_sorted_and_deduped():
    ds = _dataset(_rows() + [dict(movie_id=2, title="dup", year=2016,
                                  keyframes=["k"], script_path=None,
                                  subtitles_path="s")])
    genome = _genome([2, 3, 8])
    ids = select_newfilm_ids(ds, genome, eval_ids=[])
    assert ids == sorted(set(ids))


def test_select_accepts_ndarray_keyframes():
    """服务器 parquet 读回的 keyframes 是 numpy.ndarray（本地测试用 list 会漏）。"""
    ds = _dataset(_rows())
    ds["keyframes"] = ds["keyframes"].apply(
        lambda k: np.array(k) if isinstance(k, list) else k)
    assert select_newfilm_ids(ds, _genome([2, 3, 8]), eval_ids=[7]) == [2, 3, 8]


def test_select_missing_genome_column_not_required():
    """genome_scores 只需 movie_id 列即可判覆盖。"""
    ds = _dataset(_rows())
    genome = pd.DataFrame({"movie_id": [2, 3, 8]})
    assert select_newfilm_ids(ds, genome, eval_ids=[]) == [2, 3, 8]


def test_select_requires_eval_list():
    """防泄露依赖 eval_test.txt：缺失必须报错，不得静默当空。"""
    ds = _dataset(_rows())
    with pytest.raises(ValueError):
        select_newfilm_ids(ds, _genome([2]), eval_ids=None)


def test_write_movie_list_idempotent(tmp_path):
    out = tmp_path / "newfilm2015.txt"
    assert write_movie_list(out, [3, 2, 2]) == "written"
    assert out.read_text(encoding="utf-8") == "2\n3\n"
    assert write_movie_list(out, [9, 9]) == "exists"
    assert out.read_text(encoding="utf-8") == "2\n3\n"
