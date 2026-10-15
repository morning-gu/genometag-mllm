"""对齐：IMDb 规范化、主键匹配、翻拍歧义、报告内容。"""
import pandas as pd
import pytest

from genometag.data.align import AlignResult, align, normalize_imdb


def test_normalize_imdb():
    assert normalize_imdb("tt0123456") == 123456
    assert normalize_imdb("123456") == 123456
    assert normalize_imdb(123456) == 123456
    assert normalize_imdb(123456.0) == 123456     # pandas 读 int 列可能 float 化
    assert normalize_imdb("tt0000001") == 1
    assert normalize_imdb("junk") is None
    assert normalize_imdb(None) is None


def _movies_ml():
    return pd.DataFrame([
        {"movie_id": 1, "title": "Alpha", "year": 2001, "genres": ["Drama"]},
        {"movie_id": 2, "title": "Beta", "year": 2002, "genres": ["Action"]},
        {"movie_id": 3, "title": "Gamma", "year": 2003, "genres": ["Crime"]},
    ])


def _links():
    return pd.DataFrame([{"movie_id": 1, "imdb_id": 111},
                         {"movie_id": 2, "imdb_id": 222}])  # 3 无 IMDb


def test_align_imdb_primary():
    from genometag.data.movienet import MovieMultimodal
    mn = [MovieMultimodal("mnA", "Alpha", 2001, "tt0000111", [], None, None),
          MovieMultimodal("mnB", "Beta", 2002, "tt0000222", [], None, None)]
    result = align(_movies_ml(), _links(), mn)
    df = result.aligned
    assert set(df["match_method"]) == {"imdb"}
    assert list(df["movie_id"]) == [1, 2]
    assert "IMDb 2" in result.report            # 报告含匹配统计


def test_align_title_fallback_for_missing_imdb():
    from genometag.data.movienet import MovieMultimodal
    mn = [MovieMultimodal("mnC", "gamma ", 2003, None, [], None, None)]  # 大小写/空格差异
    result = align(_movies_ml(), _links(), mn)
    row = result.aligned[result.aligned["movie_id"] == 3]
    assert len(row) == 1
    assert row.iloc[0]["match_method"] == "title"


def test_ambiguous_remake_unmatched():
    """同名同年多部（翻拍撞年份）→ 不匹配并写进报告歧义清单。"""
    from genometag.data.movienet import MovieMultimodal
    mn = [MovieMultimodal("mnX", "Gamma", 2003, None, [], None, None),
          MovieMultimodal("mnY", "Gamma", 2003, None, [], None, None)]
    result = align(_movies_ml(), _links(), mn)
    assert result.aligned[result.aligned["movie_id"] == 3].empty
    assert "Gamma" in result.report


def _synthetic_paths(tmp_path):
    """为 tmp_path 构建指向合成数据的 Paths（Task 12 fixture 复用同一模式）。"""
    from genometag.config import Paths
    return Paths(
        root=tmp_path,
        movienet=tmp_path / "raw" / "movienet",
        movielens=tmp_path / "raw" / "movielens",
        interim=tmp_path / "interim", processed=tmp_path / "processed",
        cache=tmp_path / ".cache", results=tmp_path / "results" / "x",
        tag_groups=tmp_path / "processed" / "tag_groups.json",
        dataset=tmp_path / "processed" / "dataset.parquet",
        genome_scores=tmp_path / "processed" / "genome_scores.parquet",
        genome_tags=tmp_path / "processed" / "genome_tags.parquet",
        aligned=tmp_path / "interim" / "aligned.parquet")


def test_build_dataset_from_synthetic(tmp_path):
    from genometag.data.align import build_dataset
    from genometag.data.sample import make_synthetic_dataset
    make_synthetic_dataset(tmp_path)
    paths = _synthetic_paths(tmp_path)
    stats = build_dataset(paths)
    assert stats == {"movies": 5, "genome_rows": 120}
    ds = pd.read_parquet(paths.dataset)
    assert len(ds) == 5
    assert ds.loc[0, "keyframes"][0].startswith("keyframes")   # 相对路径
    assert (paths.interim / "align_report.md").exists()
