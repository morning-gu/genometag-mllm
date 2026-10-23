"""错配探针：采样条件正确性 + mock 端到端（mock 只看标题 = 天然"记忆者"，探针应检出）。"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genometag.annotate.mock import MockClient  # noqa: E402
from genometag.config import Paths, load_config  # noqa: E402
from genometag.data.align import build_dataset  # noqa: E402
from genometag.data.sample import make_synthetic_dataset  # noqa: E402
from genometag.probes.mismatch import run_mismatch_probe, sample_pairs  # noqa: E402


def _paths_for(tmp_path) -> Paths:
    return Paths(
        root=tmp_path,
        movienet=tmp_path / "raw" / "movienet",
        movielens=tmp_path / "raw" / "movielens",
        interim=tmp_path / "interim", processed=tmp_path / "processed",
        cache=tmp_path / ".cache", results=tmp_path / "results" / "smoke",
        tag_groups=tmp_path / "processed" / "tag_groups.json",
        dataset=tmp_path / "processed" / "dataset.parquet",
        genome_scores=tmp_path / "processed" / "genome_scores.parquet",
        genome_tags=tmp_path / "processed" / "genome_tags.parquet",
        aligned=tmp_path / "interim" / "aligned.parquet")


@pytest.fixture()
def synthetic_env(tmp_path):
    make_synthetic_dataset(tmp_path)
    paths = _paths_for(tmp_path)
    build_dataset(paths)
    cfg = load_config("configs/smoke.yaml")
    yield cfg, paths


def test_sample_pairs_conditions(synthetic_env):
    """同/跨类型各半，类型关系与采样约束成立。"""
    _, paths = synthetic_env
    dataset = pd.read_parquet(paths.dataset)
    pairs = sample_pairs(dataset, n_pairs=2, seed=42,
                         modalities=["video", "script", "dialogue"])
    assert len(pairs) == 2
    assert {p.condition for p in pairs} == {"same_genre", "cross_genre"}
    genres = dict(zip(dataset["movie_id"], dataset["genres"]))
    for p in pairs:
        share = bool(set(genres[p.movie_a]) & set(genres[p.movie_b]))
        assert share == (p.condition == "same_genre")
        assert p.movie_a != p.movie_b


def test_run_mismatch_probe_parallel_equivalent(synthetic_env):
    """并行（workers=3）与串行等价：行集与 corr 汇总一致（mock 确定性）。"""
    cfg, paths = synthetic_env
    cfg.probes.n_pairs = 2
    out_seq = run_mismatch_probe(cfg, paths, MockClient(), cache=None)
    df_seq = pd.read_parquet(out_seq)
    s_seq = json.loads((paths.results / "mismatch_summary.json").read_text(encoding="utf-8"))

    paths.results = paths.results.with_name("smoke-par")
    cfg.annotation.workers = 3
    out_par = run_mismatch_probe(cfg, paths, MockClient(), cache=None)
    df_par = pd.read_parquet(out_par)
    s_par = json.loads((paths.results / "mismatch_summary.json").read_text(encoding="utf-8"))

    key = ["pair_id", "tag_id"]
    pd.testing.assert_frame_equal(
        df_seq.sort_values(key).reset_index(drop=True),
        df_par.sort_values(key).reset_index(drop=True))
    for cond in ("same_genre", "cross_genre"):
        assert s_par[cond]["mean_corr_a"] == pytest.approx(s_seq[cond]["mean_corr_a"])
        assert s_par[cond]["mean_corr_b"] == pytest.approx(s_seq[cond]["mean_corr_b"])


def test_run_mismatch_probe_mock_detects_memorization(synthetic_env):
    """端到端：mock 打分仅由（A 的）标题驱动 → 预测与 truth_A 相关、与 truth_B 无关，
    即 mock 表现为"记忆者"——探针的判别方向必须检出这一点。"""
    cfg, paths = synthetic_env
    cfg.probes.n_pairs = 2
    out = run_mismatch_probe(cfg, paths, MockClient(), cache=None)

    df = pd.read_parquet(out)
    assert list(df.columns) == ["pair_id", "movie_a", "movie_b",
                                "condition", "tag_id", "score"]
    assert df["score"].notna().all()
    assert set(df["condition"]) == {"same_genre", "cross_genre"}

    summary = json.loads((paths.results / "mismatch_summary.json")
                         .read_text(encoding="utf-8"))
    assert summary["same_genre"]["n_pairs"] == 1
    assert summary["cross_genre"]["n_pairs"] == 1
    # 判别方向：记忆者（标题驱动）→ corr_a > corr_b，shift < 0
    for cond in ("same_genre", "cross_genre"):
        assert summary[cond]["mean_corr_a"] > summary[cond]["mean_corr_b"]
    assert "interpretation" in summary
