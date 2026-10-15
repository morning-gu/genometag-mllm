"""报告：指标表数值与 metrics 直算一致、子集过滤、热力图产出。"""
import numpy as np
import pandas as pd
import pytest

from genometag.config import ProtocolConfig
from genometag.eval.report import metrics_table, modality_heatmap


def _data():
    dataset = pd.DataFrame({"movie_id": [1, 2], "year": [2020, 2023],
                            "genres": [["A"], ["A"]]})
    truth = pd.DataFrame({"movie_id": [1, 1, 2, 2], "tag_id": [1, 2, 1, 2],
                          "relevance": [0.9, 0.1, 0.8, 0.2]})
    perfect = truth.rename(columns={"relevance": "score"}).assign(method="perfect")
    wrong = truth.assign(score=1.0 - truth["relevance"]).assign(method="wrong")
    return dataset, truth, perfect, wrong


def test_metrics_table_matches_direct_computation():
    dataset, truth, perfect, wrong = _data()
    cfg = ProtocolConfig(top_k=2, relevance_threshold=0.5)
    table = metrics_table({"perfect": perfect[["movie_id", "tag_id", "score"]],
                           "wrong": wrong[["movie_id", "tag_id", "score"]]},
                          truth, dataset, cfg, tag_groups=None)
    row = table[table["method"] == "perfect"].iloc[0]
    assert row["rmse"] == pytest.approx(0.0)
    assert row["ndcg@k"] == pytest.approx(1.0)
    row_w = table[table["method"] == "wrong"].iloc[0]
    assert row_w["rmse"] > 0.5


def test_metrics_table_movie_subset():
    """限定电影集合（如仅被标注的子集）时 truth 也被过滤。"""
    dataset, truth, perfect, wrong = _data()
    cfg = ProtocolConfig(top_k=2)
    table = metrics_table({"wrong": wrong[["movie_id", "tag_id", "score"]]},
                          truth, dataset, cfg, tag_groups=None, movie_ids=[2])
    assert len(table) == 1
    # movie-2-only mae = 0.6（全集为 0.7）——真正钉住 movie_ids 过滤
    assert table["mae"].iloc[0] == pytest.approx(0.6)


def test_modality_heatmap_writes_png(tmp_path):
    rows = pd.DataFrame({
        "modality_set": ["video", "script", "video", "script"],
        "tag_type": ["content", "content", "preference", "preference"],
        "ndcg": [0.8, 0.6, 0.5, 0.4]})
    out = tmp_path / "heat.png"
    modality_heatmap(rows, out)
    assert out.exists() and out.stat().st_size > 0


def test_write_report_and_consistency(tmp_path):
    """write_report 主输出路径：指标表 + 报告 + 一致性分析（两轮 → ICC）。"""
    import json as _json

    from genometag.eval.report import write_report

    dataset = pd.DataFrame({"movie_id": [1, 2], "year": [2020, 2023],
                            "genres": [["A"], ["A"]]})
    truth = pd.DataFrame({"movie_id": [1, 1, 2, 2], "tag_id": [1, 2, 1, 2],
                          "relevance": [0.9, 0.1, 0.8, 0.2]})
    # 两轮标注，run-1 比 run-0 恒定高 0.05（系统性的评分者偏差）
    rows = []
    for run in ("run-0", "run-1"):
        for _, r in truth.iterrows():
            rows.append({"movie_id": r["movie_id"], "tag_id": r["tag_id"],
                         "group": "G", "modality_set": "video|script",
                         "run_id": run,
                         "score": float(r["relevance"]) + (0.05 if run == "run-1" else 0.0)})
    exp = tmp_path / "exp"
    exp.mkdir()
    pd.DataFrame(rows).to_parquet(exp / "annotations.parquet", index=False)

    out = write_report([exp], None, tmp_path / "evaluation", ProtocolConfig(top_k=2),
                       dataset, truth, tag_groups=None, user_tags=None)
    assert (out / "metrics.csv").exists()
    assert (out / "report.md").exists()
    assert (out / "consistency.json").exists()       # 两轮 → 一致性分析产出
    consistency = _json.loads((out / "consistency.json").read_text(encoding="utf-8"))
    assert consistency["n_runs"] == 2
    assert consistency["icc_2_1"] > 0.95            # 恒定偏移下 ICC 接近 1
    assert consistency["mean_std"] == pytest.approx(0.025)  # 每项两轮差 0.05 → std=0.025
    # 单一实验 → 无热力图
    assert not (out / "heatmap.png").exists()


def test_write_report_train_and_tag_type_subsets(tmp_path):
    """终审 I2：metrics.csv 必含 时序训练(<y) 与 tag:<类型> 视图——单实验也产出，
    不依赖热力图路径（content vs preference 对比是论文核心论证）。"""
    from genometag.eval.report import write_report

    dataset = pd.DataFrame({"movie_id": [1, 2], "year": [2020, 2023],
                            "genres": [["A"], ["A"]]})
    truth = pd.DataFrame({"movie_id": [1, 1, 2, 2], "tag_id": [1, 2, 1, 2],
                          "relevance": [0.9, 0.1, 0.8, 0.2]})
    rows = [{"movie_id": r["movie_id"], "tag_id": r["tag_id"],
             "group": "G", "modality_set": "video|script", "run_id": run,
             "score": float(r["relevance"])}
            for run in ("run-0", "run-1") for _, r in truth.iterrows()]
    exp = tmp_path / "exp"
    exp.mkdir()
    pd.DataFrame(rows).to_parquet(exp / "annotations.parquet", index=False)
    tag_groups = pd.DataFrame([
        {"tag_id": 1, "tag": "visual", "group": "G1", "tag_type": "content",
         "definition": "d"},
        {"tag_id": 2, "tag": "overrated", "group": "G2", "tag_type": "preference",
         "definition": "d"},
    ])

    out = write_report([exp], None, tmp_path / "evaluation", ProtocolConfig(top_k=2),
                       dataset, truth, tag_groups=tag_groups, user_tags=None)
    table = pd.read_csv(out / "metrics.csv")
    subsets = set(table["subset"])
    assert "全部" in subsets
    assert "时序训练(<2022)" in subsets
    assert "时序测试(≥2022)" in subsets
    assert "tag:content" in subsets
    assert "tag:preference" in subsets
    # 每个视图下方法集合与全集一致（预测未被过滤丢失）
    all_methods = set(table[table["subset"] == "全部"]["method"])
    for sub in ("tag:content", "tag:preference", "时序训练(<2022)"):
        assert set(table[table["subset"] == sub]["method"]) == all_methods
