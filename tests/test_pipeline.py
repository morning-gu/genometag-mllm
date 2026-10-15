"""标注管道：矩阵、few-shot 防泄露、断点续跑、解析失败降级、空模态跳过、dry-run。"""
import json

import pandas as pd
import pytest

from genometag.annotate.mock import MockClient
from genometag.annotate.pipeline import (
    build_task_matrix, run_annotation, select_few_shots,
)
from genometag.config import Paths, load_config
from genometag.data.align import build_dataset
from genometag.data.sample import make_synthetic_dataset


def _paths_for(tmp_path) -> Paths:
    """指向 tmp_path 的 Paths（同 tests/test_align.py 的 _synthetic_paths 模式）。"""
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
    """临时目录内搭建完整合成数据集（封闭测试，不碰仓库全局 data/synthetic）。"""
    make_synthetic_dataset(tmp_path)
    paths = _paths_for(tmp_path)
    build_dataset(paths)                      # 产出 dataset/genome_scores/tag_groups
    cfg = load_config("configs/smoke.yaml")   # n_runs=2, 三模态, few_shot=1
    yield cfg, paths


def test_build_task_matrix():
    tg = pd.DataFrame([
        {"tag_id": 1, "tag": "a", "group": "G1", "tag_type": "content", "definition": "d"},
        {"tag_id": 2, "tag": "b", "group": "G1", "tag_type": "content", "definition": "d"},
        {"tag_id": 3, "tag": "c", "group": "G2", "tag_type": "preference", "definition": "d"},
    ])
    tasks = build_task_matrix([10, 11], tg, n_runs=2)
    assert len(tasks) == 2 * 2 * 2            # 电影 × 组 × 轮
    assert {t.run_id for t in tasks} == {"run-0", "run-1"}


def test_select_few_shots_train_only():
    """Review Focus #4：few-shot 只能来自分割年之前。"""
    dataset = pd.DataFrame({"movie_id": [1, 2, 3], "title": ["A", "B", "C"],
                            "year": [2020, 2021, 2023]})
    scores = pd.DataFrame({"movie_id": [1, 2, 3, 1, 2, 3],
                           "tag_id": [7, 7, 7, 8, 8, 8],
                           "relevance": [0.9, 0.1, 0.99, 0.8, 0.2, 0.98]})
    shots = select_few_shots(dataset, scores, n=2, split_year=2022, top_k=2,
                             tag_names={7: "t7", 8: "t8"})
    assert len(shots) <= 2
    assert all(s.year < 2022 for s in shots)   # 2023 的电影绝不能进 few-shot
    assert shots[0].title == "A"               # top-k 分数和：A(1.7) > B(0.3)
    assert set(shots[0].scores) <= {"t7", "t8"}  # 键是 tag 文本而非 id


def test_run_annotation_end_to_end(synthetic_env):
    cfg, paths = synthetic_env
    out = run_annotation(cfg, paths, MockClient(), cache=None)
    df = pd.read_parquet(out)
    # 5 电影 × 24 tag × 2 轮 = 240 行（若 no_content 则少；合成数据三模态俱全 → 全量）
    assert len(df) == 5 * 24 * 2
    assert set(df.columns) == {"movie_id", "tag_id", "group", "modality_set", "run_id", "score"}
    assert df["score"].notna().all()
    assert (paths.results / "cost_report.json").exists()


def test_run_annotation_resume_skips_done(synthetic_env):
    """Review Focus（续跑不重复调用）：删一半 jsonl 行后重跑，client 调用数只补缺失。"""
    cfg, paths = synthetic_env
    calls = {"n": 0}

    class CountingMock(MockClient):
        def complete(self, messages, run_id):
            calls["n"] += 1
            return super().complete(messages, run_id)

    out = run_annotation(cfg, paths, CountingMock(), cache=None)
    full_calls = calls["n"]
    # 截断 jsonl 到一半 → 重跑只补后半
    jsonl = paths.results / "annotations.jsonl"
    lines = jsonl.read_text(encoding="utf-8").splitlines()
    jsonl.write_text("\n".join(lines[: len(lines) // 2]) + "\n", encoding="utf-8")
    calls["n"] = 0
    out2 = run_annotation(cfg, paths, CountingMock(), cache=None)
    df = pd.read_parquet(out2)
    assert len(df) == 5 * 24 * 2                  # 结果完整
    assert 0 < calls["n"] < full_calls            # 只补了缺失部分


def test_run_annotation_tolerates_malformed_jsonl_line(synthetic_env, capsys):
    """终审 I3：崩溃残留的截断 jsonl 行被跳过并计数，续跑不崩、结果完整。"""
    cfg, paths = synthetic_env
    run_annotation(cfg, paths, MockClient(), cache=None)
    jsonl = paths.results / "annotations.jsonl"
    with jsonl.open("a", encoding="utf-8") as f:
        f.write('{"task_key": "999|G|run-0", "rows": [{"movie_id"')  # 模拟写一半崩溃
    out2 = run_annotation(cfg, paths, MockClient(), cache=None)
    df = pd.read_parquet(out2)
    assert len(df) == 5 * 24 * 2            # 坏行未影响结果完整性
    out_text = capsys.readouterr().out
    assert "跳过损坏的 jsonl 行: 1" in out_text


def test_parse_failure_records_null(synthetic_env):
    """Review Focus #2：解析重试后仍失败 → null 分数 + manifest 记 parse_failed。"""
    cfg, paths = synthetic_env

    class BadMock(MockClient):
        def complete(self, messages, run_id):
            resp = super().complete(messages, run_id)
            resp.content = "抱歉我无法输出 JSON"    # 永远不可解析
            return resp

    out = run_annotation(cfg, paths, BadMock(), cache=None)
    df = pd.read_parquet(out)
    assert df["score"].isna().all()
    manifest = json.loads((paths.results / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["tasks"]["parse_failed"] > 0
    assert manifest["tasks"]["done"] == 0


def test_no_content_task_skipped(synthetic_env):
    """Review Focus #5：电影无剧本且 modalities=[script] → no_content，不调用 API。"""
    cfg, paths = synthetic_env
    cfg.annotation.modalities = ["script"]
    dataset = pd.read_parquet(paths.dataset)
    # 让所有电影没有剧本
    dataset["script_path"] = None
    dataset.to_parquet(paths.dataset, index=False)

    class NoCallMock(MockClient):
        def complete(self, messages, run_id):
            raise AssertionError("no_content 任务不应调用 MLLM")

    out = run_annotation(cfg, paths, NoCallMock(), cache=None)
    df = pd.read_parquet(out)
    assert len(df) == 0
    manifest = json.loads((paths.results / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["tasks"]["no_content"] == 5 * 4 * 2    # 全部任务 no_content


def test_dry_run_prints_prompts_without_calls(synthetic_env, capsys):
    """spec §4.5 的 --dry-run：打印 prompt、不调用客户端、不写文件。"""
    cfg, paths = synthetic_env

    class NoCallMock(MockClient):
        def complete(self, messages, run_id):
            raise AssertionError("dry-run 不应调用 MLLM")

    out = run_annotation(cfg, paths, NoCallMock(), cache=None, dry_run=True)
    assert out is None
    assert not (paths.results / "annotations.jsonl").exists()
    assert not (paths.results / "annotations.parquet").exists()
    out_text = capsys.readouterr().out
    assert "[dry-run]" in out_text
    assert "电影：《" in out_text
