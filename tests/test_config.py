"""配置加载、校验、路径推导。"""
from pathlib import Path

import pytest
import yaml

from genometag.config import load_config, paths_for, validate_config


def _write(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return p


MINIMAL = {
    "exp_name": "t",
    "data": {"source": "synthetic"},
    "annotation": {"modalities": ["video", "script"]},
}


def test_load_minimal_config(tmp_path):
    cfg = load_config(_write(tmp_path, MINIMAL))
    assert cfg.exp_name == "t"
    assert cfg.data.source == "synthetic"
    assert cfg.annotation.n_runs == 3          # 默认值
    assert cfg.protocol.temporal_split_year == 2022


def test_invalid_modality_rejected(tmp_path):
    bad = {**MINIMAL, "annotation": {"modalities": ["video", "audio"]}}
    with pytest.raises(ValueError, match="模态"):
        load_config(_write(tmp_path, bad))


def test_invalid_source_rejected(tmp_path):
    bad = {**MINIMAL, "data": {"source": "imaginary"}}
    with pytest.raises(ValueError, match="data.source"):
        load_config(_write(tmp_path, bad))


def test_paths_synthetic_isolated(tmp_path):
    cfg = load_config(_write(tmp_path, MINIMAL))
    p = paths_for(cfg)
    assert p.movienet == p.root / "data" / "synthetic" / "raw" / "movienet"
    assert p.movielens == p.root / "data" / "synthetic" / "raw" / "movielens"
    assert p.processed == p.root / "data" / "synthetic" / "processed"
    assert p.tag_groups == p.processed / "tag_groups.json"
    assert p.results == p.root / "results" / "t"


def test_paths_real(tmp_path):
    cfg = load_config(_write(tmp_path, {**MINIMAL, "data": {"source": "real"}}))
    p = paths_for(cfg)
    assert p.movielens == p.root / "data" / "raw" / "movielens" / "ml-latest"
    assert p.processed == p.root / "data" / "processed"


def test_validate_rejects_bad_threshold():
    from genometag.config import AppConfig, ProtocolConfig
    bad = AppConfig(protocol=ProtocolConfig(relevance_threshold=1.5))
    with pytest.raises(ValueError, match="relevance_threshold"):
        validate_config(bad)


def test_unknown_top_level_key_rejected(tmp_path):
    """终审 I4：未知顶层键必须报错，不得静默忽略（consistency 曾是被无视的死配置）。"""
    bad = {**MINIMAL, "consistency": {"n_runs": 5}}
    with pytest.raises(ValueError, match="consistency"):
        load_config(_write(tmp_path, bad))


def test_unknown_top_level_key_message_mentions_replacement(tmp_path):
    """报错信息要指出 consistency 块的替代写法。"""
    bad = {**MINIMAL, "consistency": {"n_runs": 5}}
    with pytest.raises(ValueError, match="annotation.n_runs"):
        load_config(_write(tmp_path, bad))


def test_workers_validated(tmp_path):
    """annotation.workers 并行线程数：必须 ≥ 1。"""
    bad = {**MINIMAL, "annotation": {"workers": 0}}
    with pytest.raises(ValueError, match="workers"):
        load_config(_write(tmp_path, bad))


def test_extra_body_type_validated(tmp_path):
    """extra_body 必须是字典（端点专有参数透传，如 DashScope enable_thinking）。"""
    bad = {**MINIMAL, "model": {"extra_body": ["not", "a", "dict"]}}
    with pytest.raises(ValueError, match="extra_body"):
        load_config(_write(tmp_path, bad))


def test_extra_body_loads_from_yaml(tmp_path):
    cfg_data = {**MINIMAL, "model": {"extra_body": {"enable_thinking": False}}}
    cfg = load_config(_write(tmp_path, cfg_data))
    assert cfg.model.extra_body == {"enable_thinking": False}


def test_default_yaml_still_loads():
    """default.yaml 的 consistency 块移除后仍可加载且无该字段。"""
    cfg = load_config("configs/default.yaml")
    assert cfg.exp_name == "default"
    assert not hasattr(cfg, "consistency")
