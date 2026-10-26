"""newfilm 双臂配置：值与 v3 模板一致（隔离人群变量），模态与 movie_list 正确。"""
from genometag.config import load_config


def test_newfilm_full_config():
    cfg = load_config("configs/newfilm/full.yaml")
    assert cfg.exp_name == "newfilm-full"
    assert cfg.data.movie_list == "data/processed/newfilm2015.txt"
    assert cfg.annotation.modalities == ["video", "dialogue"]
    assert cfg.model.model == "qwen3.7-plus"
    assert cfg.model.max_frames == 12
    assert cfg.model.script_token_budget == 4000
    assert cfg.model.dialogue_token_budget == 3000
    assert cfg.annotation.n_runs == 3
    assert cfg.annotation.few_shot_examples == 2
    assert cfg.annotation.workers == 8
    assert cfg.protocol.top_k == 10
    assert cfg.protocol.relevance_threshold == 0.5


def test_newfilm_title_config():
    cfg = load_config("configs/newfilm/title.yaml")
    assert cfg.exp_name == "newfilm-title"
    assert cfg.data.movie_list == "data/processed/newfilm2015.txt"
    assert cfg.annotation.modalities == []
    assert cfg.model.model == "qwen3.7-plus"
    assert cfg.annotation.n_runs == 3
    assert cfg.annotation.workers == 8
