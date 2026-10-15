"""全部实验配置可加载且合法。"""
from pathlib import Path

import pytest

from genometag.config import load_config

CONFIG_FILES = sorted(
    str(p) for p in Path("configs").rglob("*.yaml"))


def test_configs_found():
    assert len(CONFIG_FILES) >= 8   # default + smoke + baselines + 5 ablation


@pytest.mark.parametrize("path", CONFIG_FILES)
def test_config_loads(path):
    cfg = load_config(path)
    assert cfg.exp_name
