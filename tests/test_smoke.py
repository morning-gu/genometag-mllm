"""端到端冒烟：合成样本 + mock，零网络零成本。"""
from scripts.run_smoke import main


def test_smoke_pipeline():
    assert main() == 0
