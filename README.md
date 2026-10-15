# GenomeTag-MLLM

MLLM 从电影多模态内容（关键帧+剧本+台词）预测 MovieLens genome tag 关联度。
设计文档：`docs/superpowers/specs/`；实现计划：`docs/superpowers/plans/`。

## 快速开始

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest tests/ -v          # 全部测试（无网络无 key）
.venv/Scripts/python scripts/run_smoke.py         # 端到端冒烟（合成样本 + mock）
```
