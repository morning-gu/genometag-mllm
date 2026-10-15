# GenomeTag-MLLM

MLLM 从电影多模态内容（关键帧 + 剧本 + 台词）预测 MovieLens genome tag 关联度评分。
研究问题、章节计划与文献定位见 `docs/research-plan.md`；框架设计见
`docs/superpowers/specs/`。

## 安装

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
# 可选（BERT 基线）：
.venv/Scripts/python -m pip install -r requirements-baselines.txt
cp .env.example .env   # 填入 OPENAI_API_KEY / OPENAI_BASE_URL
```

## 验证安装（零成本）

```bash
.venv/Scripts/python -m pytest tests/ -v      # 全部测试：无网络、无 key
.venv/Scripts/python scripts/run_smoke.py     # 端到端冒烟（合成样本 + mock）
```

## 真实数据流程

1. 按 `docs/download-guide.md` 下载数据到 `data/raw/`
2. `.venv/Scripts/python scripts/01_verify_data.py`（检查文件与**年份分布**）
3. `.venv/Scripts/python scripts/02_build_dataset.py` → 检查 `data/interim/align_report.md`
4. `.venv/Scripts/python scripts/03_group_tags.py`（一次性，产物入 git）
5. 小规模试跑：复制 `configs/default.yaml`，设 `data.movie_limit: 50`，
   `.venv/Scripts/python scripts/04_run_annotation.py --config <你的配置>`
6. `.venv/Scripts/python scripts/05_run_baselines.py --config configs/baselines.yaml`
7. `.venv/Scripts/python scripts/06_evaluate.py --config <你的配置>`

## 消融实验

`configs/ablation/` 下五个配置仅 `modalities` 不同；逐个跑 04，
最后 `06_evaluate.py --config configs/ablation/full.yaml --extra ablation-video-only ablation-script-only ...` 生成对比表与热力图。

## 一致性实验

复制 `configs/default.yaml`，设 `annotation.n_runs: 5`、`data.movie_limit: 100`
（即 spec §5.5 的 consistency 参数），`exp_name: consistency` 另存。
跑 04 → 06，评估目录自动产出 `consistency.json`（ICC(2,1) 与平均轮次标准差）。

## 目录约定与产物 schema

见 `docs/data-format.md`。API 调用缓存在 `.cache/llm/`（重跑零成本、可断点续跑）。
