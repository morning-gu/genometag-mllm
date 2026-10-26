# ≥2015 新片双臂 + 剂量-反应分析 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 ≥2015 新片子集（帧+字幕）上运行 full(2模态) vs title-only 双臂，并按"先验强度剂量-反应"分析内容净贡献，检验错配探针的可证伪预测（先验越弱内容增量越大）。

**Architecture:** 复用协议 v3 的 movie_list 机制（`load_eval_movie_ids`/`fitting_pool_ids`，genometag/eval/protocol.py:11-40）——新增清单文件 `data/processed/newfilm2015.txt` + 两个配置，`04`/`06`/`chain.sh`/`watchdog.sh` 零改动。分析侧新增逐片 NDCG 纯函数模块与 `10_dose_response.py`（v3 主实验未落盘逐片指标，需从 annotations.parquet 重算）。

**Tech Stack:** Python 3.12 / pandas / numpy / matplotlib；pytest；服务器 nohup + chain.sh。

**Spec:** 设计决策来自 2026-10-26 会话（方案 A+B 打包：先测先验条件化解读 + 剂量-反应；方案 C 匿名标题臂暂缓）；框架背景见 `docs/superpowers/specs/2026-10-15-genometag-mllm-project-design.md` 与协议 v3（`6811c4a`+`3d45c77`）。

## Global Constraints

- 本地一律用 `.venv/Scripts/python.exe`（全局 Python 被污染：torch 损坏 + numpy 2.5.4）
- 模型 qwen3.7-plus，thinking 默认开；预算 `max_frames: 12 / script_token_budget: 4000 / dialogue_token_budget: 3000`
- `n_runs: 3`、`few_shot_examples: 2`、`workers: 8`、`protocol.top_k: 10`、`relevance_threshold: 0.5`（与 v3 完全一致，隔离人群变量）
- 防泄露铁律：newfilm2015 清单必须排除 `eval_test.txt` 的 88 部（`assert_no_leakage` 语义）
- 模态命名固定为 `video` / `script` / `dialogue`（configs/ablation/full.yaml:11）
- 结果目录由 `exp_name` 决定（`results/<exp_name>/`），annotations.jsonl 支持断点续跑，不得清空重跑
- 服务器操作：先 `source server-env.sh`（SSL 修复）；nohup 必须 `< /dev/null`；`pkill -f` 用 `[x]` 括号技巧
- 少样本（few-shot）保持拟合池 top-2（预期仍为 Goodfellas/Terminator），任务 1 落盘清单时打印诊断确认

## Review Focus

1. **部分-整体相关陷阱**：Δ(full−title) 对 title 回归存在机械负斜率（若 full⊥title，斜率期望 = −1）——主检验必须是 full~title 回归斜率 vs 1（冗余零假设 b=1；内容价值预测 b<1 且截距>0），Δ 仅作描述。测试钉在任务 3（合成斜率恢复用例：b<1 数据必须检出）。
2. **部分标签无 genome 真值**：某片某组标签 truth 缺失时，期望该片在"有真值的标签"上照常评估（与 06 官方 inner-join 语义一致），不得整片丢弃或 NaN 污染。测试钉在任务 3（手工小样例含缺失真值标签）。
3. **清单边界**：恰好 2015 年、keyframes 为 NaN 或空列表、script 缺失但帧+字幕齐——期望：2015 含入、NaN/空帧排除、无剧本含入。测试钉在任务 1。
4. **与 88 部测试集重叠**：≥2015 三模态全有的至多 1 部可能已在 eval_test.txt——期望被排除且诊断可见。测试钉在任务 1。
5. **每片 n_runs 平均时机**：逐片 NDCG 必须先对 3 轮 score 求均值再算指标（镜像 report.py:119 语义），不得逐轮算再平均。测试钉在任务 3（与 write_report 聚合值一致性用例）。

---

### Task 1: 新片清单生成器 `scripts/09_build_newfilm_split.py`

**Files:**
- Create: `scripts/09_build_newfilm_split.py`
- Test: `tests/test_newfilm_split.py`
- Produce（运行时产物，不入 git）: `data/processed/newfilm2015.txt`

**Interfaces:**
- Consumes: `load_config`/`paths_for`（genometag/config）；dataset parquet 列 `movie_id, title, year, keyframes(list), script_path, subtitles_path`；`genome_scores.parquet` 列 `movie_id, tag_id, relevance`；`data/processed/eval_test.txt`
- Produces: `data/processed/newfilm2015.txt`（每行一个 movie_id，升序去重）；`main()` 返回 0，打印：入选数、年份范围、与 eval_test 交集数（必须 0）、少样本预览

- [ ] **Step 1: 写失败测试**

```python
# tests/test_newfilm_split.py 核心断言（合成 dataset parquet 于 tmp_path）：
# - 2014-12 片子排除；2015-01 与 2018 片含入
# - keyframes=[] 与 keyframes=NaN 排除；script_path=NaN 但帧+字幕齐 → 含入
# - movie_id 不在 genome_scores → 排除
# - 与预置 eval_test.txt（含 1 个满足条件的 ≥2015 id）交集为 0
# - 幂等：文件已存在时 main() 跳过且不改写
```

- [ ] **Step 2: 运行验证失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_newfilm_split.py -v`
Expected: FAIL（ModuleNotFoundError / 函数未定义）

- [ ] **Step 3: 实现脚本**

镜像 08_build_eval_split.py 的结构（argparse `--config`，幂等跳过，写 `paths.processed / "newfilm2015.txt"`）。选择谓词：`year >= 2015` & `keyframes.apply(len) > 0`（先 fillna 空列表）& `subtitles_path.notna()` & `movie_id ∈ genome_scores` & `movie_id ∉ eval_test`。少样本诊断复用 `select_few_shots`（拟合池 = 全集 − 新片清单）。

- [ ] **Step 4: 运行验证通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_newfilm_split.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/09_build_newfilm_split.py tests/test_newfilm_split.py
git commit -m "feat: newfilm(>=2015, frames+subs) split builder with leakage guard"
```

### Task 2: 双臂配置 `configs/newfilm/{full,title}.yaml`

**Files:**
- Create: `configs/newfilm/full.yaml`、`configs/newfilm/title.yaml`
- Test: `tests/test_newfilm_configs.py`

**Interfaces:**
- Consumes: `load_config`；Task 1 的清单路径 `data/processed/newfilm2015.txt`
- Produces: `exp_name=newfilm-full` / `newfilm-title`；`04` 据此写 `results/newfilm-{full,title}/`

- [ ] **Step 1: 写失败测试**

```python
# 两个配置 load 后断言：exp_name、data.movie_list、
# full.modalities == ["video", "dialogue"]、title.modalities == []、
# model == "qwen3.7-plus"、max_frames==12、budgets==4000/3000、
# n_runs==3、few_shot_examples==2、workers==8、top_k==10、threshold==0.5
```

- [ ] **Step 2: 运行验证失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_newfilm_configs.py -v`
Expected: FAIL（文件不存在）

- [ ] **Step 3: 写两个 yaml**

以 configs/ablation/full.yaml 与 configs/probes/title-only.yaml 为模板，仅改 `exp_name` 与 `data.movie_list`，`full` 的 modalities 去掉 script（95 部无剧本）。

- [ ] **Step 4: 运行验证通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_newfilm_configs.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add configs/newfilm/ tests/test_newfilm_configs.py
git commit -m "feat: newfilm dual-arm configs (2-modality full vs title-only)"
```

### Task 3: 逐片 NDCG 模块 + 剂量-反应脚本

**Files:**
- Create: `genometag/eval/per_movie.py`
- Create: `scripts/10_dose_response.py`
- Test: `tests/test_dose_response.py`

**Interfaces:**
- Consumes: `ndcg_at_k(pred, truth, k, threshold)`（genometag/eval/metrics.py:27，纯 numpy）；`tag_type_breakdown(tag_groups)`（protocol.py:56）；annotations.parquet 列 `movie_id, tag_id, run_id, score`
- Produces: `per_movie_ndcg(pred_df, truth_df, tag_groups_df, top_k: int, threshold: float) -> pd.DataFrame`（列 `movie_id, ndcg_all, ndcg_content, ndcg_preference`）；`10_dose_response.py` 产物 `results/<exp>/dose_response/{delta_per_movie.csv, dose_summary.json, dose_response.png}`

- [ ] **Step 1: 写失败测试**

```python
# test_per_movie_matches_report：小合成数据上
# per_movie_ndcg 的 mean(ndcg_all) == write_report 路径的 ndcg@k 聚合值
# test_nan_truth_tag：某片某标签 truth 缺失 → 该标签剔除、该片仍有分数
# test_run_mean_first：构造 3 轮分数使"先平均"与"逐轮平均"结果可区分，
#   断言取"先平均"值
# test_slope_recovers：构造 full = 0.9 - 0.3*title + 噪声(seed 固定) 的逐片表，
#   断言 full~title 斜率 < 1 且 dose_summary 含 slope/se/p、
#   分箱 Δ（title 四分位）随先验升而降
```

- [ ] **Step 2: 运行验证失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_dose_response.py -v`
Expected: FAIL（模块未定义）

- [ ] **Step 3: 实现 `per_movie_ndcg`**

镜像 report.py `_evaluate_method`：3 轮按 (movie_id, tag_id) 均值 → 与 truth inner-join（truth 缺失标签自然剔除）→ 逐片 `ndcg_at_k(宽行 pred, 宽行 truth, top_k, threshold)` → 按 `tag_type_breakdown` 分组列 content/preference。不写任何新指标公式。

- [ ] **Step 4: 实现 `10_dose_response.py`**

CLI：`--config`（取 paths）、`--full <exp_name>`、`--title <exp_name>`、`--pooled-old <full>,<title>`（可选：并入 v3 老片 88 部，绘图分人口标记）。统计：① headline mean Δ + paired bootstrap(10k, seed=42) 95% CI 与 p；② **主检验** OLS full~title 斜率 b + movie 级 bootstrap CI，报告 H0: b=1 的 p（冗余零假设）；③ Spearman ρ(Δ, title) 仅描述并明示部分-整体相关警告；④ title 四分位分箱 Δ 表；⑤ matplotlib 散点 + 拟合线。

- [ ] **Step 5: 运行验证通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_dose_response.py tests/test_metrics.py -v`
Expected: PASS（含既有指标测试回归）

- [ ] **Step 6: Commit**

```bash
git add genometag/eval/per_movie.py scripts/10_dose_response.py tests/test_dose_response.py
git commit -m "feat: per-movie NDCG + dose-response analysis (slope-vs-1 primary test)"
```

### Task 4: 部署与运行手册（运维步骤，无新代码）

**Files:** 无创建；操作服务器 9.15.154.212（经 `SRV_PASS='...' bash C:/Users/admin/AppData/Local/Temp/srv.sh`）

- [ ] **Step 1: 本地全量测试绿**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: 全绿（当前基线 121+ 新增）

- [ ] **Step 2: 推送 git 并同步服务器**

`git push`；scp 新增文件到服务器项目根：`srv.sh --scp scripts/09_build_newfilm_split.py scripts/10_dose_response.py genometag/eval/per_movie.py configs/newfilm/full.yaml configs/newfilm/title.yaml tests/... /cache/g00619218/genometag-mllm/...`（保持相对路径）

- [ ] **Step 3: 服务器生成清单并核验**

```bash
ssh "cd /cache/g00619218/genometag-mllm; source server-env.sh; .venv/bin/python scripts/09_build_newfilm_split.py"
```
Expected: 打印入选 ~95 部、与 eval_test 交集 0、few-shot 仍为 Goodfellas/Terminator；`wc -l data/processed/newfilm2015.txt`

- [ ] **Step 4: 启动双臂链**

```bash
ssh "cd /cache/g00619218/genometag-mllm; source server-env.sh; nohup bash scripts/chain.sh configs/newfilm/full.yaml configs/newfilm/title.yaml >> results/newfilm_chain.log 2>&1 < /dev/null & disown"
```
验证：`tail results/newfilm_chain.log` 出现 `[chain] === configs/newfilm/full.yaml`，`pgrep -f "04_run_annotatio[n]"` 存活

- [ ] **Step 5: 监控（完成时人工触发或建 cron，每 4h）**

进度行：`tail -2 results/newfilm-full/run.log`（04 每 50 任务打印）。预计 full 臂 9,405 任务 ≈ 12-16h，title 臂（纯文本）≈ 4-6h，合计 ~20-30h @8 workers；token 预算 full ~55-60M in + ~22M out，title ~40M in + ~22M out（±30% 外推自 v3）。

- [ ] **Step 6: 完成后评估 + 分析**

```bash
ssh "cd /cache/g00619218/genometag-mllm; .venv/bin/python scripts/06_evaluate.py --config configs/newfilm/full.yaml --extra newfilm-title; .venv/bin/python scripts/10_dose_response.py --config configs/newfilm/full.yaml --full newfilm-full --title newfilm-title"
```
（pooled 版追加 `--pooled-old v3-full,v3-title-only`）

- [ ] **Step 7: 决策门（对照解读，写回记忆）**

① 先验门：newfilm-title 均值 NDCG 是否显著低于老片 0.782（描述性对比）；② 主检验：full~title 斜率 b<1（p<0.05）→ 内容在弱先验处接管；③ headline：Δ>0 显著 → F2 反转成立、叙事收官；④ 任一门不成立 → 触发方案 C（匿名标题臂）再议。
