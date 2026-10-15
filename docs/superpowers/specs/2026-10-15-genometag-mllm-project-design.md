# GenomeTag-MLLM 项目设计文档（实验框架）

> 日期：2026-10-15
> 状态：待用户审阅
> 上游文档：`docs/research-plan.md`（方向 A 研究计划，构建项目时复制自上级目录）

---

## 1. 项目背景与目标

**研究问题**：MLLM 能否从电影多模态内容（关键帧 + 剧本 + 台词）预测出与 MovieLens genome tag 体系（GroupLens SVD++ 从用户行为训练）一致的关联度评分？

**本项目的定位**：支撑该研究从数据准备到论文投稿（RecSys/CIKM 2027）全周期的实验代码库。本文档定义项目结构与实验框架的设计。

**完成度目标（本次构建）**：完整实验框架——数据准备管道（MovieNet↔MovieLens 对齐）、MLLM 标注管道、评估框架、全部基线，在合成小样本上端到端可跑通。真实数据由用户手动下载到位后，同一套代码经配置切换即可处理。

### 已确认的需求决策

| 决策点 | 结论 |
|--------|------|
| 范围 | 完整实验框架（非仅骨架或仅数据管道） |
| MLLM 访问 | OpenAI 兼容协议为唯一抽象（base_url + model 可配置） |
| 数据现状 | MovieNet 与 MovieLens 均未下载 → 提供下载指南 + 校验脚本；开发期以合成样本为主干 |
| 环境管理 | pip + venv + requirements.txt |
| 实验追踪 | 本地结果目录（每次实验一个自包含子目录），无云端依赖 |
| 语言约定 | README/注释/文档中文，代码标识符英文 |

---

## 2. 总体架构

**架构风格**：可导入包 + 编号阶段脚本 + 配置驱动。

- `genometag/` 包承载所有可复用逻辑（数据解析、对齐、tag 分组、MLLM 客户端、指标、基线），脚本与 notebook、pytest 均可导入。
- `scripts/00_~06_` 编号脚本串联管道，每个阶段独立可重跑，读写明确的中间产物。
- 一次实验 = 一个 YAML 配置 + 一个独立结果目录（配置快照随结果保存）。

**两个横切设计**（贯穿全框架）：

1. **API 调用磁盘缓存**：每次 MLLM 调用以 `(prompt + model + 参数 + run_id)` 哈希为键落盘缓存（`.cache/llm/`）。这是成本控制、断点续跑、可复现的共同基础；"3 次取平均"与"5 次一致性分析"不重复计费。
2. **Mock 模式 + 合成样本**：确定性假客户端与合成数据集让全管道在无 API key、无真实数据、无网络下可运行与测试。

### 目录结构

```
genometag-mllm/
├── README.md                      # 快速开始：冒烟 → 下载 → 真实数据
├── requirements.txt               # 轻量核心依赖
├── requirements-baselines.txt     # torch/transformers（BERT 基线可选）
├── .env.example                   # OPENAI_API_KEY / OPENAI_BASE_URL
├── .gitignore
├── configs/
│   ├── default.yaml               # 主配置模板
│   ├── smoke.yaml                 # 冒烟配置（合成样本 + mock）
│   ├── ablation/                  # 5 个消融配置（见 §5.4）
│   └── baselines.yaml
├── genometag/
│   ├── __init__.py
│   ├── config.py                  # 配置加载与校验
│   ├── data/
│   │   ├── movienet.py            # MovieNet 适配器
│   │   ├── movielens.py           # MovieLens 适配器
│   │   ├── align.py               # IMDb 主键对齐
│   │   └── sample.py              # 合成样本生成
│   ├── tags/
│   │   └── grouping.py            # tag 分组与类型标注
│   ├── annotate/
│   │   ├── client.py              # OpenAI 兼容客户端
│   │   ├── cache.py               # 磁盘缓存
│   │   ├── prompts.py             # prompt 组装
│   │   ├── pipeline.py            # 标注编排
│   │   └── mock.py                # 确定性假客户端
│   ├── eval/
│   │   ├── metrics.py             # RMSE/MAE/P@K/NDCG@K/ICC/bootstrap
│   │   ├── protocol.py            # 时序/冷启动/tag类型 子集选择器
│   │   └── report.py              # 表格与图表生成
│   └── baselines/
│       ├── random_baseline.py
│       ├── mean_baseline.py
│       ├── genre_baseline.py
│       └── bert_baseline.py
├── scripts/
│   ├── 00_make_synthetic.py       # 生成合成样本
│   ├── 01_verify_data.py          # 真实数据完整性校验（下载后）
│   ├── 02_build_dataset.py        # 对齐 + 产出实验数据集
│   ├── 03_group_tags.py           # tag 分组（一次性，产物入库）
│   ├── 04_run_annotation.py       # MLLM 标注主跑
│   ├── 05_run_baselines.py        # 基线
│   ├── 06_evaluate.py             # 评估与报告
│   └── run_smoke.py               # 端到端冒烟（00→06，mock）
├── data/                          # gitignore（tag_groups.json 例外）
│   ├── raw/                       # MovieNet/、ml-latest/（用户手动下载）
│   ├── interim/                   # aligned.parquet + align_report.md
│   ├── processed/                 # dataset.parquet、genome_scores、tag_groups.json
│   └── synthetic/                 # 合成样本（含自己的 mini tag 集）
├── results/                       # gitignore；每次实验一个子目录
├── .cache/                        # gitignore；LLM 调用缓存
├── tests/                         # pytest
├── docs/
│   ├── research-plan.md           # 方向 A 计划副本
│   ├── download-guide.md          # 下载步骤与校验
│   ├── data-format.md             # 各产物 schema
│   └── superpowers/specs/         # 本文档
└── notebooks/                     # 分析与案例研究
```

---

## 3. 数据层

### 3.1 数据流

```
data/raw/movienet/                MovieNet 原始（元数据 + 关键帧 + 剧本/字幕）
data/raw/movielens/ml-latest/     genome-scores.csv, genome-tags.csv, movies.csv, links.csv
        │  scripts/02_build_dataset.py
        ▼
data/interim/aligned.parquet      对齐后的电影表
data/interim/align_report.md      匹配率、未匹配清单、歧义案例（人工检查用）
        │
        ▼
data/processed/dataset.parquet            电影元数据 + 多模态文件引用
data/processed/genome_scores.parquet      (movie_id, tag_id, relevance) 长表
data/processed/genome_tags.parquet        (tag_id, tag)
```

tag 的分组与类型标注不在上表内——由 §4.1 的 `tag_groups.json`（独立产物、入 git）承载，下游按 `tag_id` join。

### 3.2 适配器

**`genometag/data/movienet.py`** —— 唯一接触 MovieNet 原始格式的模块。解析 MovieNet 元数据（CSV/JSON）、关键帧目录、剧本/字幕文件，输出统一中间表示：

```python
@dataclass
class MovieMultimodal:
    movienet_id: str
    title: str
    year: int | None
    imdb_id: str | None
    keyframes: list[Path]     # 按时间排序的帧文件相对路径
    script: str | None        # 剧本文本（可能缺失，见风险 §9）
    subtitles: str | None     # 字幕/台词文本
```

**`genometag/data/movielens.py`** —— 解析 genome-scores（movieId × tagId → relevance 0-1，共 1,128 tag）、genome-tags、movies（genres、含年份的标题）、links（imdbId）。

**风险隔离**：MovieNet 的确切文件格式需下载后确认。格式知识被隔离在适配器内，格式变动只改 `movienet.py` 一个文件，下游全部不动。

### 3.3 对齐（`genometag/data/align.py`）

- **主键**：IMDb ID（links.csv 的 `imdbId` ↔ MovieNet 元数据的 IMDb 字段）。
- **Fallback**：标题 + 年份规范化匹配（去大小写/标点差异），仅用于 IMDb 缺失的少数案例。
- **产出**：`aligned.parquet`（见 §3.1）+ `align_report.md`（匹配率统计、按原因分类的未匹配清单、歧义案例），对齐报告是 M1-M2 阶段的可交付物，直接决定论文可信度。

### 3.4 合成样本（`genometag/data/sample.py` + `scripts/00_make_synthetic.py`）

生成 ~5 部假电影：

- 关键帧：PIL 生成的渐变/纯色图（每部 8-16 帧），区分度足以让 mock 客户端产生差异化输出；
- 剧本/台词：预置的短文本模板（按假电影主题变化）；
- genome scores：24 个 mini tag（分 4 组，每组 6 个）× 5 部电影的假关联度矩阵，分布形状参照真实数据（多数 tag 低分、少数高分，长尾）；
- 全部落在 `data/synthetic/`，配置 `data.source: synthetic` 切换，明确标记防止与真实数据混淆。

---

## 4. 标注层

### 4.1 tag 分组（`genometag/tags/grouping.py` + `scripts/03_group_tags.py`）

- 将 1,128 个 genome tag 按语义聚类为 ~20 组（视觉风格、情绪氛围、叙事主题、角色特征、制作要素等）。
- **实现**：脚本用 MLLM 离线分组（分批输入 tag 列表，走缓存，一次性成本），同时为每个 tag 标注 `tag_type: content-driven | preference-driven`（论文核心分析维度，如 "dark atmosphere" vs "overrated"）。
- **产物**：`data/processed/tag_groups.json`，格式 `{tag_id, tag, group, tag_type, definition}[]`（`definition` 为该 tag 的一句话定义，供 §4.4 的 prompt 使用，分组时一并生成），**checked into git**（gitignore 例外）——复现论文不需重跑分组，且支持手工修正个别分组错误后入库。
- 手工修正方式：直接编辑该 JSON 文件提交，文件头注释注明生成模型与修正记录。

### 4.2 客户端（`genometag/annotate/client.py`）

- 基于 `openai` SDK（官方包，兼容所有 OpenAI 协议端点）；`base_url`、`model`、`api_key`（经环境变量）全部来自配置。
- 自动重试（指数退避，最多 3 次）；速率限制用简单的请求间隔 sleep。
- **结构化输出**：优先 `response_format: json_object`；若端点不支持则降级为文本 JSON 提取（正则截取 + `json.loads`）。解析失败重试至多 2 次，仍失败记 null 并保留原始响应。
- 每次响应记录 token 用量（usage 字段），累计进 `cost_report.json`。

### 4.3 缓存（`genometag/annotate/cache.py`）

- 键 = SHA256(prompt + model + 采样参数 + **run_id**)。**run_id 必须入键**——3 次取平均是同 prompt 发 3 次，不含轮次会撞缓存。
- 值 = 完整原始响应（JSON 文件）。原始响应永远保留：审计、换解析逻辑重算不需重新付费。
- 命中缓存的任务在 pipeline 里直接跳过 API 调用。

### 4.4 Prompt 组装（`genometag/annotate/prompts.py`）

输入 = `电影 × tag 组 × 模态组合`：

- **video 模态**：关键帧均匀采样 `max_frames`（默认 12）张，base64 编码为 image_url content parts；
- **script 模态**：剧本文本，按 `script_token_budget`（默认 4000 token）截断，截断位置在场景边界优先；
- **dialogue 模态**：字幕文本，按 `dialogue_token_budget`（默认 3000 token）截断；
- **tag 组定义**：组内每个 tag 的名称与一句话定义；
- **few-shot**：2 部已知标签的示例电影（默认 2 部、配置可调至 3；取自时序训练参考集，即分割年份前的电影，防止测试集泄露进 prompt），示例为文本形式（tag → 分数）；
- **输出要求**：JSON `{"tag_name": score}`，score ∈ [0,1]。

**模态消融 = 输入构造配置**：`modalities: [video, script, dialogue]` 是配置项，prompt 组装层按配置裁剪。`configs/ablation/` 预置 5 个配置：`video-only` / `script-only` / `dialogue-only` / `video-script`（两两组合的代表）/ `full`（三模态）。任意其他模态组合（如 `video-dialogue`）只需新增一个 YAML 文件，零代码改动。

### 4.5 标注编排（`genometag/annotate/pipeline.py` + `scripts/04_run_annotation.py`）

任务矩阵 = `电影 × tag组(~20) × run(3) `（模态组合由实验配置决定，不进矩阵维度）：

- 遍历前先查缓存与 `run_manifest.json`（任务状态清单），已完成跳过——任意时刻中断，重跑自动续；
- 每个任务输出记录：`(movie_id, tag_id, group, modality_set, run_id, score)`，畸形输出（缺 tag、分数越界）重试后仍失败记 null；
- 产出 `results/<exp>/annotations.parquet` + `cost_report.json`；
- 配置 `movie_limit` 支持先小规模试跑（如 50 部）验证 prompt 有效再放量；`--dry-run` 打印 prompt 不调用。

### 4.6 Mock 客户端（`genometag/annotate/mock.py`）

- 与真客户端实现同一接口；打分由 `(movie_id, tag_id, modality_set, run_id)` 哈希决定的确定性伪随机 + 与该电影假 ground truth 的可控相关性（可配置相关强度，用于测试指标函数的区分度）；
- 用途：pytest、`run_smoke.py`、真实数据未到位时的开发。

---

## 5. 评估层与基线

### 5.1 指标（`genometag/eval/metrics.py`）

| 指标 | 定义 |
|------|------|
| RMSE / MAE | 逐 `(movie, tag)` 对计算，支持按 tag / tag 类型 / tag 组分解 |
| Precision@10 / NDCG@10 | 每部电影：ground truth 相关集 = relevance ≥ 0.5 的 tag（阈值可配）；预测 = 按预测分排序取 top-10 |
| 一致性 | 同一电影多次调用（5 次，子集上）的分数标准差；ICC(2,1) 双向随机效应模型 |
| 显著性 | 配对 bootstrap（10,000 次重采样）检验两个系统的每电影指标差 |

所有指标函数纯函数化（输入预测/真值数组，输出标量或分组表），用手算值做单元测试。

### 5.2 评估协议（`genometag/eval/protocol.py`）

子集选择器，全部作用于 `dataset.parquet`：

- **时序分割**：上映年份 < 2022 → 训练参考集；≥ 2022 → 测试集（年份可配）；
- **冷启动**：tags.csv 中无用户标签（或标签数 ≤ 阈值）的电影子集，单独评估；
- **tag 类型分解**：content-driven vs preference-driven 标签上的性能对比（论文最重要发现的支持实验）。

**防泄露原则（全框架级约束）**：任何需要拟合的组件——全局均值基线、genre-only 基线、BERT 基线、few-shot 示例——**只允许使用时序训练参考集（2022 年前）的数据**。

### 5.3 基线（`genometag/baselines/` + `scripts/05_run_baselines.py`）

| 基线 | 实现 | 拟合数据 |
|------|------|---------|
| 随机 | 固定种子均匀采样 [0,1] | 无 |
| 全局均值 | 每 tag 在训练参考集上的 relevance 均值（非 0.5 常数） | 2022 前 |
| genre-only | genre → tag 均值映射表（训练参考集统计，未见过的 genre 组合回退到全局均值） | 2022 前 |
| 剧本 + BERT | transformers 编码器 + 多标签回归头，预测 1,128 维 relevance | 2022 前训练，2022 后 + 全集评估 |

BERT 基线依赖 torch/transformers，放 `requirements-baselines.txt` 可选安装；`05_run_baselines.py` 未安装时跳过该项并提示，不阻塞其余基线。

### 5.4 报告（`genometag/eval/report.py` + `scripts/06_evaluate.py`）

- 输入：一个或多个实验目录的 `annotations.parquet` + 基线输出；
- 输出：主实验表（所有方法 × 所有指标，CSV + markdown）、模态×标签类别贡献热力图（matplotlib）、时序分割前后对比表、冷启动子集表、一致性分析表；
- 报告落在被评估实验的 `results/<exp>/` 内（含配置快照，目录自包含）。

### 5.5 实验配置约定

一次实验 = `configs/` 下一个 YAML + `results/<exp_name>/`（exp_name 取配置文件名）。核心配置结构：

```yaml
data:
  source: real | synthetic       # synthetic = 合成样本（冒烟/测试）
  movie_limit: null              # null = 全量；int = 随机采样种子固定
  seed: 42
model:
  base_url_env: OPENAI_BASE_URL  # 从环境变量读取，配置不落密钥
  model: gpt-4o
  temperature: 0.7
  max_frames: 12
  script_token_budget: 4000
  dialogue_token_budget: 3000
annotation:
  n_runs: 3
  modalities: [video, script, dialogue]
  few_shot_examples: 2
protocol:
  temporal_split_year: 2022
  relevance_threshold: 0.5
  top_k: 10
  cold_start_max_tags: 0
consistency:                     # 独立实验配置项，非主跑默认
  n_runs: 5
  movie_limit: 100
mock: false                      # true = mock 客户端
```

`configs/ablation/` 五个文件仅 `annotation.modalities` 不同：`video-only` / `script-only` / `dialogue-only` / `video-script` / `full`。

---

## 6. 运行约定与冒烟路径

- **冒烟**：`python scripts/run_smoke.py` = 以 `configs/smoke.yaml`（合成样本 + mock：5 部电影 × 4 个 mini tag 组 × 1 run）串联 00→06 全管道，秒级、零成本。它同时是 pytest 集成测试的内容（`tests/test_smoke.py`）和日常回归手段。
- **真实数据首次路径**：按 `docs/download-guide.md` 下载 → `01_verify_data.py` 校验 → `02_build_dataset.py` 检查对齐报告 → 小规模 `movie_limit: 50` 试跑 → 放量。
- **结果目录自包含**：config 快照 + annotations + 指标 + 图表全在 `results/<exp>/`，gitignore（最终论文用表图复制到 docs/）。

---

## 7. 测试策略（TDD）

| 层 | 测试内容 |
|----|---------|
| 单元 | 指标函数手算值验证；对齐逻辑（IMDb 匹配、标题规范化）；prompt 组装（模态裁剪、token 预算截断、few-shot 来自训练集断言）；缓存键稳定性与 run_id 区分；输出校验降级路径；tag 分组产物 schema |
| 集成 | 合成样本 + mock 全管道冒烟（`test_smoke.py`） |
| 约束 | 全部测试无网络、无 API key、无真实数据 |

---

## 8. 依赖

**`requirements.txt`**（核心，Python ≥ 3.11）：pandas、pyarrow、pyyaml、pillow、openai、matplotlib、scipy、scikit-learn、pytest。

**`requirements-baselines.txt`**（可选）：torch、transformers。

不引入：web 框架、任务队列、wandb/mlflow、DVC——当前规模下均为过度设计。

---

## 9. 风险与缓解

| 风险 | 缓解 |
|------|------|
| MovieNet 原始格式未确认（下载前无法验证适配器） | 格式知识隔离在 `movienet.py`；`01_verify_data.py` 在数据到位后首先校验结构并报告差异 |
| MovieNet 剧本（scripts）覆盖不足 | 字幕文本作为台词模态代理；适配器层处理，管道结构不变 |
| 全量成本（~1,000 电影 × 20 组 × 3 run × 5 消融 ≈ 30 万次调用） | 缓存 + 断点续跑；`movie_limit` 递增放量；消融实验允许在采样子集上跑（论文中注明） |
| OpenAI 兼容端点对 `response_format` 支持不一 | 解析降级路径 + 原始响应保留（§4.2/4.3） |
| MLLM 记忆 vs 推理混淆 | 时序分割协议为框架一级公民（§5.2）；研究计划已有叙事 fallback |
| **MovieNet 年份覆盖与分割点冲突**：MovieNet（2019 发布）大概率不含 2022 后电影，而研究计划时序分割定在 2022（MLLM 训练截止逻辑需分割年份 ≥ 2023） | 框架层面：分割年份纯配置（`temporal_split_year`），任意年份可用。研究层面：数据到位后按实际年份覆盖重定分割点（配置改一行），若覆盖确实不足，"推理 vs 回忆"论证需研究计划层面重新设计——届时回到设计流程，不改代码框架 |

---

## 10. 研究计划里程碑对应

| 研究计划阶段 | 项目模块 |
|-------------|---------|
| M1-M2 数据准备（对齐 + 分组） | scripts 00-03，`data/`、`tags/` |
| M2-M4 主跑 + 消融 + 时序 | scripts 04-06，`annotate/`、`eval/` |
| M4-M5 冷启动 + 一致性 + 案例 | `eval/protocol.py` 子集选择器 + notebooks |
| M5-M7 论文写作 | docs/ 产物 + notebooks 分析 |
