# GenomeTag-MLLM 实验框架实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建 GenomeTag-MLLM 研究项目的完整实验框架——数据准备（MovieNet↔MovieLens 对齐）、MLLM 标注管道、评估框架、基线——在合成小样本上端到端可跑通。

**Architecture:** 可导入包 `genometag/` + 编号阶段脚本 `scripts/00~06` + YAML 配置驱动。横切设计：API 调用磁盘缓存（成本/续跑/复现基础）+ mock 客户端与合成样本（无网络、无 key、无真实数据可全流程运行）。

**Tech Stack:** Python ≥ 3.11，pip + venv；pandas/pyarrow（数据）、openai SDK（OpenAI 兼容端点）、Pillow（合成关键帧）、matplotlib（图表）、scipy/numpy（统计）、pytest（测试）；torch/transformers 为可选层（BERT 基线）。

**Spec:** `docs/superpowers/specs/2026-10-15-genometag-mllm-project-design.md`（本计划从 spec 出发论证，执行者需同时阅读两者）

## Global Constraints

- Python ≥ 3.11；仅用 `requirements.txt` 核心依赖；torch/transformers 只出现在 `requirements-baselines.txt` 与 `bert_baseline.py`（懒导入）
- 代码标识符英文，注释/docstring 中文；提交信息中文、 conventional 前缀（feat:/test:/docs:/chore:）
- 所有自动化测试**无网络、无 API key、无真实数据**（唯一例外：BERT 基线测试需联网下载模型，用 `RUN_BERT_TEST=1` 环境变量显式开启，默认跳过）
- **防泄露约束**：任何拟合组件（mean/genre/BERT 基线、few-shot 示例选择）只允许使用 `protocol.temporal_split_year` 之前（严格小于）的电影数据，违反即抛错
- 路径约定（不可配置，约定优于配置）：`data/raw/movienet/`、`data/raw/movielens/ml-latest/`、`data/synthetic/{raw,interim,processed}/`、`data/interim/`、`data/processed/`、`.cache/llm/`、`results/<exp_name>/`
- `mock: true` 配置下**不得发起任何网络调用**（mock 客户端不读不写缓存）
- Python 解释器统一用 `.venv/Scripts/python`（Windows venv），下文命令简写为 `$PY`（执行前 `export PY=.venv/Scripts/python`）
- 每个 TDD 任务顺序：写失败测试 → 运行确认失败 → 最小实现 → 运行确认通过 → 提交
- 主键约定：全管道以 MovieLens `movie_id`（int）为主键；MovieNet 侧 `movienet_id`（str）仅在适配层出现
- 所有命令在仓库根目录执行（脚本默认路径 `configs/...` 依赖 cwd）
- **脚本 sys.path 自举**（Task 3 执行时发现）：包未 pip 安装，直接 `python scripts/xx.py` 时仓库根不在 sys.path——每个 scripts/ 下的脚本在导入 genometag 之前须加入：
  ```python
  import sys
  from pathlib import Path
  sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
  ```
  （pytest 不受影响——rootdir 已在 sys.path；run_smoke 经 importlib 加载同目录脚本时继承 pytest 的 sys.path）

## Review Focus

spec 是愿景文档，以下五类输入/故障模式 spec 有要求但分散、最易在实现中出问题。每条已锚定到对应任务的测试：

1. **缓存键缺维度**——多轮调用（3 次取平均）prompt 相同，若 run_id 不入键则 3 次全命中同一缓存，均值恒等于单次（假一致性）→ Task 7 `test_cache.py::test_key_differs_by_run_id`、`test_key_changes_with_params`
2. **MLLM 畸形输出**——缺 tag、分数越界、JSON 外带杂物 → 崩溃或静默错分 → Task 10 `test_validation.py`（parse_scores 全分支）+ Task 12 `test_pipeline.py::test_parse_failure_records_null`（重试后仍失败记 null、原始响应保留）
3. **对齐规范化**——IMDb ID 的 `tt` 前缀/前导零/pandas float 化（`12345.0`）、翻拍同名同年歧义 → 错配 ground truth → Task 6 `test_align.py::test_normalize_imdb`、`test_ambiguous_remake_unmatched`
4. **泄露路径**——few-shot 示例或基线拟合用了 ≥ 分割年的电影 → 时序实验整体无效 → Task 14 `test_protocol.py::test_assert_no_leakage`、Task 12 `test_prompts.py::test_few_shots_from_train_only`
5. **空模态输入**——电影无剧本且 `modalities=[script]` → prompt 无实质内容仍计费调用 → Task 12 `test_pipeline.py::test_no_content_task_skipped`（记 `no_content`，不调用 API）

---

### Task 1: 项目脚手架

**Files:**
- Create: `.gitignore`、`.env.example`、`requirements.txt`、`requirements-baselines.txt`、`README.md`（初版）、`genometag/__init__.py`、`genometag/{data,tags,annotate,eval,baselines}/__init__.py`、`tests/__init__.py`、`notebooks/.gitkeep`

**Interfaces:**
- Consumes: 无（首个任务）
- Produces: 可导入的 `genometag` 包与空子包；可运行的 pytest；`.venv` 环境

- [ ] **Step 1: 创建包骨架与工程文件**

```
# 目录结构（全部为空 __init__.py，除 README 外无内容）
genometag-mllm/
├── .gitignore
├── .env.example
├── requirements.txt
├── requirements-baselines.txt
├── README.md
├── genometag/
│   ├── __init__.py
│   ├── data/__init__.py
│   ├── tags/__init__.py
│   ├── annotate/__init__.py
│   ├── eval/__init__.py
│   └── baselines/__init__.py
├── tests/__init__.py
└── notebooks/.gitkeep
```

`.gitignore`（注意：`data/processed/*` 用通配排除而非排除目录本身，否则 negation 不生效）：

```gitignore
.venv/
__pycache__/
*.pyc
.pytest_cache/
.env
data/raw/
data/interim/
data/synthetic/
data/processed/*
!data/processed/tag_groups.json
results/
.cache/
notebooks/.ipynb_checkpoints/
```

`.env.example`：

```
# OpenAI 兼容端点配置（复制为 .env 并填入真实值；.env 已被 gitignore）
OPENAI_API_KEY=sk-your-key
OPENAI_BASE_URL=https://api.openai.com/v1
```

`requirements.txt`：

```
pandas>=2.0
pyarrow>=14.0
PyYAML>=6.0
Pillow>=10.0
openai>=1.40
matplotlib>=3.8
scipy>=1.11
scikit-learn>=1.3
pytest>=8.0
```

`requirements-baselines.txt`：

```
# 可选：BERT 基线依赖（scripts/05 未安装时自动跳过该基线）
torch>=2.2
transformers>=4.40
```

`README.md`（初版，Task 19 扩充）：

```markdown
# GenomeTag-MLLM

MLLM 从电影多模态内容（关键帧+剧本+台词）预测 MovieLens genome tag 关联度。
设计文档：`docs/superpowers/specs/`；实现计划：`docs/superpowers/plans/`。

## 快速开始

​```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest tests/ -v          # 全部测试（无网络无 key）
.venv/Scripts/python scripts/run_smoke.py         # 端到端冒烟（合成样本 + mock）
​```
```

- [ ] **Step 2: 创建 venv 并安装依赖**

```bash
cd genometag-mllm
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

- [ ] **Step 3: 写冒烟测试验证环境**

`tests/test_scaffold.py`：

```python
"""脚手架验证：包与全部子包可导入。"""


def test_package_importable():
    import genometag
    assert genometag is not None


def test_subpackages_importable():
    import genometag.data  # noqa: F401
    import genometag.tags  # noqa: F401
    import genometag.annotate  # noqa: F401
    import genometag.eval  # noqa: F401
    import genometag.baselines  # noqa: F401
```

- [ ] **Step 4: 运行测试**

Run: `.venv/Scripts/python -m pytest tests/test_scaffold.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: 项目脚手架（包结构、依赖、gitignore、pytest 就绪）"
```

---

### Task 2: 配置模块 config.py

**Files:**
- Create: `genometag/config.py`
- Create: `configs/default.yaml`、`configs/smoke.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `AppConfig`（字段见下）、`load_config(path: str | Path) -> AppConfig`、`validate_config(cfg: AppConfig) -> None`（非法抛 `ValueError`）
  - `paths_for(config: AppConfig) -> Paths`；`Paths` 含 `root, movienet, movielens, interim, processed, cache, results, tag_groups, dataset, genome_scores, genome_tags, aligned`（全 `Path`）
  - `project_root() -> Path`（= 仓库根，由 `genometag/config.py` 位置推出）
  - 全部后续任务经 `load_config` + `paths_for` 取配置与路径

- [ ] **Step 1: 写失败测试**

`tests/test_config.py`：

```python
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
```

注意：最后一个测试直接构造 `AppConfig` 测 `validate_config`，绕过 YAML。

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_config.py -v`
Expected: FAIL（`ModuleNotFoundError: genometag.config`）

- [ ] **Step 3: 实现 config.py**

`genometag/config.py`：

```python
"""配置加载、校验与项目路径推导。

约定优于配置：数据/缓存/结果目录布局固定（见 Paths），YAML 只描述实验参数。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

VALID_MODALITIES = ("video", "script", "dialogue")
VALID_SOURCES = ("real", "synthetic")


@dataclass
class DataConfig:
    source: str = "real"            # real | synthetic
    movie_limit: int | None = None  # None = 全量
    seed: int = 42


@dataclass
class ModelConfig:
    base_url_env: str = "OPENAI_BASE_URL"
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_frames: int = 12
    script_token_budget: int = 4000
    dialogue_token_budget: int = 3000
    request_interval: float = 0.5   # 两次请求间隔秒数（简单限速）


@dataclass
class AnnotationConfig:
    n_runs: int = 3
    modalities: list[str] = field(default_factory=lambda: list(VALID_MODALITIES))
    few_shot_examples: int = 2


@dataclass
class ProtocolConfig:
    temporal_split_year: int = 2022
    relevance_threshold: float = 0.5
    top_k: int = 10
    cold_start_max_tags: int = 0


@dataclass
class ConsistencyConfig:
    n_runs: int = 5
    movie_limit: int = 100


@dataclass
class BaselinesConfig:
    run: list[str] = field(default_factory=lambda: ["random", "mean", "genre", "bert"])


@dataclass
class AppConfig:
    exp_name: str = "default"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    annotation: AnnotationConfig = field(default_factory=AnnotationConfig)
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    consistency: ConsistencyConfig = field(default_factory=ConsistencyConfig)
    baselines: BaselinesConfig = field(default_factory=BaselinesConfig)
    mock: bool = False


@dataclass
class Paths:
    root: Path
    movienet: Path      # MovieNet 原始数据根（real 或 synthetic）
    movielens: Path     # MovieLens 数据根（含 genome-*.csv 的目录）
    interim: Path
    processed: Path
    cache: Path
    results: Path
    tag_groups: Path
    dataset: Path
    genome_scores: Path
    genome_tags: Path
    aligned: Path


def project_root() -> Path:
    """仓库根 = genometag/ 的上一级。"""
    return Path(__file__).resolve().parent.parent


def paths_for(config: AppConfig) -> Paths:
    root = project_root()
    if config.data.source == "synthetic":
        base = root / "data" / "synthetic"
        movienet = base / "raw" / "movienet"
        movielens = base / "raw" / "movielens"
        interim = base / "interim"
        processed = base / "processed"
    else:
        movienet = root / "data" / "raw" / "movienet"
        movielens = root / "data" / "raw" / "movielens" / "ml-latest"
        interim = root / "data" / "interim"
        processed = root / "data" / "processed"
    return Paths(
        root=root, movienet=movienet, movielens=movielens,
        interim=interim, processed=processed,
        cache=root / ".cache" / "llm",
        results=root / "results" / config.exp_name,
        tag_groups=processed / "tag_groups.json",
        dataset=processed / "dataset.parquet",
        genome_scores=processed / "genome_scores.parquet",
        genome_tags=processed / "genome_tags.parquet",
        aligned=interim / "aligned.parquet",
    )


def validate_config(cfg: AppConfig) -> None:
    if cfg.data.source not in VALID_SOURCES:
        raise ValueError(f"data.source 必须是 {VALID_SOURCES}，得到 {cfg.data.source!r}")
    bad = [m for m in cfg.annotation.modalities if m not in VALID_MODALITIES]
    if bad:
        raise ValueError(f"未知模态 {bad}，可选 {VALID_MODALITIES}")
    if cfg.annotation.n_runs < 1:
        raise ValueError("annotation.n_runs 必须 ≥ 1")
    if cfg.annotation.few_shot_examples < 0:
        raise ValueError("annotation.few_shot_examples 必须 ≥ 0")
    if not 0.0 < cfg.protocol.relevance_threshold < 1.0:
        raise ValueError("protocol.relevance_threshold 必须在 (0,1) 开区间")
    if cfg.protocol.top_k < 1:
        raise ValueError("protocol.top_k 必须 ≥ 1")
    if cfg.data.movie_limit is not None and cfg.data.movie_limit < 1:
        raise ValueError("data.movie_limit 为 None 或 ≥ 1")


def load_config(path: str | Path) -> AppConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg = AppConfig(
        exp_name=raw.get("exp_name", "default"),
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        annotation=AnnotationConfig(**raw.get("annotation", {})),
        protocol=ProtocolConfig(**raw.get("protocol", {})),
        consistency=ConsistencyConfig(**raw.get("consistency", {})),
        baselines=BaselinesConfig(**raw.get("baselines", {})),
        mock=bool(raw.get("mock", False)),
    )
    validate_config(cfg)
    return cfg
```

- [ ] **Step 4: 写 configs/default.yaml 与 configs/smoke.yaml**

`configs/default.yaml`：

```yaml
exp_name: default
data:
  source: real
  movie_limit: null
  seed: 42
model:
  base_url_env: OPENAI_BASE_URL
  api_key_env: OPENAI_API_KEY
  model: gpt-4o
  temperature: 0.7
  max_frames: 12
  script_token_budget: 4000
  dialogue_token_budget: 3000
  request_interval: 0.5
annotation:
  n_runs: 3
  modalities: [video, script, dialogue]
  few_shot_examples: 2
protocol:
  temporal_split_year: 2022
  relevance_threshold: 0.5
  top_k: 10
  cold_start_max_tags: 0
consistency:
  n_runs: 5
  movie_limit: 100
baselines:
  run: [random, mean, genre, bert]
mock: false
```

`configs/smoke.yaml`：

```yaml
exp_name: smoke
data:
  source: synthetic
  seed: 42
model:
  model: mock-1
  temperature: 0.7
  max_frames: 6
  script_token_budget: 500
  dialogue_token_budget: 300
annotation:
  n_runs: 2
  modalities: [video, script, dialogue]
  few_shot_examples: 1
protocol:
  temporal_split_year: 2022
  relevance_threshold: 0.5
  top_k: 5
  cold_start_max_tags: 0
baselines:
  run: [random, mean, genre]
mock: true
```

- [ ] **Step 5: 运行测试**

Run: `.venv/Scripts/python -m pytest tests/test_config.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add genometag/config.py configs/ tests/test_config.py
git commit -m "feat: 配置模块（加载、校验、路径推导）与 default/smoke 配置"
```

---

### Task 3: 合成样本生成

**Files:**
- Create: `genometag/data/sample.py`
- Create: `scripts/00_make_synthetic.py`
- Test: `tests/test_sample.py`

**Interfaces:**
- Consumes: `genometag.annotate.mock.base_score(title: str, tag: str) -> float`（Task 9 提供；本任务先定义**本模块内私有副本** `_hash_unit`，Task 9 实现后切换为导入——见 Step 3 注释）。为避免前向依赖，此处直接实现 `base_score` 的等价逻辑并写在 sample.py 内，Task 9 的 mock.py 反向导入 sample.py 的 `base_score`（唯一权威实现在此）。
- Produces:
  - `SYNTHETIC_TAGS: list[dict]`（24 个 tag，各含 `tag_id, tag, group, tag_type, definition`，4 组 × 6）
  - `SYNTHETIC_MOVIES: list[dict]`（5 部，含 `movie_id, movienet_id, title, year, genres, imdb_id`）
  - `base_score(title: str, tag: str) -> float`（确定性 [0,1)，mock 相关性的基准函数）
  - `make_synthetic_dataset(out_root: Path) -> None`（产出 `out_root/raw/movienet/`、`out_root/raw/movielens/`、`out_root/processed/tag_groups.json`）

- [ ] **Step 1: 写失败测试**

`tests/test_sample.py`：

```python
"""合成样本生成：结构、形状、冷启动分布、年份分割。"""
import pandas as pd

from genometag.data.sample import (
    SYNTHETIC_MOVIES, SYNTHETIC_TAGS, base_score, make_synthetic_dataset,
)


def test_constants_shape():
    assert len(SYNTHETIC_TAGS) == 24
    assert len({t["group"] for t in SYNTHETIC_TAGS}) == 4
    assert len(SYNTHETIC_MOVIES) == 5
    # tag_type 只有 content / preference 两种
    assert {t["tag_type"] for t in SYNTHETIC_TAGS} == {"content", "preference"}


def test_base_score_deterministic_and_bounded():
    assert base_score("Crimson Horizon", "neon visuals") == base_score("Crimson Horizon", "neon visuals")
    assert 0.0 <= base_score("A", "x") <= 1.0
    assert base_score("A", "x") != base_score("B", "x")  # 不同电影不同分


def test_make_synthetic_dataset_structure(tmp_path):
    make_synthetic_dataset(tmp_path)
    # MovieLens 侧
    ml = tmp_path / "raw" / "movielens"
    scores = pd.read_csv(ml / "genome-scores.csv")
    tags = pd.read_csv(ml / "genome-tags.csv")
    movies = pd.read_csv(ml / "movies.csv")
    links = pd.read_csv(ml / "links.csv")
    utags = pd.read_csv(ml / "tags.csv")
    assert len(scores) == 5 * 24
    assert len(tags) == 24
    assert len(movies) == 5
    assert len(links) == 5
    # 冷启动：仅 movie_id 1-3 有用户标签，4-5 无
    tagged = set(utags["movieId"].unique())
    assert tagged == {1, 2, 3}
    # 年份分割（2022）：4 部 < 2022，1 部 ≥ 2022（年份编码在标题内，同真实格式）
    titles = movies["title"].tolist()
    assert sum(1 for t in titles if "(2023)" in t) == 1
    assert sum(1 for t in titles if any(y in t for y in
               ("(2018)", "(2019)", "(2020)", "(2021)"))) == 4
    # MovieNet 侧
    mn = tmp_path / "raw" / "movienet"
    mn_movies = pd.read_csv(mn / "movies.csv")
    assert len(mn_movies) == 5
    frames = sorted((mn / "keyframes" / "mn01").glob("*.jpg"))
    assert len(frames) == 8  # 每部 8 帧
    assert (mn / "subtitles" / "mn01.srt").exists()
    # processed：tag_groups.json 已随样本生成
    import json
    tg = json.loads((tmp_path / "processed" / "tag_groups.json").read_text(encoding="utf-8"))
    assert len(tg) == 24
    assert {"tag_id", "tag", "group", "tag_type", "definition"} <= set(tg[0].keys())
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_sample.py -v`
Expected: FAIL（`ModuleNotFoundError` / `ImportError`）

- [ ] **Step 3: 实现 sample.py**

`genometag/data/sample.py`：

```python
"""合成小样本：结构镜像真实原始数据格式，用于开发、测试与冒烟。

产出三块：
- raw/movienet/   （镜像 MovieNet 假定格式，见 docs/data-format.md）
- raw/movielens/  （镜像 MovieLens ml-latest 格式）
- processed/tag_groups.json（合成样本的分组是已知的，直接生成，不走 LLM）

ground truth 生成规则：relevance = 0.85 * base_score(title, tag) + 0.15 * 独立噪声。
MockClient（genometag/annotate/mock.py）预测 base_score + 小幅轮次噪声，
两者天然相关——这是 mock"可控相关性"的实现基础。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

SYNTHETIC_MOVIES = [
    # movie_id(ML), movienet_id, title, year, genres, imdb_id(不带 tt 前缀的数字)
    {"movie_id": 1, "movienet_id": "mn01", "title": "Crimson Horizon", "year": 2018,
     "genres": ["Action", "Thriller"], "imdb_id": 1},
    {"movie_id": 2, "movienet_id": "mn02", "title": "Neon Alley", "year": 2019,
     "genres": ["Crime", "Drama"], "imdb_id": 2},
    {"movie_id": 3, "movienet_id": "mn03", "title": "Quiet Harbor", "year": 2020,
     "genres": ["Drama", "Romance"], "imdb_id": 3},
    {"movie_id": 4, "movienet_id": "mn04", "title": "Last Ember", "year": 2021,
     "genres": ["Sci-Fi", "Action"], "imdb_id": 4},
    {"movie_id": 5, "movienet_id": "mn05", "title": "Paper Skies", "year": 2023,
     "genres": ["Drama", "Mystery"], "imdb_id": 5},
]

SYNTHETIC_TAGS = [
    # group: 视觉风格 / 情绪氛围 / 叙事主题 / 受众偏好
    {"tag_id": 1,  "tag": "neon visuals",    "group": "视觉风格", "tag_type": "content",
     "definition": "画面以霓虹灯光与高饱和夜景为主"},
    {"tag_id": 2,  "tag": "dark palette",    "group": "视觉风格", "tag_type": "content",
     "definition": "整体色调偏暗、低照明场景占主导"},
    {"tag_id": 3,  "tag": "vivid colors",    "group": "视觉风格", "tag_type": "content",
     "definition": "色彩鲜艳明亮、对比强烈"},
    {"tag_id": 4,  "tag": "grainy texture",  "group": "视觉风格", "tag_type": "content",
     "definition": "胶片颗粒感或做旧质感"},
    {"tag_id": 5,  "tag": "static camera",   "group": "视觉风格", "tag_type": "content",
     "definition": "大量固定机位长镜头"},
    {"tag_id": 6,  "tag": "slow pans",       "group": "视觉风格", "tag_type": "content",
     "definition": "缓慢的横摇/横移镜头"},
    {"tag_id": 7,  "tag": "melancholic",     "group": "情绪氛围", "tag_type": "content",
     "definition": "整体情绪忧郁、低落"},
    {"tag_id": 8,  "tag": "tense",           "group": "情绪氛围", "tag_type": "content",
     "definition": "持续紧张、压迫感"},
    {"tag_id": 9,  "tag": "hopeful",         "group": "情绪氛围", "tag_type": "content",
     "definition": "充满希望与治愈感"},
    {"tag_id": 10, "tag": "bleak",           "group": "情绪氛围", "tag_type": "content",
     "definition": "荒凉、绝望的世界观"},
    {"tag_id": 11, "tag": "dreamlike",       "group": "情绪氛围", "tag_type": "content",
     "definition": "梦幻、超现实的氛围"},
    {"tag_id": 12, "tag": "nostalgic",       "group": "情绪氛围", "tag_type": "content",
     "definition": "怀旧、追忆过去的情绪"},
    {"tag_id": 13, "tag": "revenge plot",    "group": "叙事主题", "tag_type": "content",
     "definition": "以复仇为核心驱动力"},
    {"tag_id": 14, "tag": "family bonds",    "group": "叙事主题", "tag_type": "content",
     "definition": "家庭关系与亲情羁绊"},
    {"tag_id": 15, "tag": "identity crisis", "group": "叙事主题", "tag_type": "content",
     "definition": "主角经历身份认同危机"},
    {"tag_id": 16, "tag": "heist plan",      "group": "叙事主题", "tag_type": "content",
     "definition": "围绕精心策划的抢劫/盗窃行动"},
    {"tag_id": 17, "tag": "lost love",       "group": "叙事主题", "tag_type": "content",
     "definition": "失去的爱情或错过的恋人"},
    {"tag_id": 18, "tag": "survival",        "group": "叙事主题", "tag_type": "content",
     "definition": "绝境求生主线"},
    {"tag_id": 19, "tag": "overrated",       "group": "受众偏好", "tag_type": "preference",
     "definition": "名过其实、评价高于实际质量"},
    {"tag_id": 20, "tag": "cult classic",    "group": "受众偏好", "tag_type": "preference",
     "definition": "小众狂热追捧的经典"},
    {"tag_id": 21, "tag": "feel-good",       "group": "受众偏好", "tag_type": "preference",
     "definition": "看完心情变好的轻松作品"},
    {"tag_id": 22, "tag": "slow burn",       "group": "受众偏好", "tag_type": "preference",
     "definition": "节奏缓慢、渐入佳境"},
    {"tag_id": 23, "tag": "rewatchable",     "group": "受众偏好", "tag_type": "preference",
     "definition": "值得反复观看"},
    {"tag_id": 24, "tag": "niche appeal",    "group": "受众偏好", "tag_type": "preference",
     "definition": "只对特定人群有吸引力"},
]


def _hash_unit(key: str) -> float:
    """哈希 → 确定性 [0,1) 均匀值。"""
    h = hashlib.sha256(key.encode()).hexdigest()
    return int(h[:8], 16) / 0x1_0000_0000


def base_score(title: str, tag: str) -> float:
    """电影×标签 的确定性基准分。ground truth 与 MockClient 共用（相关性的来源）。"""
    return _hash_unit(f"base|{title}|{tag}")


def _truth(title: str, tag: str) -> float:
    """ground truth = 0.85*base + 0.15*独立噪声，clip 到 [0,1]。"""
    noise = _hash_unit(f"truth|{title}|{tag}")
    return min(1.0, max(0.0, 0.85 * base_score(title, tag) + 0.15 * noise))


def _make_frame(path: Path, color: tuple, index: int) -> None:
    """生成一张渐变关键帧（尺寸小，冒烟要快）。"""
    img = Image.new("RGB", (64, 36), color)
    draw = ImageDraw.Draw(img)
    shade = int(255 * index / 8)
    draw.rectangle([0, 0, 63, 18], fill=(min(255, color[0] + shade), color[1], color[2]))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=60)


_MOVIE_COLORS = [(120, 20, 20), (20, 20, 120), (20, 120, 60), (80, 80, 20), (60, 20, 80)]


def _script_text(title: str) -> str:
    return (
        f"INT. {title.upper()} - OPENING - NIGHT\n\n"
        "The city hums outside. Rain streaks the window.\n\n"
        f"Our protagonist studies a photograph. The story of {title} begins here.\n\n"
        "EXT. STREET - CONTINUOUS\n\nTraffic. Neon reflections in puddles.\n"
    )


def _srt_text(title: str) -> str:
    return (
        "1\n00:00:01,000 --> 00:00:03,500\n"
        f"They said {title} would change everything.\n\n"
        "2\n00:00:04,000 --> 00:00:06,000\n"
        "Nothing changes. We just keep moving.\n\n"
        "3\n00:00:07,000 --> 00:00:09,500\n"
        "Then move faster.\n"
    )


def make_synthetic_dataset(out_root: Path) -> None:
    """生成完整合成数据集。out_root 即 data/synthetic/。"""
    out_root = Path(out_root)
    mn = out_root / "raw" / "movienet"
    ml = out_root / "raw" / "movielens"
    mn.mkdir(parents=True, exist_ok=True)
    ml.mkdir(parents=True, exist_ok=True)

    # MovieNet 侧
    for i, m in enumerate(SYNTHETIC_MOVIES):
        for f in range(8):
            _make_frame(mn / "keyframes" / m["movienet_id"] / f"frame_{f:03d}.jpg",
                        _MOVIE_COLORS[i], f)
        (mn / "scripts" / f'{m["movienet_id"]}.txt').write_text(
            _script_text(m["title"]), encoding="utf-8")
        (mn / "subtitles" / f'{m["movienet_id"]}.srt').write_text(
            _srt_text(m["title"]), encoding="utf-8")
    pd_rows = [{"movie_id": m["movienet_id"], "title": m["title"], "year": m["year"],
                "imdb_id": f'tt{m["imdb_id"]:07d}'} for m in SYNTHETIC_MOVIES]
    import pandas as pd
    pd.DataFrame(pd_rows).to_csv(mn / "movies.csv", index=False)

    # MovieLens 侧（movies.csv 镜像真实三列格式，年份由 load_movies 从标题解析）
    movies_rows = [{"movieId": m["movie_id"],
                    "title": f'{m["title"]} ({m["year"]})',
                    "genres": "|".join(m["genres"])}
                   for m in SYNTHETIC_MOVIES]
    pd.DataFrame(movies_rows).to_csv(ml / "movies.csv", index=False)
    pd.DataFrame([{"movieId": m["movie_id"], "imdbId": m["imdb_id"], "tmdbId": m["movie_id"]}
                  for m in SYNTHETIC_MOVIES]).to_csv(ml / "links.csv", index=False)
    pd.DataFrame([{"tagId": t["tag_id"], "tag": t["tag"]}
                  for t in SYNTHETIC_TAGS]).to_csv(ml / "genome-tags.csv", index=False)
    score_rows = [{"movieId": m["movie_id"], "tagId": t["tag_id"],
                   "relevance": round(_truth(m["title"], t["tag"]), 4)}
                  for m in SYNTHETIC_MOVIES for t in SYNTHETIC_TAGS]
    pd.DataFrame(score_rows).to_csv(ml / "genome-scores.csv", index=False)
    # 用户标签：电影 1-3 各 2 条（4-5 冷启动）
    user_rows = [{"userId": u, "movieId": mid, "tag": "fun", "timestamp": 1700000000}
                 for mid in (1, 2, 3) for u in (1, 2)]
    pd.DataFrame(user_rows).to_csv(ml / "tags.csv", index=False)

    # processed：tag_groups.json（合成分组已知，直接生成）
    processed = out_root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    (processed / "tag_groups.json").write_text(
        json.dumps(SYNTHETIC_TAGS, ensure_ascii=False, indent=1), encoding="utf-8")
```

- [ ] **Step 4: 写脚本 00**

`scripts/00_make_synthetic.py`：

```python
"""00 · 生成合成样本到 data/synthetic/。幂等：重新生成覆盖旧文件。"""
from __future__ import annotations

import argparse

from genometag.config import load_config, paths_for
from genometag.data.sample import make_synthetic_dataset


def main(config_path: str = "configs/smoke.yaml") -> int:
    config = load_config(config_path)
    paths = paths_for(config)
    out_root = paths.movienet.parent.parent  # data/synthetic/
    make_synthetic_dataset(out_root)
    print(f"[00] 合成样本已生成: {out_root}")
    print(f"     MovieNet 镜像: {paths.movienet}")
    print(f"     MovieLens 镜像: {paths.movielens}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args()
    raise SystemExit(main(args.config))
```

注意：`main` 默认参数为 smoke 配置（合成样本只对 synthetic 源有意义）；命令行可用 `--config` 覆盖。

- [ ] **Step 5: 运行测试**

Run: `.venv/Scripts/python -m pytest tests/test_sample.py -v`
Expected: 3 passed

Run: `.venv/Scripts/python scripts/00_make_synthetic.py`
Expected: 打印生成路径；`data/synthetic/raw/` 下出现文件

- [ ] **Step 6: Commit**

```bash
git add genometag/data/sample.py scripts/00_make_synthetic.py tests/test_sample.py
git commit -m "feat: 合成样本生成（镜像原始格式、ground truth 与 mock 相关性基准）"
```

---

### Task 4: MovieLens 适配器

**Files:**
- Create: `genometag/data/movielens.py`
- Test: `tests/test_movielens.py`

**Interfaces:**
- Consumes: 无（纯 pandas）
- Produces:
  - `parse_title_year(title: str) -> tuple[str, int | None]`（`"Toy Story (1995)" → ("Toy Story", 1995)`）
  - `load_genome_scores(path) -> pd.DataFrame`（列 `movie_id, tag_id, relevance`）
  - `load_genome_tags(path) -> pd.DataFrame`（列 `tag_id, tag`）
  - `load_movies(path) -> pd.DataFrame`（列 `movie_id, title, year, genres: list[str]`；`genres` 为列表，`(no genres listed)` → `[]`）
  - `load_links(path) -> pd.DataFrame`（列 `movie_id, imdb_id`）
  - `load_user_tags(path) -> pd.DataFrame`（列 `user_id, movie_id, tag, timestamp`）

- [ ] **Step 1: 写失败测试**

`tests/test_movielens.py`：

```python
"""MovieLens 适配器：格式解析与规范化。"""
import pandas as pd

from genometag.data.movielens import (
    load_genome_scores, load_genome_tags, load_links, load_movies,
    load_user_tags, parse_title_year,
)


def test_parse_title_year():
    assert parse_title_year("Toy Story (1995)") == ("Toy Story", 1995)
    assert parse_title_year("No Year Here") == ("No Year Here", None)
    assert parse_title_year("City of God (2002)") == ("City of God", 2002)


def _write(tmp_path, name, rows, columns):
    p = tmp_path / name
    pd.DataFrame(rows, columns=columns).to_csv(p, index=False)
    return p


def test_load_genome_scores(tmp_path):
    p = _write(tmp_path, "genome-scores.csv",
               [(1, 5, 0.5), (1, 6, 0.9)], ["movieId", "tagId", "relevance"])
    df = load_genome_scores(p)
    assert list(df.columns) == ["movie_id", "tag_id", "relevance"]
    assert df["movie_id"].dtype.kind == "i"


def test_load_genome_tags(tmp_path):
    p = _write(tmp_path, "genome-tags.csv", [(5, "dark")], ["tagId", "tag"])
    assert list(load_genome_tags(p).columns) == ["tag_id", "tag"]


def test_load_movies_genres_list(tmp_path):
    p = _write(tmp_path, "movies.csv",
               [(1, "Toy Story (1995)", "Animation|Children"),
                (2, "Unknown (2001)", "(no genres listed)")],
               ["movieId", "title", "genres"])
    df = load_movies(p)
    assert df.loc[0, "genres"] == ["Animation", "Children"]
    assert df.loc[1, "genres"] == []          # (no genres listed) → 空列表
    assert df.loc[0, "year"] == 1995
    assert df.loc[0, "title"] == "Toy Story"


def test_load_links_and_user_tags(tmp_path):
    links = _write(tmp_path, "links.csv", [(1, 114709, 862)], ["movieId", "imdbId", "tmdbId"])
    assert list(load_links(links).columns) == ["movie_id", "imdb_id"]
    tags = _write(tmp_path, "tags.csv",
                  [(9, 1, "classic", 1700000000)], ["userId", "movieId", "tag", "timestamp"])
    df = load_user_tags(tags)
    assert list(df.columns) == ["user_id", "movie_id", "tag", "timestamp"]
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_movielens.py -v`
Expected: FAIL（ImportError）

- [ ] **Step 3: 实现 movielens.py**

`genometag/data/movielens.py`：

```python
"""MovieLens 适配器：genome-scores/tags、movies、links、tags（用户标签）。

统一输出规范列名（snake_case），movie_id/tag_id 为 int。原始文件格式稳定
（ml-latest 与 ml-25m 的 genome 文件结构相同）。
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

_TITLE_YEAR_RE = re.compile(r"^(.*) \((\d{4})\)$")


def parse_title_year(title: str) -> tuple[str, int | None]:
    """'Toy Story (1995)' → ('Toy Story', 1995)；无年份 → (原标题, None)。"""
    m = _TITLE_YEAR_RE.match(title.strip())
    if m:
        return m.group(1).strip(), int(m.group(2))
    return title.strip(), None


def load_genome_scores(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.rename(columns={"movieId": "movie_id", "tagId": "tag_id"})[
        ["movie_id", "tag_id", "relevance"]]


def load_genome_tags(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.rename(columns={"tagId": "tag_id"})[["tag_id", "tag"]]


def load_movies(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame({
        "movie_id": df["movieId"],
        "title": df["title"],
    })
    parsed = out["title"].map(parse_title_year)
    out["title"] = parsed.map(lambda t: t[0])
    out["year"] = parsed.map(lambda t: t[1]).astype("Int64")
    out["genres"] = df["genres"].map(
        lambda g: [] if g == "(no genres listed)" else g.split("|"))
    return out


def load_links(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.rename(columns={"movieId": "movie_id", "imdbId": "imdb_id"})[
        ["movie_id", "imdb_id"]]


def load_user_tags(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.rename(columns={"userId": "user_id", "movieId": "movie_id"})[
        ["user_id", "movie_id", "tag", "timestamp"]]
```

- [ ] **Step 4: 运行测试**

Run: `.venv/Scripts/python -m pytest tests/test_movielens.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add genometag/data/movielens.py tests/test_movielens.py
git commit -m "feat: MovieLens 适配器（genome/movies/links/tags 解析与规范化）"
```

---

### Task 5: MovieNet 适配器

**Files:**
- Create: `genometag/data/movienet.py`
- Test: `tests/test_movienet.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `@dataclass MovieMultimodal`：`movienet_id: str, title: str, year: int | None, imdb_id: str | None, keyframes: list[Path], script: str | None, subtitles: str | None`
  - `load_movienet(root: Path) -> list[MovieMultimodal]`（root = MovieNet 数据根；keyframes 为绝对路径、按文件名排序）
  - `load_srt(path: Path) -> str`（SRT → 纯文本台词，去序号/时间轴/空行）
  - 列名常量 `MOVIES_COLUMNS`（集中定义假定格式，真实数据到位后只改这里）

- [ ] **Step 1: 写失败测试**

`tests/test_movienet.py`：

```python
"""MovieNet 适配器：假定格式解析（合成样本即该格式的镜像）。"""
from genometag.data.movienet import MovieMultimodal, load_movienet, load_srt
from genometag.data.sample import make_synthetic_dataset


def test_load_movienet_from_synthetic(tmp_path):
    make_synthetic_dataset(tmp_path)
    movies = load_movienet(tmp_path / "raw" / "movienet")
    assert len(movies) == 5
    m = next(x for x in movies if x.movienet_id == "mn01")
    assert m.title == "Crimson Horizon"
    assert m.year == 2018
    assert m.imdb_id == "tt0000001"
    assert len(m.keyframes) == 8
    assert all(p.suffix == ".jpg" for p in m.keyframes)
    assert m.script is not None and "CRIMSON HORIZON" in m.script
    assert m.subtitles is not None and "change everything" in m.subtitles


def test_load_movienet_missing_optional_files(tmp_path):
    make_synthetic_dataset(tmp_path)
    # 删除 mn01 的剧本 → script=None 但不崩溃
    (tmp_path / "raw" / "movienet" / "scripts" / "mn01.txt").unlink()
    movies = load_movienet(tmp_path / "raw" / "movienet")
    m = next(x for x in movies if x.movienet_id == "mn01")
    assert m.script is None
    assert m.subtitles is not None


def test_load_srt(tmp_path):
    srt = tmp_path / "a.srt"
    srt.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nWorld\n", encoding="utf-8")
    assert load_srt(srt) == "Hello\nWorld"
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_movienet.py -v`
Expected: FAIL（ImportError）

- [ ] **Step 3: 实现 movienet.py**

`genometag/data/movienet.py`：

```python
"""MovieNet 适配器：唯一接触 MovieNet 原始格式的模块。

假定格式（合成样本即此格式的镜像；真实数据下载后如列名/布局有出入，
只改本文件的常量与 load_movienet，下游不动）:

    <root>/movies.csv              # movie_id,title,year,imdb_id
    <root>/keyframes/<id>/*.jpg
    <root>/scripts/<id>.txt        # 可选缺失
    <root>/subtitles/<id>.srt      # 可选缺失
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

MOVIES_COLUMNS = {"movie_id": "movie_id", "title": "title",
                  "year": "year", "imdb_id": "imdb_id"}

_SRT_TIME = re.compile(r"^\d{2}:\d{2}:\d{2}")


def load_srt(path: Path) -> str:
    """SRT → 纯文本（去序号、时间轴行、空行）。"""
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8", errors="replace").splitlines()]
    out = []
    for ln in lines:
        if not ln or ln.isdigit() or _SRT_TIME.match(ln) or "-->" in ln:
            continue
        out.append(ln)
    return "\n".join(out)


@dataclass
class MovieMultimodal:
    movienet_id: str
    title: str
    year: int | None
    imdb_id: str | None
    keyframes: list[Path]
    script: str | None
    subtitles: str | None


def _read_optional(path: Path) -> str | None:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else None


def load_movienet(root: Path) -> list[MovieMultimodal]:
    root = Path(root)
    df = pd.read_csv(root / "movies.csv").rename(columns=MOVIES_COLUMNS)
    movies: list[MovieMultimodal] = []
    for _, row in df.iterrows():
        mid = str(row["movie_id"])
        frames = sorted((root / "keyframes" / mid).glob("*.jpg"))
        script_path = root / "scripts" / f"{mid}.txt"
        sub_path = root / "subtitles" / f"{mid}.srt"
        year = row.get("year")
        movies.append(MovieMultimodal(
            movienet_id=mid,
            title=str(row["title"]),
            year=int(year) if pd.notna(year) else None,
            imdb_id=str(row["imdb_id"]) if pd.notna(row.get("imdb_id")) else None,
            keyframes=frames,
            script=_read_optional(script_path),
            subtitles=load_srt(sub_path) if sub_path.exists() else None,
        ))
    return movies
```

- [ ] **Step 4: 运行测试**

Run: `.venv/Scripts/python -m pytest tests/test_movienet.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add genometag/data/movienet.py tests/test_movienet.py
git commit -m "feat: MovieNet 适配器（假定格式隔离在单文件，含 SRT 解析）"
```

---

### Task 6: 对齐、数据集构建与数据校验

**Files:**
- Create: `genometag/data/align.py`（align + build_dataset）
- Create: `scripts/01_verify_data.py`、`scripts/02_build_dataset.py`（薄封装）
- Test: `tests/test_align.py`

**Interfaces:**
- Consumes: `load_movies`、`load_links`、`load_genome_scores`、`load_genome_tags`（Task 4）；`load_movienet`（Task 5）；`Paths`（Task 2）
- Produces:
  - `normalize_imdb(value) -> int | None`（`"tt0123456"→123456`，`12345.0→12345`，非法→None）
  - `@dataclass AlignResult: aligned: pd.DataFrame, report: str`
  - `align(movies_ml, links, movienet_movies) -> AlignResult`（aligned 列：`movie_id, title, year, genres, imdb_id, movienet_id, match_method`，仅含匹配成功的行）
  - `build_dataset(paths: Paths) -> dict[str, int]`（在 align.py；读 raw、写 interim/processed，返回 `{"movies": N, "genome_rows": N}`）——02 脚本与 Task 12 测试 fixture 均复用
  - `verify_data(paths) -> list[str]`（在脚本 01；返回缺失文件清单；空 = 通过）
  - 产物 schema：`aligned.parquet`（同上列）；`dataset.parquet`（`movie_id, title, year, genres, imdb_id, movienet_id, keyframes: list[str](相对 movienet 根), script_path, subtitles_path`）；`genome_scores.parquet`（`movie_id, tag_id, relevance`，仅对齐电影）；`genome_tags.parquet`（`tag_id, tag`）

- [ ] **Step 1: 写失败测试**

`tests/test_align.py`：

```python
"""对齐：IMDb 规范化、主键匹配、翻拍歧义、报告内容。"""
import pandas as pd
import pytest

from genometag.data.align import AlignResult, align, normalize_imdb


def test_normalize_imdb():
    assert normalize_imdb("tt0123456") == 123456
    assert normalize_imdb("123456") == 123456
    assert normalize_imdb(123456) == 123456
    assert normalize_imdb(123456.0) == 123456     # pandas 读 int 列可能 float 化
    assert normalize_imdb("tt0000001") == 1
    assert normalize_imdb("junk") is None
    assert normalize_imdb(None) is None


def _movies_ml():
    return pd.DataFrame([
        {"movie_id": 1, "title": "Alpha", "year": 2001, "genres": ["Drama"]},
        {"movie_id": 2, "title": "Beta", "year": 2002, "genres": ["Action"]},
        {"movie_id": 3, "title": "Gamma", "year": 2003, "genres": ["Crime"]},
    ])


def _links():
    return pd.DataFrame([{"movie_id": 1, "imdb_id": 111},
                         {"movie_id": 2, "imdb_id": 222}])  # 3 无 IMDb


def test_align_imdb_primary():
    from genometag.data.movienet import MovieMultimodal
    mn = [MovieMultimodal("mnA", "Alpha", 2001, "tt0000111", [], None, None),
          MovieMultimodal("mnB", "Beta", 2002, "tt0000222", [], None, None)]
    result = align(_movies_ml(), _links(), mn)
    df = result.aligned
    assert set(df["match_method"]) == {"imdb"}
    assert list(df["movie_id"]) == [1, 2]
    assert "IMDb 2" in result.report            # 报告含匹配统计


def test_align_title_fallback_for_missing_imdb():
    from genometag.data.movienet import MovieMultimodal
    mn = [MovieMultimodal("mnC", "gamma ", 2003, None, [], None, None)]  # 大小写/空格差异
    result = align(_movies_ml(), _links(), mn)
    row = result.aligned[result.aligned["movie_id"] == 3]
    assert len(row) == 1
    assert row.iloc[0]["match_method"] == "title"


def test_ambiguous_remake_unmatched():
    """同名同年多部（翻拍撞年份）→ 不匹配并写进报告歧义清单。"""
    from genometag.data.movienet import MovieMultimodal
    mn = [MovieMultimodal("mnX", "Gamma", 2003, None, [], None, None),
          MovieMultimodal("mnY", "Gamma", 2003, None, [], None, None)]
    result = align(_movies_ml(), _links(), mn)
    assert result.aligned[result.aligned["movie_id"] == 3].empty
    assert "Gamma" in result.report


def _synthetic_paths(tmp_path):
    """为 tmp_path 构建指向合成数据的 Paths（Task 12 fixture 复用同一模式）。"""
    from genometag.config import Paths
    return Paths(
        root=tmp_path,
        movienet=tmp_path / "raw" / "movienet",
        movielens=tmp_path / "raw" / "movielens",
        interim=tmp_path / "interim", processed=tmp_path / "processed",
        cache=tmp_path / ".cache", results=tmp_path / "results" / "x",
        tag_groups=tmp_path / "processed" / "tag_groups.json",
        dataset=tmp_path / "processed" / "dataset.parquet",
        genome_scores=tmp_path / "processed" / "genome_scores.parquet",
        genome_tags=tmp_path / "processed" / "genome_tags.parquet",
        aligned=tmp_path / "interim" / "aligned.parquet")


def test_build_dataset_from_synthetic(tmp_path):
    from genometag.data.align import build_dataset
    from genometag.data.sample import make_synthetic_dataset
    make_synthetic_dataset(tmp_path)
    paths = _synthetic_paths(tmp_path)
    stats = build_dataset(paths)
    assert stats == {"movies": 5, "genome_rows": 120}
    ds = pd.read_parquet(paths.dataset)
    assert len(ds) == 5
    assert ds.loc[0, "keyframes"][0].startswith("keyframes")   # 相对路径
    assert (paths.interim / "align_report.md").exists()
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_align.py -v`
Expected: FAIL（ImportError）

- [ ] **Step 3: 实现 align.py**

`genometag/data/align.py`：

```python
"""MovieNet ↔ MovieLens 对齐：IMDb 主键优先，标题+年份 fallback。

产出 aligned 表（仅匹配成功行）与人类可读的对齐报告（匹配率、未匹配清单、歧义清单）。
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import pandas as pd

from genometag.data.movienet import MovieMultimodal


def normalize_imdb(value) -> int | None:
    """'tt0123456' / 123456 / 123456.0 → 123456；非法 → None。"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip().lower().removeprefix("tt")
    if s.endswith(".0"):  # pandas float 化
        s = s[:-2]
    s = s.lstrip("0") or "0"
    return int(s) if s.isdigit() else None


def _norm_title(title: str) -> str:
    """标题规范化：NFKC、小写、去标点、压缩空白。"""
    t = unicodedata.normalize("NFKC", str(title)).lower()
    t = re.sub(r"[^\w\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


@dataclass
class AlignResult:
    aligned: pd.DataFrame
    report: str


def align(movies_ml: pd.DataFrame, links: pd.DataFrame,
          movienet_movies: list[MovieMultimodal]) -> AlignResult:
    imdb_to_mn: dict[int, MovieMultimodal] = {}
    for m in movienet_movies:
        norm = normalize_imdb(m.imdb_id)
        if norm is not None:
            imdb_to_mn[norm] = m

    link_map: dict[int, int] = {}  # movie_id(ML) -> imdb(int)
    for _, r in links.iterrows():
        norm = normalize_imdb(r["imdb_id"])
        if norm is not None:
            link_map[int(r["movie_id"])] = norm

    # 标题索引：norm_title|year -> [MovieMultimodal]
    title_idx: dict[tuple[str, int], list[MovieMultimodal]] = {}
    for m in movienet_movies:
        if m.year is not None:
            title_idx.setdefault((_norm_title(m.title), m.year), []).append(m)

    rows = []
    ambiguous, unmatched_ml = [], []
    imdb_hits = title_hits = 0
    for _, ml in movies_ml.iterrows():
        matched, method = None, None
        imdb = link_map.get(int(ml["movie_id"]))
        if imdb is not None and imdb in imdb_to_mn:
            matched, method = imdb_to_mn[imdb], "imdb"
        elif ml["year"] is not None and pd.notna(ml["year"]):
            cands = title_idx.get((_norm_title(ml["title"]), int(ml["year"])), [])
            if len(cands) == 1:
                matched, method = cands[0], "title"
            elif len(cands) > 1:
                ambiguous.append(f'{ml["title"]} ({int(ml["year"])})')
        if matched is None:
            unmatched_ml.append(f'{ml["movie_id"]}: {ml["title"]} ({ml["year"]})')
            continue
        if method == "imdb":
            imdb_hits += 1
        else:
            title_hits += 1
        rows.append({"movie_id": int(ml["movie_id"]), "title": ml["title"],
                     "year": int(ml["year"]) if pd.notna(ml["year"]) else None,
                     "genres": list(ml["genres"]), "imdb_id": imdb,
                     "movienet_id": matched.movienet_id, "match_method": method})

    n_total = len(movies_ml)
    n_matched = len(rows)
    matched_mn = {r["movienet_id"] for r in rows}
    unmatched_mn = [m.movienet_id for m in movienet_movies if m.movienet_id not in matched_mn]
    report = (
        f"# 对齐报告\n\n"
        f"- MovieLens 电影总数: {n_total}\n"
        f"- 匹配成功: {n_matched}（IMDb {imdb_hits} + 标题 {title_hits}）\n"
        f"- 匹配率: {n_matched / n_total:.1%}\n\n"
        f"## 歧义（同名同年多部，未匹配）\n"
        + ("\n".join(f"- {a}" for a in ambiguous) or "- 无") + "\n\n"
        f"## 未匹配的 MovieLens 电影（前 50）\n"
        + ("\n".join(f"- {u}" for u in unmatched_ml[:50]) or "- 无") + "\n\n"
        f"## 未匹配的 MovieNet 电影（前 50）\n"
        + ("\n".join(f"- {u}" for u in unmatched_mn[:50]) or "- 无") + "\n"
    )
    return AlignResult(aligned=pd.DataFrame(rows), report=report)
```

- [ ] **Step 4: 实现 build_dataset（align.py 末尾追加）与脚本 01/02**

`scripts/01_verify_data.py`：

```python
"""01 · 数据完整性校验：必需文件存在、行数、年份分布（spec §9 风险的检查点）。"""
from __future__ import annotations

import argparse

import pandas as pd

from genometag.config import load_config, paths_for

REQUIRED_FILES = [
    "genome-scores.csv", "genome-tags.csv", "movies.csv", "links.csv", "tags.csv",
]


def verify_data(paths) -> list[str]:
    """返回问题描述清单；空列表 = 通过。"""
    problems: list[str] = []
    for name in REQUIRED_FILES:
        if not (paths.movielens / name).exists():
            problems.append(f"缺少 MovieLens 文件: {paths.movielens / name}")
    if not (paths.movienet / "movies.csv").exists():
        problems.append(f"缺少 MovieNet 元数据: {paths.movienet / 'movies.csv'}")
    if problems:
        return problems
    # 年份分布（对齐可行性与时序分割点的关键输入）
    try:
        mn = pd.read_csv(paths.movienet / "movies.csv")
        if "year" in mn.columns:
            dist = mn["year"].dropna().astype(int).value_counts().sort_index()
            print("[01] MovieNet 年份分布（前 10 / 后 10）：")
            print(dist.head(10).to_string())
            print("...")
            print(dist.tail(10).to_string())
            latest = int(dist.index.max())
            print(f"[01] 最新年份: {latest}（检查是否满足时序分割需要）")
    except Exception as e:  # noqa: BLE001 —— 校验脚本要报告而非崩溃
        problems.append(f"MovieNet movies.csv 解析失败: {e}")
    return problems


def main(config_path: str = "configs/default.yaml") -> int:
    config = load_config(config_path)
    problems = verify_data(paths_for(config))
    if problems:
        print("[01] 数据校验失败：")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("[01] 数据校验通过")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    raise SystemExit(main(parser.parse_args().config))
```

align.py 顶部补 import：`from genometag.config import Paths` 与
`from genometag.data.movielens import (load_genome_scores, load_genome_tags, load_links, load_movies)`，末尾追加：

```python
def build_dataset(paths: Paths) -> dict[str, int]:
    """读 raw → 对齐 → 写 interim/processed，返回统计。02 脚本与测试 fixture 复用。"""
    movies_ml = load_movies(paths.movielens / "movies.csv")
    links = load_links(paths.movielens / "links.csv")
    mn_movies = load_movienet(paths.movienet)

    result = align(movies_ml, links, mn_movies)
    paths.interim.mkdir(parents=True, exist_ok=True)
    paths.processed.mkdir(parents=True, exist_ok=True)
    result.aligned.to_parquet(paths.aligned, index=False)
    (paths.interim / "align_report.md").write_text(result.report, encoding="utf-8")

    # 多模态文件引用（相对 movienet 根，便于仓库迁移）
    mn_by_id = {m.movienet_id: m for m in mn_movies}
    dataset_rows = []
    for _, r in result.aligned.iterrows():
        m = mn_by_id[r["movienet_id"]]
        dataset_rows.append({
            "movie_id": r["movie_id"], "title": r["title"], "year": r["year"],
            "genres": r["genres"], "imdb_id": r["imdb_id"],
            "movienet_id": r["movienet_id"],
            "keyframes": [str(p.relative_to(paths.movienet)) for p in m.keyframes],
            "script_path": f'scripts/{m.movienet_id}.txt' if m.script else None,
            "subtitles_path": f'subtitles/{m.movienet_id}.srt' if m.subtitles else None,
        })
    dataset = pd.DataFrame(dataset_rows)
    dataset.to_parquet(paths.dataset, index=False)

    scores = load_genome_scores(paths.movielens / "genome-scores.csv")
    scores = scores[scores["movie_id"].isin(set(dataset["movie_id"]))]
    scores.to_parquet(paths.genome_scores, index=False)
    load_genome_tags(paths.movielens / "genome-tags.csv").to_parquet(
        paths.genome_tags, index=False)
    return {"movies": len(dataset), "genome_rows": len(scores)}
```

`scripts/02_build_dataset.py`（薄封装）：

```python
"""02 · 对齐 + 产出实验数据集（逻辑在 genometag.data.align.build_dataset）。"""
from __future__ import annotations

import argparse

from genometag.config import load_config, paths_for
from genometag.data.align import build_dataset


def main(config_path: str = "configs/default.yaml") -> int:
    stats = build_dataset(paths_for(load_config(config_path)))
    print(f"[02] 数据集构建完成: {stats['movies']} 部电影, "
          f"{stats['genome_rows']} 行 genome scores")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    raise SystemExit(main(parser.parse_args().config))
```

- [ ] **Step 5: 运行测试**

Run: `.venv/Scripts/python -m pytest tests/test_align.py -v`
Expected: 5 passed

再手动验证全链路（合成样本）：

```bash
.venv/Scripts/python scripts/00_make_synthetic.py
.venv/Scripts/python scripts/01_verify_data.py --config configs/smoke.yaml
.venv/Scripts/python scripts/02_build_dataset.py --config configs/smoke.yaml
```

Expected: 01 通过；02 打印 `5 部电影, 120 行 genome scores`；`data/synthetic/interim/align_report.md` 存在

- [ ] **Step 6: Commit**

```bash
git add genometag/data/align.py scripts/01_verify_data.py scripts/02_build_dataset.py tests/test_align.py
git commit -m "feat: IMDb 主键对齐 + 数据集构建 + 数据校验（含年份分布报告）"
```

---

### Task 7: LLM 调用缓存

**Files:**
- Create: `genometag/annotate/cache.py`
- Test: `tests/test_cache.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `cache_key(messages: list[dict], model: str, params: dict, run_id: str) -> str`（SHA256 十六进制）
  - `class ResponseCache`：`__init__(root: Path)`；`get(key: str) -> dict | None`；`put(key: str, response: dict) -> None`（值 = 完整原始响应 JSON，落盘为 `<root>/<key>.json`）

- [ ] **Step 1: 写失败测试**

`tests/test_cache.py`：

```python
"""缓存：键稳定性、run_id/参数维度、落盘往返。"""
from genometag.annotate.cache import ResponseCache, cache_key

MSGS = [{"role": "user", "content": "hi"}]


def test_key_stable():
    a = cache_key(MSGS, "gpt-4o", {"temperature": 0.7}, "run-0")
    b = cache_key(MSGS, "gpt-4o", {"temperature": 0.7}, "run-0")
    assert a == b and len(a) == 64


def test_key_differs_by_run_id():
    """Review Focus #1：同 prompt 不同轮次必须不同键，否则多轮取平均失效。"""
    a = cache_key(MSGS, "gpt-4o", {}, "run-0")
    b = cache_key(MSGS, "gpt-4o", {}, "run-1")
    assert a != b


def test_key_changes_with_params():
    a = cache_key(MSGS, "gpt-4o", {"temperature": 0.0}, "run-0")
    b = cache_key(MSGS, "gpt-4o", {"temperature": 1.0}, "run-0")
    assert a != b


def test_put_get_roundtrip(tmp_path):
    cache = ResponseCache(tmp_path)
    assert cache.get("nonexistent") is None
    resp = {"choices": [{"message": {"content": "{\"a\": 1}"}}], "usage": {"prompt_tokens": 3}}
    cache.put("k1", resp)
    assert cache.get("k1") == resp
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_cache.py -v`
Expected: FAIL（ImportError）

- [ ] **Step 3: 实现 cache.py**

`genometag/annotate/cache.py`：

```python
"""LLM 调用磁盘缓存：成本控制、断点续跑、可复现的公共基础。

键 = SHA256(messages + model + 采样参数 + run_id)。run_id 必须入键：
多轮取平均/一致性分析会对同一 prompt 发多次请求。
值 = 完整原始响应（审计与换解析逻辑重算不需要重新付费）。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def cache_key(messages: list[dict], model: str, params: dict, run_id: str) -> str:
    payload = json.dumps(
        {"messages": messages, "model": model, "params": params, "run_id": run_id},
        ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


class ResponseCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> dict | None:
        p = self._path(key)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def put(self, key: str, response: dict) -> None:
        self._path(key).write_text(
            json.dumps(response, ensure_ascii=False, indent=1), encoding="utf-8")
```

- [ ] **Step 4: 运行测试**

Run: `.venv/Scripts/python -m pytest tests/test_cache.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add genometag/annotate/cache.py tests/test_cache.py
git commit -m "feat: LLM 调用磁盘缓存（run_id 入键，原始响应全量保留）"
```

---

### Task 8: LLM 客户端（OpenAI 兼容）

**Files:**
- Create: `genometag/annotate/client.py`
- Test: `tests/test_client.py`

**Interfaces:**
- Consumes: 无（openai SDK 懒导入，仅真实调用时需要）
- Produces:
  - `@dataclass LLMResponse: content: str, usage: dict, raw: dict, ok: bool`（`ok=False` 表示传输层失败，`content=""`）
  - `OpenAIClient(model, base_url_env, api_key_env="OPENAI_API_KEY", temperature=0.7, request_interval=0.5, transport=None, max_retries=3)`——`transport: Callable[[model, messages, temperature, response_format], dict]`（注入用，None 时构建 openai SDK transport）
  - `OpenAIClient.complete(messages: list[dict], run_id: str) -> LLMResponse`——异常指数退避重试至多 `max_retries` 次；`response_format` 先带 `json_object` 请求，端点不支持时降级去掉重发一次；每次真实调用前 `sleep(request_interval)`
  - `make_client(config: AppConfig) -> LLMClient`（`config.mock=True` → `MockClient`，否则 `OpenAIClient`；MockClient 由 Task 9 提供，此处懒导入）

- [ ] **Step 1: 写失败测试**

`tests/test_client.py`：

```python
"""客户端：transport 注入测试（无网络）。重试、response_format 降级、usage 提取。"""
import pytest

from genometag.annotate.client import LLMResponse, OpenAIClient


def _ok_transport(content='{"a": 1}'):
    def transport(model, messages, temperature, response_format):
        assert response_format is not None or True  # 降级路径会传 None
        return {"choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    return transport


def test_complete_success():
    client = OpenAIClient(model="m", base_url_env="X", transport=_ok_transport(),
                          request_interval=0.0)
    resp = client.complete([{"role": "user", "content": "hi"}], run_id="run-0")
    assert isinstance(resp, LLMResponse)
    assert resp.ok and resp.content == '{"a": 1}'
    assert resp.usage == {"prompt_tokens": 10, "completion_tokens": 5}


def test_retry_on_transport_error_then_success():
    calls = {"n": 0}

    def flaky(model, messages, temperature, response_format):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("网络抖动")
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    client = OpenAIClient(model="m", base_url_env="X", transport=flaky,
                          request_interval=0.0)
    resp = client.complete([{"role": "user", "content": "hi"}], run_id="r")
    assert resp.ok and calls["n"] == 2


def test_gives_up_after_max_retries():
    def always_fail(model, messages, temperature, response_format):
        raise ConnectionError("down")

    client = OpenAIClient(model="m", base_url_env="X", transport=always_fail,
                          request_interval=0.0, max_retries=2)
    resp = client.complete([{"role": "user", "content": "hi"}], run_id="r")
    assert not resp.ok and resp.content == ""


def test_response_format_fallback():
    """端点不支持 json_object → 去掉 response_format 重发一次。"""
    calls = {"n": 0}

    def picky(model, messages, temperature, response_format):
        calls["n"] += 1
        if response_format is not None:
            raise ValueError("response_format is not supported")
        return {"choices": [{"message": {"content": "plain"}}], "usage": {}}

    client = OpenAIClient(model="m", base_url_env="X", transport=picky,
                          request_interval=0.0)
    resp = client.complete([{"role": "user", "content": "hi"}], run_id="r")
    assert resp.ok and resp.content == "plain" and calls["n"] == 2


def test_missing_env_raises():
    import os
    client = OpenAIClient(model="m", base_url_env="NO_SUCH_ENV_XYZ")
    with pytest.raises(RuntimeError, match="NO_SUCH_ENV_XYZ"):
        client._build_transport()  # 环境变量缺失时构建即失败（不静默）
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_client.py -v`
Expected: FAIL（ImportError）

- [ ] **Step 3: 实现 client.py**

`genometag/annotate/client.py`：

```python
"""OpenAI 兼容 LLM 客户端：可注入 transport 便于测试，懒导入 openai SDK。

不做输出解析——parse_scores 在 prompts.py（Task 10），重试解析在 pipeline（Task 12）。
本层只负责：传输、重试、response_format 降级、限速间隔、usage 记录。
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable

Transport = Callable[[str, list[dict], float, dict | None], dict]


@dataclass
class LLMResponse:
    content: str
    usage: dict
    raw: dict
    ok: bool


class OpenAIClient:
    def __init__(self, model: str, base_url_env: str,
                 api_key_env: str = "OPENAI_API_KEY", temperature: float = 0.7,
                 request_interval: float = 0.5, transport: Transport | None = None,
                 max_retries: int = 3):
        self.model = model
        self.base_url_env = base_url_env
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.request_interval = request_interval
        self.max_retries = max_retries
        self._transport = transport

    def _build_transport(self) -> Transport:
        base_url = os.environ.get(self.base_url_env)
        api_key = os.environ.get(self.api_key_env)
        if not base_url or not api_key:
            raise RuntimeError(
                f"环境变量 {self.base_url_env} / {self.api_key_env} 未设置"
                "（复制 .env.example 为 .env 并填入）")
        from openai import OpenAI  # 懒导入：mock/测试路径不需要安装后首次 import 的开销

        sdk = OpenAI(base_url=base_url, api_key=api_key)

        def transport(model: str, messages: list[dict], temperature: float,
                      response_format: dict | None) -> dict:
            kwargs: dict = {"model": model, "messages": messages,
                            "temperature": temperature}
            if response_format is not None:
                kwargs["response_format"] = response_format
            return sdk.chat.completions.create(**kwargs).model_dump()

        return transport

    def complete(self, messages: list[dict], run_id: str) -> LLMResponse:
        """run_id 仅用于日志语境；缓存键在 pipeline 层构造。

        重试语义：json_object 不被端点支持 → 降级为无 response_format 重试（不耗退避）；
        其他传输异常 → 指数退避重试至多 max_retries 次，仍失败返回 ok=False。
        """
        if self._transport is None:
            self._transport = self._build_transport()
        time.sleep(self.request_interval)

        last_err: Exception | None = None
        use_json = True
        for attempt in range(self.max_retries):
            try:
                return self._call(self._transport, messages, allow_json=use_json)
            except Exception as e:  # noqa: BLE001 —— 传输层任何异常统一处理
                if use_json and _detect_json_format_error(e):
                    use_json = False   # 端点不支持 json_object → 降级后重试
                    continue
                last_err = e
                time.sleep(0.5 * (2 ** attempt))
        return LLMResponse(content="", usage={}, raw={"error": str(last_err)}, ok=False)

    def _call(self, transport: Transport, messages: list[dict],
              allow_json: bool) -> LLMResponse:
        rf = {"type": "json_object"} if allow_json else None
        raw = transport(self.model, messages, self.temperature, rf)
        content = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
        return LLMResponse(content=content or "", usage=raw.get("usage", {}),
                           raw=raw, ok=True)


def _detect_json_format_error(e: Exception) -> bool:
    """端点不支持 response_format=json_object 的错误文本识别。"""
    text = str(e).lower()
    return "response_format" in text or "json_object" in text or "json mode" in text


def make_client(config):
    """工厂：mock 配置返回 MockClient（不联网、不碰缓存）。"""
    if config.mock:
        from genometag.annotate.mock import MockClient
        return MockClient()
    return OpenAIClient(
        model=config.model.model, base_url_env=config.model.base_url_env,
        api_key_env=config.model.api_key_env, temperature=config.model.temperature,
        request_interval=config.model.request_interval)
```

- [ ] **Step 4: 运行测试**

Run: `.venv/Scripts/python -m pytest tests/test_client.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add genometag/annotate/client.py tests/test_client.py
git commit -m "feat: OpenAI 兼容客户端（transport 注入、退避重试、json 格式降级、限速）"
```

---

### Task 9: Mock 客户端

**Files:**
- Create: `genometag/annotate/mock.py`
- Test: `tests/test_mock.py`

**Interfaces:**
- Consumes: `base_score`（Task 3 的 `genometag/data/sample.py`——唯一权威实现）；`prompts.MOVIE_TITLE_LINE` 与 `prompts.TAGS_BLOCK_START`（Task 10 提供——**注意循环依赖**：mock 需要这两个常量解析 prompt，而 prompts 不导入 mock。实现顺序：本任务先在 mock.py 内定义与 Task 10 约定一致的**字符串常量**（`_TITLE_RE`、`_TAGS_HEADER`），Task 10 实现 prompts.py 时保证 `MOVIE_TITLE_LINE.format(title=..., year=...)` 与 `TAGS_BLOCK_START` 生成的文本可被这两个正则解析；Task 10 的测试将锁定该契约）
- Produces:
  - `MockClient(correlation_noise: float = 0.05, model: str = "mock-1")`，实现 `complete(messages: list[dict], run_id: str) -> LLMResponse`（复用 Task 8 的 `LLMResponse`）
  - 行为：从 prompt 文本提取标题（正则 `电影：《(.+?)》（`）与标签列表（`## 需要打分的标签` 块内 `- (tag)：定义` 行）；每 tag 分数 = `clip(base_score(title, tag) + U(-noise, noise; hash(title|tag|run_id)))`；输出 `json.dumps`

- [ ] **Step 1: 写失败测试**

`tests/test_mock.py`：

```python
"""Mock 客户端：确定性、与合成 ground truth 的相关性、轮次噪声。"""
import json

import numpy as np

from genometag.annotate.mock import MockClient
from genometag.data.sample import SYNTHETIC_MOVIES, SYNTHETIC_TAGS, base_score

PROMPT = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": (
        "电影：《Crimson Horizon》（2018）\n\n"
        "## 剧本节选\nINT. ...\n\n"
        "## 需要打分的标签\n"
        "- neon visuals：画面以霓虹灯光为主\n"
        "- dark palette：整体色调偏暗\n"
        "- overrated：名过其实\n\n"
        "请输出 JSON。")},
]


def test_complete_returns_json_for_all_tags():
    client = MockClient()
    resp = client.complete(PROMPT, run_id="run-0")
    scores = json.loads(resp.content)
    assert set(scores) == {"neon visuals", "dark palette", "overrated"}
    assert all(0.0 <= v <= 1.0 for v in scores.values())
    assert resp.ok


def test_deterministic_same_run():
    a = MockClient().complete(PROMPT, run_id="run-0")
    b = MockClient().complete(PROMPT, run_id="run-0")
    assert a.content == b.content


def test_run_noise_varies_but_small():
    a = json.loads(MockClient().complete(PROMPT, "run-0").content)
    b = json.loads(MockClient().complete(PROMPT, "run-1").content)
    for tag in a:
        assert abs(a[tag] - b[tag]) <= 0.1  # correlation_noise=0.05 → |diff| ≤ 0.1


def test_correlation_with_synthetic_ground_truth():
    """合成 ground truth 与 mock 预测的 Pearson 相关 > 0.5（spec 的可控相关性）。"""
    client = MockClient()
    truth, pred = [], []
    for m in SYNTHETIC_MOVIES:
        tag_lines = "\n".join(f'- {t["tag"]}：{t["definition"]}' for t in SYNTHETIC_TAGS)
        prompt = [{"role": "user", "content": (
            f'电影：《{m["title"]}》（{m["year"]}）\n\n'
            f'## 需要打分的标签\n{tag_lines}\n\n请输出 JSON。')}]
        scores = json.loads(client.complete(prompt, "run-0").content)
        for t in SYNTHETIC_TAGS:
            truth.append(base_score(m["title"], t["tag"]))
            pred.append(scores[t["tag"]])
    r = np.corrcoef(truth, pred)[0, 1]
    assert r > 0.5
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_mock.py -v`
Expected: FAIL（ImportError）

- [ ] **Step 3: 实现 mock.py**

`genometag/annotate/mock.py`：

```python
"""确定性 Mock 客户端：测试与冒烟用，零网络、零缓存。

打分规则：score = clip(base_score(title, tag) + hash 噪声(run 维度), 0, 1)。
base_score 与合成样本 ground truth 共用（见 data/sample.py），
因此 mock 预测与 ground truth 天然相关、轮次间小幅波动——
这使指标函数在集成测试中产生非退化输出。
"""
from __future__ import annotations

import hashlib
import json
import re

from genometag.annotate.client import LLMResponse
from genometag.data.sample import base_score

_TITLE_RE = re.compile(r"电影：《(.+?)》（")
_TAG_LINE_RE = re.compile(r"^- (.+?)：.*$", re.MULTILINE)
_TAGS_HEADER = "## 需要打分的标签"


def _extract_text(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):  # 多模态 content parts
            parts.extend(p.get("text", "") for p in c if isinstance(p, dict))
    return "\n".join(parts)


def _hash_unit(key: str) -> float:
    h = hashlib.sha256(key.encode()).hexdigest()
    return int(h[:8], 16) / 0x1_0000_0000


class MockClient:
    def __init__(self, correlation_noise: float = 0.05, model: str = "mock-1"):
        self.correlation_noise = correlation_noise
        self.model = model

    def complete(self, messages: list[dict], run_id: str) -> LLMResponse:
        text = _extract_text(messages)
        m = _TITLE_RE.search(text)
        title = m.group(1) if m else "unknown"
        tags: list[str] = []
        if _TAGS_HEADER in text:
            block = text.split(_TAGS_HEADER, 1)[1]
            for line in block.splitlines():
                lm = _TAG_LINE_RE.match(line)
                if lm:
                    tags.append(lm.group(1))
        scores = {}
        for tag in tags:
            noise = (_hash_unit(f"noise|{title}|{tag}|{run_id}") - 0.5) * 2 * self.correlation_noise
            scores[tag] = min(1.0, max(0.0, base_score(title, tag) + noise))
        content = json.dumps(scores, ensure_ascii=False)
        return LLMResponse(content=content,
                           usage={"prompt_tokens": 100, "completion_tokens": 50},
                           raw={"mock": True}, ok=True)
```

- [ ] **Step 4: 运行测试**

Run: `.venv/Scripts/python -m pytest tests/test_mock.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add genometag/annotate/mock.py tests/test_mock.py
git commit -m "feat: 确定性 Mock 客户端（与合成 ground truth 可控相关）"
```

---

### Task 10: Prompt 组装与输出解析

**Files:**
- Create: `genometag/annotate/prompts.py`
- Test: `tests/test_prompts.py`、`tests/test_validation.py`

**Interfaces:**
- Consumes: `MovieMultimodal`（Task 5）；`ModelConfig`（Task 2）
- Produces:
  - 常量 `MOVIE_TITLE_LINE = "电影：《{title}》（{year}）"`、`TAGS_BLOCK_START = "## 需要打分的标签"`（**必须**与 Task 9 mock 的正则兼容——测试锁定）
  - `@dataclass TagDef: tag_id: int, tag: str, definition: str`
  - `@dataclass FewShot: title: str, year: int, scores: dict[str, float]`
  - `approx_tokens(text: str) -> int`（≈ `max(1, len(text)//4)`）
  - `truncate_to_budget(text: str, token_budget: int) -> str`
  - `sample_keyframes(frames: list[Path], max_frames: int) -> list[Path]`（均匀采样，保序）
  - `encode_image(path: Path) -> str`（base64）
  - `build_messages(movie: MovieMultimodal, tag_defs: list[TagDef], modalities: list[str], few_shots: list[FewShot], model_cfg: ModelConfig) -> list[dict]`（system + user；user content 为 `[{"type": "text", ...}, {"type": "image_url", ...} × N]`）
  - `parse_scores(text: str, expected_tags: list[str]) -> dict[str, float] | None`（任一 tag 缺失 / 非数值 / 越界 / JSON 不可解析 → None）

- [ ] **Step 1: 写失败测试**

`tests/test_prompts.py`：

```python
"""Prompt 组装：模态裁剪、帧采样、截断、mock 兼容契约、few-shot。"""
from pathlib import Path

from genometag.annotate.prompts import (
    FewShot, MOVIE_TITLE_LINE, TagDef, approx_tokens, build_messages,
    sample_keyframes, truncate_to_budget,
)
from genometag.config import ModelConfig
from genometag.data.movienet import MovieMultimodal


def _movie(tmp_path, *, script="S" * 2000, subs="D" * 1000):
    frames = []
    for i in range(10):
        p = tmp_path / f"f{i}.jpg"
        p.write_bytes(b"\xff\xd8fake")
        frames.append(p)
    return MovieMultimodal("mn1", "Test Film", 2020, "tt1", frames, script, subs)


TAGS = [TagDef(1, "dark palette", "色调偏暗"), TagDef(2, "overrated", "名过其实")]


def test_modality_filtering(tmp_path):
    movie = _movie(tmp_path)
    cfg = ModelConfig(max_frames=3)
    msgs = build_messages(movie, TAGS, ["script"], [], cfg)
    content = msgs[1]["content"]
    assert isinstance(content, list)
    assert len(content) == 1                       # 纯文本，无图像
    assert "剧本节选" in content[0]["text"]
    msgs_v = build_messages(movie, TAGS, ["video"], [], cfg)
    assert len(msgs_v[1]["content"]) == 1 + 3      # 文本 + 3 帧（10 帧采样到 3）


def test_sample_keyframes_even():
    frames = [Path(f"f{i}") for i in range(10)]
    assert sample_keyframes(frames, 3) == [Path("f0"), Path("f3"), Path("f6")]
    assert sample_keyframes(frames, 10) == frames   # 不超过则原样
    assert sample_keyframes(frames, 20) == frames


def test_truncate_to_budget():
    text = "a" * 1000
    assert truncate_to_budget(text, 100) == "a" * 400   # 4 字符/token
    assert truncate_to_budget("short", 10000) == "short"


def test_title_line_contract_with_mock():
    """Review Focus 契约：MOVIE_TITLE_LINE 生成的行必须能被 mock 的正则解析。"""
    import re
    line = MOVIE_TITLE_LINE.format(title="Crimson Horizon", year=2018)
    assert re.search(r"电影：《(.+?)》（", line).group(1) == "Crimson Horizon"


def test_few_shot_rendered(tmp_path):
    movie = _movie(tmp_path)
    fs = [FewShot("Old Film", 2015, {"dark palette": 0.9})]
    msgs = build_messages(movie, TAGS, ["script"], fs, ModelConfig(max_frames=2))
    text = msgs[1]["content"][0]["text"]
    assert "Old Film" in text and "dark palette: 0.9" in text
```

`tests/test_validation.py`：

```python
"""Review Focus #2：MLLM 输出解析的全分支。"""
from genometag.annotate.prompts import parse_scores

EXPECTED = ["dark palette", "overrated"]


def test_valid_json():
    assert parse_scores('{"dark palette": 0.8, "overrated": 0.1}', EXPECTED) == {
        "dark palette": 0.8, "overrated": 0.1}


def test_json_with_surrounding_text():
    text = '好的，以下是评分：\n{"dark palette": 0.5, "overrated": 0.2}\n以上。'
    assert parse_scores(text, EXPECTED) == {"dark palette": 0.5, "overrated": 0.2}


def test_missing_tag_returns_none():
    assert parse_scores('{"dark palette": 0.8}', EXPECTED) is None


def test_out_of_range_returns_none():
    assert parse_scores('{"dark palette": 1.5, "overrated": 0.2}', EXPECTED) is None
    assert parse_scores('{"dark palette": -0.1, "overrated": 0.2}', EXPECTED) is None


def test_non_numeric_returns_none():
    assert parse_scores('{"dark palette": "high", "overrated": 0.2}', EXPECTED) is None


def test_garbage_returns_none():
    assert parse_scores("完全不是 JSON", EXPECTED) is None
    assert parse_scores("[]", EXPECTED) is None
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_prompts.py tests/test_validation.py -v`
Expected: FAIL（ImportError）

- [ ] **Step 3: 实现 prompts.py**

`genometag/annotate/prompts.py`：

```python
"""Prompt 组装（模态裁剪、预算截断、few-shot）与输出解析。

文本块格式是 MockClient（annotate/mock.py）的解析契约：
- 电影标题行：MOVIE_TITLE_LINE
- 标签块：TAGS_BLOCK_START 后每行 "- {tag}：{definition}"
"""
from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path

from genometag.config import ModelConfig
from genometag.data.movienet import MovieMultimodal

SYSTEM_PROMPT = "你是电影内容分析专家。根据提供的电影内容为标签打关联度分，输出严格的 JSON 对象。"
MOVIE_TITLE_LINE = "电影：《{title}》（{year}）"
TAGS_BLOCK_START = "## 需要打分的标签"
_INSTRUCTIONS = (
    "对下面每一个标签，给出该电影与之的关联度评分（0.0-1.0，保留两位小数）。\n"
    "只输出 JSON 对象，键为标签名、值为分数，不要输出其他内容。"
)


@dataclass
class TagDef:
    tag_id: int
    tag: str
    definition: str


@dataclass
class FewShot:
    title: str
    year: int
    scores: dict[str, float]


def approx_tokens(text: str) -> int:
    """近似 token 数（英文约 4 字符/token；预算截断用，不追求精确）。"""
    return max(1, len(text) // 4)


def truncate_to_budget(text: str, token_budget: int) -> str:
    if approx_tokens(text) <= token_budget:
        return text
    return text[: token_budget * 4]


def sample_keyframes(frames: list[Path], max_frames: int) -> list[Path]:
    if len(frames) <= max_frames:
        return list(frames)
    step = len(frames) / max_frames
    return [frames[int(i * step)] for i in range(max_frames)]


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def build_messages(movie: MovieMultimodal, tag_defs: list[TagDef],
                   modalities: list[str], few_shots: list[FewShot],
                   model_cfg: ModelConfig) -> list[dict]:
    parts: list[str] = [
        MOVIE_TITLE_LINE.format(title=movie.title, year=movie.year if movie.year else "未知年份"),
        _INSTRUCTIONS,
    ]
    if few_shots:
        lines = ["## 示例（已知关联度的电影）"]
        for fs in few_shots:
            lines.append(f"《{fs.title}》（{fs.year}）：" + ", ".join(
                f"{t}: {s}" for t, s in fs.scores.items()))
        parts.append("\n".join(lines))

    sampled: list[Path] = []
    if "video" in modalities and movie.keyframes:
        sampled = sample_keyframes(movie.keyframes, model_cfg.max_frames)
        parts.append(f"## 关键帧（{len(sampled)} 张，见后附图像）")
    if "script" in modalities and movie.script:
        parts.append("## 剧本节选\n" + truncate_to_budget(movie.script, model_cfg.script_token_budget))
    if "dialogue" in modalities and movie.subtitles:
        parts.append("## 台词/字幕\n" + truncate_to_budget(movie.subtitles, model_cfg.dialogue_token_budget))

    tag_lines = [f"- {t.tag}：{t.definition}" for t in tag_defs]
    parts.append(TAGS_BLOCK_START + "\n" + "\n".join(tag_lines))

    content: list[dict] = [{"type": "text", "text": "\n\n".join(parts)}]
    for frame in sampled:
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encode_image(frame)}"}})
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content}]


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_scores(text: str, expected_tags: list[str]) -> dict[str, float] | None:
    """解析模型输出。任一 tag 缺失/非数值/越界/不可解析 → None（调用方决定重试）。"""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_RE.search(text)
        if not m:
            return None
        try:
            data = json.loads(m.group())
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    scores: dict[str, float] = {}
    for tag in expected_tags:
        if tag not in data:
            return None
        v = data[tag]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        if not 0.0 <= float(v) <= 1.0:
            return None
        scores[tag] = float(v)
    return scores
```

- [ ] **Step 4: 运行测试**

Run: `.venv/Scripts/python -m pytest tests/test_prompts.py tests/test_validation.py -v`
Expected: 8 + 6 passed

另跑 Task 9 的测试确认契约未破坏：

Run: `.venv/Scripts/python -m pytest tests/test_mock.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add genometag/annotate/prompts.py tests/test_prompts.py tests/test_validation.py
git commit -m "feat: Prompt 组装（模态裁剪/帧采样/预算截断/few-shot）与输出解析全分支"
```

---

### Task 11: Tag 分组

**Files:**
- Create: `genometag/tags/grouping.py`
- Create: `scripts/03_group_tags.py`
- Test: `tests/test_grouping.py`

**Interfaces:**
- Consumes: `LLMClient.complete`（Task 8/9 协议）、`ResponseCache`/`cache_key`（Task 7）、`make_client`（Task 8）
- Produces:
  - `build_grouping_messages(tag_batch: list[str], known_groups: list[str]) -> list[dict]`
  - `parse_grouping_response(text: str, expected: list[str]) -> dict[str, dict] | None`（`tag → {"group", "tag_type", "definition"}`；任一 tag 缺失或 `tag_type` 非法 → None）
  - `validate_coverage(parsed: dict[str, dict], expected: list[str]) -> list[str]`（缺失 tag 清单）
  - `group_tags_via_llm(tags: list[str], client, cache, model: str, batch_size: int = 150) -> pd.DataFrame`（列 `tag, group, tag_type, definition`；跨批合并，后续批 prompt 携带已有组名）
  - `load_tag_groups(path: Path) -> pd.DataFrame`（列 `tag_id, tag, group, tag_type, definition`）
  - 产物：`paths.tag_groups`（JSON，`{tag_id, tag, group, tag_type, definition}[]`）

- [ ] **Step 1: 写失败测试**

`tests/test_grouping.py`：

```python
"""Tag 分组：响应解析、覆盖校验、跨批合并、tag_groups 加载。"""
import json

import pandas as pd
import pytest

from genometag.tags.grouping import (
    build_grouping_messages, group_tags_via_llm, load_tag_groups,
    parse_grouping_response, validate_coverage,
)

GOOD = '{"dark": {"group": "视觉风格", "tag_type": "content", "definition": "色调偏暗"}, '\
       '"overrated": {"group": "受众偏好", "tag_type": "preference", "definition": "名过其实"}}'


def test_parse_grouping_response():
    parsed = parse_grouping_response(GOOD, ["dark", "overrated"])
    assert parsed["dark"]["tag_type"] == "content"
    assert parsed["overrated"]["group"] == "受众偏好"


def test_parse_missing_tag_returns_none():
    assert parse_grouping_response(GOOD, ["dark", "missing"]) is None


def test_parse_bad_type_returns_none():
    bad = json.dumps({"dark": {"group": "g", "tag_type": "weird", "definition": "d"}})
    assert parse_grouping_response(bad, ["dark"]) is None


def test_validate_coverage():
    parsed = json.loads(GOOD)
    assert validate_coverage(parsed, ["dark", "overrated", "tense"]) == ["tense"]


class _ScriptedClient:
    """按 prompt 中出现的 tag 顺序返回预置分组结果。"""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls = 0

    def complete(self, messages, run_id):
        self.calls += 1
        return _Resp(self.responses.pop(0))


class _Resp:
    def __init__(self, content):
        self.content, self.usage, self.raw, self.ok = content, {}, {}, True


def test_group_tags_via_llm_batches_merge():
    tags = ["dark", "overrated", "tense", "neon visuals"]
    r1 = json.dumps({t: {"group": f"G{i}", "tag_type": "content", "definition": "d"}
                     for i, t in enumerate(tags[:2])})
    r2 = json.dumps({t: {"group": f"G{i}", "tag_type": "preference", "definition": "d"}
                     for i, t in enumerate(tags[2:])})
    client = _ScriptedClient([r1, r2])
    df = group_tags_via_llm(tags, client, cache=None, model="m", batch_size=2)
    assert len(df) == 4 and client.calls == 2
    assert set(df.columns) == {"tag", "group", "tag_type", "definition"}


def test_load_tag_groups(tmp_path):
    data = [{"tag_id": 1, "tag": "dark", "group": "视觉风格",
             "tag_type": "content", "definition": "色调偏暗"}]
    p = tmp_path / "tag_groups.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    df = load_tag_groups(p)
    assert list(df.columns) == ["tag_id", "tag", "group", "tag_type", "definition"]
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_grouping.py -v`
Expected: FAIL（ImportError）

- [ ] **Step 3: 实现 grouping.py**

`genometag/tags/grouping.py`：

```python
"""Tag 语义分组：1,128 tag → ~20 组 + content/preference 类型 + 一句话定义。

分组是一次性离线产物（tag_groups.json，入 git）。跨批一致性：
后续批次的 prompt 携带已出现的组名，引导模型复用而非发明新组。
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

VALID_TAG_TYPES = ("content", "preference")

_GROUPING_SYSTEM = "你是电影标签体系专家，负责为 MovieLens genome tag 建立语义分组。"


def build_grouping_messages(tag_batch: list[str], known_groups: list[str]) -> list[dict]:
    hint = ""
    if known_groups:
        hint = "已有组名（优先复用，不要新建语义重复的组）：" + "、".join(known_groups) + "\n"
    user = (
        hint
        + "将以下电影标签按语义分组（目标约 20 组，如：视觉风格、情绪氛围、叙事主题、"
        "角色特征、制作要素、受众偏好……），并为每个标签标注类型"
        "（content = 描述电影本身内容特征；preference = 表达观众偏好/评价）"
        "与一句话定义。\n\n标签：\n"
        + "\n".join(f"- {t}" for t in tag_batch)
        + '\n\n只输出 JSON：{"标签名": {"group": "组名", "tag_type": "content|preference", '
          '"definition": "一句话定义"}, ...}'
    )
    return [{"role": "system", "content": _GROUPING_SYSTEM},
            {"role": "user", "content": user}]


def parse_grouping_response(text: str, expected: list[str]) -> dict[str, dict] | None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    out: dict[str, dict] = {}
    for tag in expected:
        entry = data.get(tag)
        if not isinstance(entry, dict):
            return None
        group, ttype, definition = entry.get("group"), entry.get("tag_type"), entry.get("definition")
        if not isinstance(group, str) or not group or ttype not in VALID_TAG_TYPES \
                or not isinstance(definition, str) or not definition:
            return None
        out[tag] = {"group": group, "tag_type": ttype, "definition": definition}
    return out


def validate_coverage(parsed: dict[str, dict], expected: list[str]) -> list[str]:
    return [t for t in expected if t not in parsed]


def group_tags_via_llm(tags: list[str], client, cache, model: str,
                       batch_size: int = 150) -> pd.DataFrame:
    """分批调用 LLM 分组；每批失败重试一次，仍失败抛 RuntimeError（不静默丢 tag）。"""
    from genometag.annotate.cache import cache_key

    rows: list[dict] = []
    known_groups: list[str] = []
    for start in range(0, len(tags), batch_size):
        batch = tags[start:start + batch_size]
        messages = build_grouping_messages(batch, known_groups)
        parsed = None
        for attempt in range(2):
            run_id = "grouping" if attempt == 0 else "grouping#r1"
            key = cache_key(messages, model, {}, run_id)
            raw = cache.get(key) if cache else None
            if raw is None:
                resp = client.complete(messages, run_id=run_id)
                raw = resp.raw
                if cache:
                    cache.put(key, raw)
                content = resp.content
            else:
                content = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
            parsed = parse_grouping_response(content, batch)
            if parsed is not None:
                break
        if parsed is None:
            raise RuntimeError(f"分组批次失败（tags 起始: {batch[0]}，重试后仍无法解析）")
        missing = validate_coverage(parsed, batch)
        if missing:
            raise RuntimeError(f"分组批次缺失标签: {missing[:5]}")
        for tag, entry in parsed.items():
            rows.append({"tag": tag, **entry})
            if entry["group"] not in known_groups:
                known_groups.append(entry["group"])
    return pd.DataFrame(rows)


def load_tag_groups(path: Path) -> pd.DataFrame:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return pd.DataFrame(data)[["tag_id", "tag", "group", "tag_type", "definition"]]
```

- [ ] **Step 4: 写脚本 03**

`scripts/03_group_tags.py`：

```python
"""03 · 生成 tag_groups.json（幂等：已存在则跳过；synthetic 源直接用已知分组）。"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from genometag.config import load_config, paths_for


def main(config_path: str = "configs/default.yaml") -> int:
    config = load_config(config_path)
    paths = paths_for(config)
    if paths.tag_groups.exists():
        print(f"[03] 已存在，跳过: {paths.tag_groups}（删除可重新生成）")
        return 0

    genome_tags = pd.read_parquet(paths.genome_tags)
    if config.data.source == "synthetic":
        from genometag.data.sample import SYNTHETIC_TAGS
        rows = SYNTHETIC_TAGS
    else:
        from genometag.annotate.cache import ResponseCache
        from genometag.annotate.client import make_client
        from genometag.tags.grouping import group_tags_via_llm
        df = group_tags_via_llm(
            tags=genome_tags["tag"].tolist(),
            client=make_client(config),
            cache=ResponseCache(paths.cache),
            model=config.model.model)
        id_map = dict(zip(genome_tags["tag"], genome_tags["tag_id"]))
        rows = [{"tag_id": id_map[r["tag"]], **r}
                for _, r in df.iterrows()]
    paths.processed.mkdir(parents=True, exist_ok=True)
    paths.tag_groups.write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                encoding="utf-8")
    print(f"[03] tag 分组完成: {len(rows)} 个 tag, "
          f"{len({r['group'] for r in rows})} 个组 → {paths.tag_groups}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    raise SystemExit(main(parser.parse_args().config))
```

- [ ] **Step 5: 运行测试**

Run: `.venv/Scripts/python -m pytest tests/test_grouping.py -v`
Expected: 6 passed

Run: `.venv/Scripts/python scripts/03_group_tags.py --config configs/smoke.yaml`
Expected: 打印 24 个 tag（合成源已由 00 生成过，此处应打印"已存在，跳过"——先删掉再跑验证重建：`rm data/synthetic/processed/tag_groups.json` 后重跑应打印 24 tag 4 组）

- [ ] **Step 6: Commit**

```bash
git add genometag/tags/grouping.py scripts/03_group_tags.py tests/test_grouping.py
git commit -m "feat: tag 语义分组（跨批组名复用、覆盖校验、幂等脚本）"
```

---

### Task 12: 标注编排管道

**Files:**
- Create: `genometag/annotate/pipeline.py`
- Create: `scripts/04_run_annotation.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `build_messages`/`parse_scores`/`FewShot`/`TagDef`（Task 10）、`cache_key`/`ResponseCache`（Task 7）、`make_client`（Task 8）、`load_tag_groups`（Task 11）、`AppConfig`（Task 2）、`build_dataset`（Task 6，仅测试 fixture 用）
- Produces:
  - `build_task_matrix(movie_ids: list[int], tag_groups: pd.DataFrame, n_runs: int) -> list[TaskSpec]`；`@dataclass(frozen=True) TaskSpec: movie_id: int, group: str, run_id: str`
  - `select_few_shots(dataset, genome_scores, n: int, split_year: int, top_k: int, tag_names: dict[int, str]) -> list[FewShot]`（**只取 `year < split_year`** 的电影；按 top_k 分数和最高的 n 部；`scores` 键为 tag 文本）
  - `run_annotation(config: AppConfig, paths, client, cache: ResponseCache | None, dry_run: bool = False) -> Path | None`
    - 读 `paths.dataset`/`paths.genome_scores`/`paths.tag_groups`
    - `config.data.movie_limit` 生效：`random.Random(seed).sample(sorted(ids), limit)`
    - 逐任务：可用模态为空 → `no_content`；否则最多 3 次解析尝试（`run_id` 不变，解析重试用 `run_id#r1`/`#r2` 变键），仍失败记 null
    - 断点续跑：append-only `annotations.jsonl`（每行 `{task_key, status, rows, usage}`），重跑跳过已有行；结束时压缩为 `annotations.parquet`（列 `movie_id, tag_id, group, modality_set, run_id, score`）
    - 写 `run_manifest.json`（各状态计数）与 `cost_report.json`
    - `dry_run=True`：打印前 3 个任务的 prompt 文本（不调用客户端、不写任何文件），返回 None——spec §4.5 的 `--dry-run`
  - 正常返回值：`annotations.parquet` 路径

- [ ] **Step 1: 写失败测试**

`tests/test_pipeline.py`：

```python
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
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_pipeline.py -v`
Expected: FAIL（ImportError）

- [ ] **Step 3: 实现 pipeline.py**

`genometag/annotate/pipeline.py`：

```python
"""标注编排：电影 × tag组 × 轮次 任务矩阵，断点续跑，解析重试降级。

annotations.jsonl 是续跑的事实来源（append-only，每任务一行）；
结束时压缩为 annotations.parquet 供评估使用。
mock 模式不读不写缓存（确定性 + 不污染真实缓存）。
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from genometag.annotate.cache import ResponseCache, cache_key
from genometag.annotate.prompts import (
    FewShot, TagDef, build_messages, parse_scores,
)
from genometag.config import AppConfig, paths_for
from genometag.data.movienet import MovieMultimodal
from genometag.tags.grouping import load_tag_groups


@dataclass(frozen=True)
class TaskSpec:
    movie_id: int
    group: str
    run_id: str


def build_task_matrix(movie_ids: list[int], tag_groups: pd.DataFrame,
                      n_runs: int) -> list[TaskSpec]:
    groups = sorted(tag_groups["group"].unique())
    return [TaskSpec(m, g, f"run-{r}")
            for m in movie_ids for g in groups for r in range(n_runs)]


def select_few_shots(dataset: pd.DataFrame, genome_scores: pd.DataFrame,
                     n: int, split_year: int, top_k: int,
                     tag_names: dict[int, str]) -> list[FewShot]:
    """从 year < split_year 的电影中选 top-k 分数和最高的 n 部（防泄露约束）。

    tag_names: tag_id → tag 文本（来自 tag_groups；few-shot 展示 tag 名而非 id）。
    """
    if n <= 0:
        return []
    train = dataset[dataset["year"] < split_year]
    scores = genome_scores[genome_scores["movie_id"].isin(set(train["movie_id"]))]
    if scores.empty:
        return []
    top_sums = (scores.groupby("movie_id")["relevance"]
                .apply(lambda s: s.nlargest(top_k).sum()))
    best = top_sums.sort_values(ascending=False).head(n).index.tolist()
    title_map = dict(zip(dataset["movie_id"], dataset["title"]))
    year_map = dict(zip(dataset["movie_id"], dataset["year"]))
    shots = []
    for movie_id in best:
        rows = genome_scores[genome_scores["movie_id"] == movie_id].nlargest(top_k, "relevance")
        shots.append(FewShot(
            title=title_map[movie_id], year=int(year_map[movie_id]),
            scores={tag_names[int(t)]: round(float(r), 2)
                    for t, r in zip(rows["tag_id"], rows["relevance"])}))
    return shots
```

`run_annotation`：

```python
def run_annotation(config: AppConfig, paths, client, cache: ResponseCache | None,
                   dry_run: bool = False) -> Path | None:
    dataset = pd.read_parquet(paths.dataset)
    genome_scores = pd.read_parquet(paths.genome_scores)
    tag_groups = load_tag_groups(paths.tag_groups)

    movie_ids = sorted(dataset["movie_id"].tolist())
    if config.data.movie_limit is not None and config.data.movie_limit < len(movie_ids):
        movie_ids = random.Random(config.data.seed).sample(movie_ids, config.data.movie_limit)

    tag_names = dict(zip(tag_groups["tag_id"], tag_groups["tag"]))
    few_shots = select_few_shots(
        dataset, genome_scores, config.annotation.few_shot_examples,
        config.protocol.temporal_split_year, config.protocol.top_k, tag_names)

    tasks = build_task_matrix(movie_ids, tag_groups, config.annotation.n_runs)
    modality_set = "|".join(config.annotation.modalities)
    groups_df = {g: df for g, df in tag_groups.groupby("group")}

    if dry_run:
        # spec §4.5 的 --dry-run：打印前 3 个任务的 prompt 文本，不调用、不写文件
        for task in tasks[:3]:
            movie_row = dataset[dataset["movie_id"] == task.movie_id].iloc[0]
            mm = _to_multimodal(movie_row, paths.movienet)
            tag_defs = [TagDef(int(r.tag_id), r.tag, r.definition)
                        for r in groups_df[task.group].itertuples()]
            messages = build_messages(
                mm, tag_defs,
                _available_modalities(mm, config.annotation.modalities),
                few_shots, config.model)
            texts = []
            for m in messages:
                c = m["content"]
                if isinstance(c, str):
                    texts.append(c)
                else:
                    texts.extend(p.get("text", "") for p in c if p.get("type") == "text")
            print(f"[dry-run] 任务 {task.movie_id}|{task.group}|{task.run_id}\n"
                  + "\n".join(texts)[:1500] + "\n")
        return None

    paths.results.mkdir(parents=True, exist_ok=True)
    jsonl_path = paths.results / "annotations.jsonl"
    done_keys: set[str] = set()
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_keys.add(json.loads(line)["task_key"])

    cost = {"total_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    counts = {"done": 0, "parse_failed": 0, "no_content": 0}

    with jsonl_path.open("a", encoding="utf-8") as sink:
        for task in tasks:
            task_key = f"{task.movie_id}|{task.group}|{task.run_id}"
            if task_key in done_keys:
                continue
            movie_row = dataset[dataset["movie_id"] == task.movie_id].iloc[0]
            mm = _to_multimodal(movie_row, paths.movienet)
            group_tags = groups_df[task.group]
            available = _available_modalities(mm, config.annotation.modalities)
            if not available:
                counts["no_content"] += 1
                sink.write(json.dumps({"task_key": task_key, "status": "no_content",
                                       "rows": [], "usage": {}}) + "\n")
                continue

            tag_defs = [TagDef(int(r.tag_id), r.tag, r.definition)
                        for r in group_tags.itertuples()]
            messages = build_messages(mm, tag_defs, available, few_shots, config.model)
            expected = [t.tag for t in tag_defs]

            scores: dict[str, float] | None = None
            usage: dict = {}
            for attempt in range(3):
                eff_run = task.run_id if attempt == 0 else f"{task.run_id}#r{attempt}"
                raw = None
                if cache is not None and not config.mock:
                    key = cache_key(messages, config.model.model,
                                    {"temperature": config.model.temperature}, eff_run)
                    raw = cache.get(key)
                if raw is None:
                    resp = client.complete(messages, run_id=eff_run)
                    raw = resp.raw
                    usage = resp.usage
                    cost["total_calls"] += 1
                    cost["prompt_tokens"] += usage.get("prompt_tokens", 0)
                    cost["completion_tokens"] += usage.get("completion_tokens", 0)
                    if cache is not None and not config.mock:
                        cache.put(key, raw)
                content = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
                scores = parse_scores(content, expected)
                if scores is not None:
                    break

            status = "done" if scores is not None else "parse_failed"
            counts[status] += 1
            rows = [{"movie_id": task.movie_id, "tag_id": int(td.tag_id),
                     "group": task.group, "modality_set": modality_set,
                     "run_id": task.run_id,
                     "score": scores.get(td.tag) if scores else None}
                    for td in tag_defs]
            sink.write(json.dumps({"task_key": task_key, "status": status,
                                   "rows": rows, "usage": usage}, ensure_ascii=False) + "\n")

    # 压缩 jsonl → parquet（含历史行，防丢）
    all_rows: list[dict] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            all_rows.extend(json.loads(line)["rows"])
    parquet_path = paths.results / "annotations.parquet"
    pd.DataFrame(all_rows,
                 columns=["movie_id", "tag_id", "group", "modality_set", "run_id", "score"]
                 ).to_parquet(parquet_path, index=False)

    (paths.results / "run_manifest.json").write_text(
        json.dumps({"tasks": counts, **cost}, ensure_ascii=False, indent=1), encoding="utf-8")
    (paths.results / "cost_report.json").write_text(
        json.dumps(cost, ensure_ascii=False, indent=1), encoding="utf-8")
    return parquet_path


def _available_modalities(mm: MovieMultimodal, wanted: list[str]) -> list[str]:
    avail = []
    if "video" in wanted and mm.keyframes:
        avail.append("video")
    if "script" in wanted and mm.script:
        avail.append("script")
    if "dialogue" in wanted and mm.subtitles:
        avail.append("dialogue")
    return avail


def _to_multimodal(row, movienet_root: Path) -> MovieMultimodal:
    frames = [movienet_root / k for k in row["keyframes"]]
    script = None
    if row["script_path"]:
        p = movienet_root / row["script_path"]
        script = p.read_text(encoding="utf-8", errors="replace") if p.exists() else None
    subtitles = None
    if row["subtitles_path"]:
        p = movienet_root / row["subtitles_path"]
        subtitles = p.read_text(encoding="utf-8", errors="replace") if p.exists() else None
    return MovieMultimodal(str(row["movienet_id"]), str(row["title"]),
                           int(row["year"]) if pd.notna(row["year"]) else None,
                           None, frames, script, subtitles)
```

（注记：`run_manifest.json` 与 `cost_report.json` 的计数只统计本次会话——续跑场景的总量以 jsonl 汇总为准。）

- [ ] **Step 4: 写脚本 04**

`scripts/04_run_annotation.py`：

```python
"""04 · MLLM 标注主跑（mock 配置零成本；--dry-run 打印 prompt 不调用）。"""
from __future__ import annotations

import argparse

from genometag.annotate.cache import ResponseCache
from genometag.annotate.client import make_client
from genometag.annotate.pipeline import run_annotation
from genometag.config import load_config, paths_for


def main(config_path: str = "configs/default.yaml", dry_run: bool = False) -> int:
    config = load_config(config_path)
    paths = paths_for(config)
    client = make_client(config)
    cache = None if config.mock else ResponseCache(paths.cache)
    out = run_annotation(config, paths, client, cache, dry_run=dry_run)
    if out is None:
        print("[04] dry-run 完成（未调用、未写文件）")
    else:
        print(f"[04] 标注完成 → {out}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dry-run", action="store_true", help="打印 prompt，不调用不写文件")
    args = parser.parse_args()
    raise SystemExit(main(args.config, args.dry_run))
```

- [ ] **Step 5: 运行测试**

Run: `.venv/Scripts/python -m pytest tests/test_pipeline.py -v`
Expected: 7 passed

- [ ] **Step 6: Commit**

```bash
git add genometag/annotate/pipeline.py scripts/04_run_annotation.py tests/test_pipeline.py
git commit -m "feat: 标注编排（任务矩阵、jsonl 断点续跑、解析重试降级、no_content 跳过、dry-run）"
```

---

### Task 13: 评估指标

**Files:**
- Create: `genometag/eval/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: numpy/scipy
- Produces（全部纯函数，输入 `np.ndarray`）：
  - `rmse(pred, truth) -> float`、`mae(pred, truth) -> float`
  - `precision_at_k(pred: np.ndarray, truth: np.ndarray, k: int, threshold: float) -> float`（单电影；truth ≥ threshold 为相关；pred 降序取 top-k 的命中率）
  - `ndcg_at_k(pred, truth, k, threshold) -> float`（二值相关 NDCG）
  - `consistency_std(scores: np.ndarray) -> np.ndarray`（形状 `(n_items, n_runs)` → 每行 std，`ddof=0`）
  - `icc_2_1(scores: np.ndarray) -> float`（双向随机效应、单次测量）
  - `paired_bootstrap(a: np.ndarray, b: np.ndarray, n_resamples: int = 10000, seed: int = 42) -> tuple[float, float]`（逐单元指标差的均值与双侧 p 值）

- [ ] **Step 1: 写失败测试**

`tests/test_metrics.py`：

```python
"""指标：手算值锁定。"""
import numpy as np
import pytest

from genometag.eval.metrics import (
    consistency_std, icc_2_1, mae, ndcg_at_k, paired_bootstrap, precision_at_k, rmse,
)


def test_rmse_mae():
    pred = np.array([1.0, 0.0, 1.0])
    truth = np.array([0.0, 1.0, 1.0])
    assert rmse(pred, truth) == pytest.approx(np.sqrt(2 / 3))
    assert mae(pred, truth) == pytest.approx(2 / 3)


def test_precision_at_k():
    pred = np.array([0.9, 0.8, 0.7, 0.6])
    truth = np.array([0.9, 0.1, 0.8, 0.2])
    # 相关集(threshold=0.5) = {0,2}; top-2 pred = {0,1} → 命中 {0}
    assert precision_at_k(pred, truth, k=2, threshold=0.5) == pytest.approx(0.5)
    assert precision_at_k(pred, truth, k=4, threshold=0.5) == pytest.approx(0.5)


def test_ndcg_at_k():
    pred = np.array([0.9, 0.8, 0.7, 0.6])
    truth = np.array([0.9, 0.1, 0.8, 0.2])
    # DCG = 1/log2(2) = 1; IDCG = 1/log2(2) + 1/log2(3) = 1.63093
    assert ndcg_at_k(pred, truth, k=2, threshold=0.5) == pytest.approx(1 / 1.63093, rel=1e-4)


def test_perfect_prediction():
    pred = np.array([0.9, 0.1, 0.8])
    assert precision_at_k(pred, pred, k=2, threshold=0.5) == pytest.approx(1.0)
    assert ndcg_at_k(pred, pred, k=2, threshold=0.5) == pytest.approx(1.0)


def test_consistency_std():
    scores = np.array([[0.5, 0.5, 0.5], [0.2, 0.4, 0.6]])
    stds = consistency_std(scores)
    assert stds[0] == pytest.approx(0.0)
    assert stds[1] == pytest.approx(np.std([0.2, 0.4, 0.6]))


def test_icc_2_1():
    """4 受试 × 2 评分者的手算例：MSR=7.5, MSE=0.8333, MSC=12.5 → ICC≈0.4706。"""
    scores = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]])
    assert icc_2_1(scores) == pytest.approx(0.470588, rel=1e-4)


def test_icc_perfect_agreement():
    scores = np.array([[0.2, 0.2, 0.2], [0.8, 0.8, 0.8], [0.5, 0.5, 0.5]])
    assert icc_2_1(scores) == pytest.approx(1.0, abs=1e-6)


def test_paired_bootstrap():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([2.0, 3.0, 4.0, 5.0])
    diff, p = paired_bootstrap(a, b, n_resamples=10000, seed=42)
    assert diff == pytest.approx(-1.0)
    assert p < 0.05                      # b 稳定优于 a
    diff0, p0 = paired_bootstrap(a, a, n_resamples=1000, seed=42)
    assert diff0 == pytest.approx(0.0) and p0 > 0.9
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_metrics.py -v`
Expected: FAIL（ImportError）

- [ ] **Step 3: 实现 metrics.py**

`genometag/eval/metrics.py`：

```python
"""评估指标：纯函数、numpy 输入、手算值锁定（tests/test_metrics.py）。

排序类指标约定：truth ≥ threshold 为相关集；按 pred 降序取 top-k。
ICC(2,1)：双向随机效应、单次测量。
"""
from __future__ import annotations

import numpy as np


def rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(truth)) ** 2)))


def mae(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(truth))))


def precision_at_k(pred: np.ndarray, truth: np.ndarray, k: int, threshold: float) -> float:
    pred, truth = np.asarray(pred), np.asarray(truth)
    top = np.argsort(-pred)[:k]
    relevant = truth >= threshold
    hits = relevant[top].sum()
    return float(hits / k)


def ndcg_at_k(pred: np.ndarray, truth: np.ndarray, k: int, threshold: float) -> float:
    pred, truth = np.asarray(pred), np.asarray(truth)
    relevant = truth >= threshold
    top = np.argsort(-pred)[:k]
    gains = relevant[top].astype(float)
    dcg = float(np.sum(gains / np.log2(np.arange(2, len(gains) + 2))))
    n_rel = int(relevant.sum())
    ideal = np.ones(min(k, n_rel))
    idcg = float(np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2))))
    return dcg / idcg if idcg > 0 else 0.0


def consistency_std(scores: np.ndarray) -> np.ndarray:
    """(n_items, n_runs) → 每行的多次调用标准差。"""
    return np.asarray(scores).std(axis=1)


def icc_2_1(scores: np.ndarray) -> float:
    """ICC(2,1)：n=受试（行），k=评分者（列）。

    ICC = (MSR - MSE) / (MSR + (k-1)·MSE + k·(MSC - MSE)/n)
    """
    x = np.asarray(scores, dtype=float)
    n, k = x.shape
    grand = x.mean()
    msr = k * np.sum((x.mean(axis=1) - grand) ** 2) / (n - 1)
    msc = n * np.sum((x.mean(axis=0) - grand) ** 2) / (k - 1)
    sse = np.sum((x - grand) ** 2) - k * np.sum((x.mean(axis=1) - grand) ** 2) \
        - n * np.sum((x.mean(axis=0) - grand) ** 2)
    mse = sse / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / denom) if denom != 0 else 0.0


def paired_bootstrap(a: np.ndarray, b: np.ndarray,
                     n_resamples: int = 10000, seed: int = 42) -> tuple[float, float]:
    """逐单元配对 bootstrap：返回 (mean(a-b), 双侧 p 值)。"""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diffs = a - b
    rng = np.random.default_rng(seed)
    n = len(diffs)
    samples = rng.choice(diffs, size=(n_resamples, n), replace=True).mean(axis=1)
    observed = float(diffs.mean())
    p = 2 * min(float((samples <= 0).mean()), float((samples >= 0).mean()))
    return observed, min(1.0, p)
```

- [ ] **Step 4: 运行测试**

Run: `.venv/Scripts/python -m pytest tests/test_metrics.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add genometag/eval/metrics.py tests/test_metrics.py
git commit -m "feat: 评估指标（RMSE/MAE/P@K/NDCG@K/一致性 std/ICC(2,1)/配对 bootstrap，手算锁定）"
```

---

### Task 14: 评估协议（子集选择与防泄露）

**Files:**
- Create: `genometag/eval/protocol.py`
- Test: `tests/test_protocol.py`

**Interfaces:**
- Consumes: 无（pandas）
- Produces:
  - `temporal_split(dataset: pd.DataFrame, year: int) -> tuple[pd.Index, pd.Index]`（train = `year < year`，test = `year >= year`，按 `movie_id` 列返回索引）
  - `cold_start_subset(dataset: pd.DataFrame, user_tags: pd.DataFrame, max_tags: int) -> pd.Index`（用户标签数 ≤ max_tags 的电影）
  - `tag_type_breakdown(tag_groups: pd.DataFrame) -> dict[str, list[int]]`
  - `assert_no_leakage(fit_movie_ids, test_movie_ids) -> None`（交集非空抛 `ValueError`）

- [ ] **Step 1: 写失败测试**

`tests/test_protocol.py`：

```python
"""评估协议：时序分割、冷启动、tag 类型分解、防泄露断言。"""
import pandas as pd
import pytest

from genometag.eval.protocol import (
    assert_no_leakage, cold_start_subset, tag_type_breakdown, temporal_split,
)


def _dataset():
    return pd.DataFrame({"movie_id": [1, 2, 3, 4], "year": [2018, 2021, 2022, 2023]})


def test_temporal_split():
    train, test = temporal_split(_dataset(), 2022)
    assert set(train) == {1, 2}       # 严格小于 2022
    assert set(test) == {3, 4}        # >= 2022


def test_cold_start_subset():
    user_tags = pd.DataFrame({"movie_id": [1, 1, 2], "tag": ["a", "b", "c"]})
    cold = cold_start_subset(_dataset(), user_tags, max_tags=0)
    assert set(cold) == {3, 4}         # 1 有 2 条、2 有 1 条（均 >0，非冷启动）；3、4 无标签


def test_tag_type_breakdown():
    tg = pd.DataFrame({"tag_id": [1, 2, 3], "tag_type": ["content", "content", "preference"]})
    breakdown = tag_type_breakdown(tg)
    assert breakdown == {"content": [1, 2], "preference": [3]}


def test_assert_no_leakage():
    """Review Focus #4：拟合集与测试集重叠必须抛错。"""
    assert_no_leakage({1, 2}, {3, 4})            # 无重叠 → 通过
    with pytest.raises(ValueError, match="泄露|重叠"):
        assert_no_leakage({1, 2}, {2, 3})
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_protocol.py -v`
Expected: FAIL（ImportError）

- [ ] **Step 3: 实现 protocol.py**

`genometag/eval/protocol.py`：

```python
"""评估协议：时序分割、冷启动子集、tag 类型分解、防泄露断言。

防泄露是全框架约束：所有拟合组件（mean/genre/BERT 基线、few-shot）
必须以 assert_no_leakage 收尾自检。
"""
from __future__ import annotations

import pandas as pd


def temporal_split(dataset: pd.DataFrame, year: int) -> tuple[pd.Index, pd.Index]:
    train = dataset.loc[dataset["year"] < year, "movie_id"]
    test = dataset.loc[dataset["year"] >= year, "movie_id"]
    return pd.Index(train), pd.Index(test)


def cold_start_subset(dataset: pd.DataFrame, user_tags: pd.DataFrame,
                      max_tags: int) -> pd.Index:
    counts = user_tags.groupby("movie_id").size()
    n = dataset["movie_id"].map(lambda m: counts.get(m, 0))
    return pd.Index(dataset.loc[n <= max_tags, "movie_id"])


def tag_type_breakdown(tag_groups: pd.DataFrame) -> dict[str, list[int]]:
    return {t: g["tag_id"].tolist() for t, g in tag_groups.groupby("tag_type")}


def assert_no_leakage(fit_movie_ids, test_movie_ids) -> None:
    leak = set(fit_movie_ids) & set(test_movie_ids)
    if leak:
        raise ValueError(
            f"训练参考集与测试集重叠（数据泄露）: {sorted(leak)[:5]}... "
            "拟合组件只允许使用 temporal_split_year 之前的数据")
```

- [ ] **Step 4: 运行测试**

Run: `.venv/Scripts/python -m pytest tests/test_protocol.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add genometag/eval/protocol.py tests/test_protocol.py
git commit -m "feat: 评估协议（时序分割/冷启动/tag类型/防泄露断言）"
```

---

### Task 15: 基线（random / mean / genre）

**Files:**
- Create: `genometag/baselines/__init__.py` 已存在；Create: `genometag/baselines/random_baseline.py`、`genometag/baselines/mean_baseline.py`、`genometag/baselines/genre_baseline.py`
- Create: `scripts/05_run_baselines.py`
- Test: `tests/test_baselines.py`

**Interfaces:**
- Consumes: `temporal_split`/`assert_no_leakage`（Task 14）、`load_user_tags`（Task 4）
- Produces（输出统一长表 schema：`movie_id, tag_id, score, method`）：
  - `random_baseline(movie_ids: list[int], tag_ids: list[int], seed: int) -> pd.DataFrame`
  - `mean_baseline(train_scores: pd.DataFrame, movie_ids: list[int]) -> pd.DataFrame`（每 tag 在 train_scores 上的均值；train_scores 必须只含 `year < split_year` 电影——由调用方保证并用 `assert_no_leakage` 自检）
  - `genre_baseline(train_scores, dataset, movie_ids) -> pd.DataFrame`（每 genre 的 tag 均值向量；电影预测 = 其 genres 向量均值；无 genre → 全局均值）
  - `run_baselines(config, paths) -> Path`（05 调用；写 `results/baselines/baselines.parquet` 与 `fit_report.json`）

- [ ] **Step 1: 写失败测试**

`tests/test_baselines.py`：

```python
"""基线：确定性、防泄露、手算值。"""
import pandas as pd
import pytest

from genometag.baselines.genre_baseline import genre_baseline
from genometag.baselines.mean_baseline import mean_baseline
from genometag.baselines.random_baseline import random_baseline


def test_random_deterministic_and_bounded():
    df1 = random_baseline([1, 2], [10, 11], seed=7)
    df2 = random_baseline([1, 2], [10, 11], seed=7)
    pd.testing.assert_frame_equal(df1, df2)
    assert len(df1) == 4                     # 2 电影 × 2 tag
    assert df1["score"].between(0, 1).all()
    assert list(df1.columns) == ["movie_id", "tag_id", "score", "method"]


def test_mean_baseline_hand_computed():
    train = pd.DataFrame({"movie_id": [1, 2, 1, 2], "tag_id": [10, 10, 11, 11],
                          "relevance": [0.8, 0.4, 0.2, 0.2]})
    df = mean_baseline(train, movie_ids=[5])
    # tag 10 → 0.6；tag 11 → 0.2
    got = dict(zip(df["tag_id"], df["score"]))
    assert got[10] == pytest.approx(0.6)
    assert got[11] == pytest.approx(0.2)
    assert (df["method"] == "mean").all()


def test_mean_baseline_leakage_guard():
    """拟合集含测试电影 → 抛错（防泄露自检）。"""
    train = pd.DataFrame({"movie_id": [1, 2], "tag_id": [10, 10], "relevance": [0.8, 0.4]})
    with pytest.raises(ValueError, match="泄露|重叠"):
        mean_baseline(train, movie_ids=[1], test_ids={1})


def test_genre_baseline_hand_computed():
    dataset = pd.DataFrame({"movie_id": [1, 2, 3, 4, 5, 6],
                            "genres": [["A"], ["A", "B"], ["B"], ["A"], ["A", "B"], []]})
    train = pd.DataFrame({  # 电影 1-3 为训练参考集
        "movie_id": [1, 2, 3], "tag_id": [10, 10, 10], "relevance": [0.8, 0.6, 0.2]})
    df = genre_baseline(train, dataset, movie_ids=[4, 5, 6])
    got = dict(zip(df["movie_id"], df["score"]))
    # v_A = mean(0.8, 0.6) = 0.7; v_B = mean(0.6, 0.2) = 0.4; 全局 = 1.6/3
    assert got[4] == pytest.approx(0.7)                 # [A]
    assert got[5] == pytest.approx(0.55)                # [A, B] → (0.7+0.4)/2
    assert got[6] == pytest.approx(1.6 / 3)             # 无 genre → 全局均值
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_baselines.py -v`
Expected: FAIL（ImportError）

- [ ] **Step 3: 实现三个基线**

`genometag/baselines/random_baseline.py`：

```python
"""随机基线：固定种子均匀采样 [0,1)。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def random_baseline(movie_ids: list[int], tag_ids: list[int], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [{"movie_id": m, "tag_id": t, "score": float(rng.uniform()), "method": "random"}
            for m in movie_ids for t in tag_ids]
    return pd.DataFrame(rows, columns=["movie_id", "tag_id", "score", "method"])
```

`genometag/baselines/mean_baseline.py`：

```python
"""全局均值基线：每 tag 在训练参考集（分割年前）上的 relevance 均值。"""
from __future__ import annotations

import pandas as pd

from genometag.eval.protocol import assert_no_leakage


def mean_baseline(train_scores: pd.DataFrame, movie_ids: list[int],
                  test_ids: set[int] | None = None) -> pd.DataFrame:
    if test_ids is not None:  # 防泄露自检：拟合数据不得含测试电影
        assert_no_leakage(set(train_scores["movie_id"]), test_ids)
    per_tag = train_scores.groupby("tag_id")["relevance"].mean()
    rows = [{"movie_id": m, "tag_id": t, "score": float(s), "method": "mean"}
            for m in movie_ids for t, s in per_tag.items()]
    return pd.DataFrame(rows, columns=["movie_id", "tag_id", "score", "method"])
```

`genometag/baselines/genre_baseline.py`：

```python
"""genre-only 基线：genre → tag 均值映射（训练参考集统计），未见 genre 组合回退全局均值。"""
from __future__ import annotations

import pandas as pd

from genometag.eval.protocol import assert_no_leakage


def genre_baseline(train_scores: pd.DataFrame, dataset: pd.DataFrame,
                   movie_ids: list[int], test_ids: set[int] | None = None) -> pd.DataFrame:
    if test_ids is not None:
        assert_no_leakage(set(train_scores["movie_id"]), test_ids)
    train_meta = dataset[dataset["movie_id"].isin(set(train_scores["movie_id"]))]
    merged = train_scores.merge(train_meta[["movie_id", "genres"]], on="movie_id")
    # 每 genre 的 tag 均值
    genre_rows = []
    for _, r in merged.iterrows():
        for g in r["genres"]:
            genre_rows.append({"genre": g, "tag_id": r["tag_id"], "relevance": r["relevance"]})
    per_genre = pd.DataFrame(genre_rows).groupby(["genre", "tag_id"])["relevance"].mean()
    global_mean = train_scores.groupby("tag_id")["relevance"].mean()
    all_tags = list(global_mean.index)

    meta = dataset.set_index("movie_id")
    rows = []
    for m in movie_ids:
        genres = meta.loc[m, "genres"] if m in meta.index else []
        for t in all_tags:
            vals = [per_genre.get((g, t)) for g in genres]
            vals = [v for v in vals if v is not None]
            score = float(sum(vals) / len(vals)) if vals else float(global_mean[t])
            rows.append({"movie_id": m, "tag_id": t, "score": score, "method": "genre"})
    return pd.DataFrame(rows, columns=["movie_id", "tag_id", "score", "method"])
```

- [ ] **Step 4: 写脚本 05（random/mean/genre；bert 挂钩 Task 16）**

`scripts/05_run_baselines.py`：

```python
"""05 · 运行基线。mean/genre 只在时序训练参考集上拟合（防泄露）。"""
from __future__ import annotations

import argparse
import importlib
import json

import pandas as pd

from genometag.config import load_config, paths_for


def run_baselines(config, paths) -> tuple[pd.DataFrame, dict]:
    dataset = pd.read_parquet(paths.dataset)
    scores = pd.read_parquet(paths.genome_scores)
    from genometag.eval.protocol import temporal_split
    train_ids, test_ids = temporal_split(dataset, config.protocol.temporal_split_year)
    train_scores = scores[scores["movie_id"].isin(set(train_ids))]
    tag_ids = sorted(scores["tag_id"].unique())
    movie_ids = sorted(dataset["movie_id"].tolist())
    test_set = set(test_ids)

    frames = []
    report = {"train_movies": len(train_ids), "test_movies": len(test_ids)}
    for name in config.baselines.run:
        if name == "random":
            from genometag.baselines.random_baseline import random_baseline
            frames.append(random_baseline(movie_ids, tag_ids, config.data.seed))
        elif name == "mean":
            from genometag.baselines.mean_baseline import mean_baseline
            frames.append(mean_baseline(train_scores, movie_ids, test_ids=test_set))
        elif name == "genre":
            from genometag.baselines.genre_baseline import genre_baseline
            frames.append(genre_baseline(train_scores, dataset, movie_ids, test_ids=test_set))
        elif name == "bert":
            try:
                mod = importlib.import_module("genometag.baselines.bert_baseline")
            except ImportError:
                print("[05] torch/transformers 未安装，跳过 BERT 基线"
                      "（pip install -r requirements-baselines.txt）")
                report["bert"] = "skipped"
                continue
            frames.append(mod.bert_baseline(config, paths, dataset, scores,
                                            train_ids, test_ids))
        else:
            raise ValueError(f"未知基线: {name}")
        report[name] = "ok"
    out = pd.concat(frames, ignore_index=True)
    return out, report


def main(config_path: str = "configs/default.yaml") -> int:
    config = load_config(config_path)
    paths = paths_for(config)
    df, report = run_baselines(config, paths)
    out_dir = paths.root / "results" / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "baselines.parquet", index=False)
    (out_dir / "fit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[05] 基线完成: {len(df)} 行 → {out_dir / 'baselines.parquet'}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    raise SystemExit(main(parser.parse_args().config))
```

（`bert_baseline` 的函数签名 `bert_baseline(config, paths, dataset, scores, train_ids, test_ids) -> pd.DataFrame` 在 Task 16 实现；本任务先跑通 random/mean/genre，bert 分支在 torch 缺失时打印跳过。）

- [ ] **Step 5: 运行测试**

Run: `.venv/Scripts/python -m pytest tests/test_baselines.py -v`
Expected: 4 passed

Run: `.venv/Scripts/python scripts/05_run_baselines.py --config configs/smoke.yaml`
Expected: 打印 `5 电影 × 24 tag × 3 基线 = 360 行`（bert 未装 → skipped）

- [ ] **Step 6: Commit**

```bash
git add genometag/baselines/ scripts/05_run_baselines.py tests/test_baselines.py
git commit -m "feat: random/mean/genre 基线（防泄露自检）+ 05 脚本"
```

---

### Task 16: BERT 基线（可选依赖）

**Files:**
- Create: `genometag/baselines/bert_baseline.py`
- Test: `tests/test_bert_baseline.py`（默认跳过，`RUN_BERT_TEST=1` 显式开启——需联网下模型，是全局"无网络测试"约束的唯一豁免，已获 spec §7 允许的标注方式：测试内显式声明）

**Interfaces:**
- Consumes: Task 15 脚本 05 定义的签名：`bert_baseline(config, paths, dataset, scores, train_ids, test_ids) -> pd.DataFrame`
- Produces: 同基线长表 schema（`method="bert"`）；文本 = 剧本文本（`script` 缺失的电影用字幕文本）

- [ ] **Step 1: 写测试（默认跳过）**

`tests/test_bert_baseline.py`：

```python
"""BERT 基线：需要 torch/transformers + 联网下载小模型。
默认跳过；RUN_BERT_TEST=1 显式运行（全局无网络约束的唯一豁免）。"""
import os

import pandas as pd
import pytest

pytestmark = [
    pytest.mark.skipif(os.environ.get("RUN_BERT_TEST") != "1",
                       reason="需要网络与模型下载，RUN_BERT_TEST=1 开启"),
]


def test_bert_baseline_learns_keyword_signal(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from genometag.baselines.bert_baseline import _train_predict

    # 文本→矩阵→训练→预测的管道验证（微型随机模型 1 epoch 不保证学到
    # 关键词信号，故只做结构性断言；学习效果由真实模型人工验证）
    train_movie_texts = {1: "neon night city neon dreams", 2: "quiet harbor dawn"}
    train_movie_matrix = pd.DataFrame({"movie_id": [1, 2], "tag_id": [1, 1],
                                       "relevance": [0.9, 0.1]})
    test_movie_texts = {3: "all neon everything"}
    pred = _train_predict(train_movie_texts, train_movie_matrix, test_movie_texts,
                          epochs=1, model_name="hf-internal-testing/tiny-random-BertModel")
    import math
    assert 3 in pred and set(pred[3]) == {1}
    assert math.isfinite(pred[3][1])
```

- [ ] **Step 2: 运行确认跳过**

Run: `.venv/Scripts/python -m pytest tests/test_bert_baseline.py -v`
Expected: 1 skipped

- [ ] **Step 3: 实现 bert_baseline.py**

`genometag/baselines/bert_baseline.py`：

```python
"""剧本 + BERT 多标签回归基线（可选依赖 torch/transformers）。

只用时序训练参考集训练（防泄露），在全部电影上输出预测。
懒导入：torch 未安装时 import 本模块即 ImportError，05 脚本捕获后跳过。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _movie_texts(dataset: pd.DataFrame, paths) -> dict[int, str]:
    texts = {}
    for _, r in dataset.iterrows():
        text = None
        if r["script_path"]:
            p = paths.movienet / r["script_path"]
            text = p.read_text(encoding="utf-8", errors="replace") if p.exists() else None
        if not text and r["subtitles_path"]:
            p = paths.movienet / r["subtitles_path"]
            text = p.read_text(encoding="utf-8", errors="replace") if p.exists() else None
        texts[int(r["movie_id"])] = (text or "")[:2000]
    return texts


def _train_predict(train_texts: dict[int, str], train_matrix: pd.DataFrame,
                   test_texts: dict[int, str], epochs: int = 3,
                   model_name: str = "distilbert-base-uncased",
                   batch_size: int = 8, max_length: int = 256,
                   lr: float = 2e-5) -> dict[int, dict[int, float]]:
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer

    tag_ids = sorted(train_matrix["tag_id"].unique())
    tag_col = {t: i for i, t in enumerate(tag_ids)}
    movies = sorted(train_texts)
    mat = np.zeros((len(movies), len(tag_ids)), dtype=np.float32)
    movie_row = {m: i for i, m in enumerate(movies)}
    for _, r in train_matrix.iterrows():
        mat[movie_row[int(r["movie_id"])], tag_col[int(r["tag_id"])]] = r["relevance"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name)
            self.out = nn.Linear(self.encoder.config.hidden_size, len(tag_ids))

        def forward(self, ids, mask):
            return self.out(self.encoder(input_ids=ids, attention_mask=mask)
                            .last_hidden_state[:, 0])

    torch.manual_seed(0)
    model = Head()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    enc = tokenizer([train_texts[m] for m in movies], truncation=True,
                    max_length=max_length, padding=True, return_tensors="pt")
    X = {"ids": enc["input_ids"], "mask": enc["attention_mask"]}
    Y = torch.from_numpy(mat)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X["ids"], X["mask"]), Y)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        tenc = tokenizer([test_texts[m] for m in sorted(test_texts)], truncation=True,
                         max_length=max_length, padding=True, return_tensors="pt")
        preds = model(tenc["input_ids"], tenc["attention_mask"]).numpy()
    return {m: {tag_ids[i]: float(v) for i, v in enumerate(row)}
            for m, row in zip(sorted(test_texts), preds)}


def bert_baseline(config, paths, dataset: pd.DataFrame, scores: pd.DataFrame,
                  train_ids, test_ids) -> pd.DataFrame:
    from genometag.eval.protocol import assert_no_leakage
    assert_no_leakage(set(train_ids), set(test_ids))

    texts = _movie_texts(dataset, paths)
    train_scores = scores[scores["movie_id"].isin(set(train_ids))]
    # 电影级 tag 矩阵（训练参考集）
    train_matrix = train_scores[["movie_id", "tag_id", "relevance"]]
    train_texts = {m: t for m, t in texts.items() if m in set(train_ids)}
    preds = _train_predict(train_texts, train_matrix, texts)
    rows = [{"movie_id": m, "tag_id": t, "score": float(np.clip(v, 0.0, 1.0)),
             "method": "bert"}
            for m, tag_scores in preds.items() for t, v in tag_scores.items()]
    return pd.DataFrame(rows, columns=["movie_id", "tag_id", "score", "method"])
```

- [ ] **Step 4: 验证**

Run: `.venv/Scripts/python -m pytest tests/test_bert_baseline.py -v`
Expected: 1 skipped（默认）

可选（联网、装可选依赖后）：`.venv/Scripts/python -m pip install -r requirements-baselines.txt && RUN_BERT_TEST=1 .venv/Scripts/python -m pytest tests/test_bert_baseline.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add genometag/baselines/bert_baseline.py tests/test_bert_baseline.py
git commit -m "feat: BERT 多标签回归基线（可选依赖、懒导入、防泄露）"
```

---

### Task 17: 报告生成与评估脚本

**Files:**
- Create: `genometag/eval/report.py`
- Create: `scripts/06_evaluate.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `metrics.py` 全部（Task 13）、`protocol.py`（Task 14）、`load_tag_groups`（Task 11）
- Produces:
  - `metrics_table(predictions: dict[str, pd.DataFrame], truth: pd.DataFrame, dataset: pd.DataFrame, cfg: ProtocolConfig, tag_groups: pd.DataFrame | None, movie_ids: list[int] | None = None) -> pd.DataFrame`（行 = `method[, subset]`，列 = `rmse, mae, precision@k, ndcg@k`；`predictions` 为方法名 → `(movie_id, tag_id, score)` 长表；`movie_ids` 限定评估电影集合）
  - `modality_heatmap(rows: pd.DataFrame, out_png: Path) -> None`（rows 列 `modality_set, tag_type, ndcg`；pivot 后 imshow）——**实现本函数前先调用 dataviz 技能**
  - `write_report(exp_dirs: list[Path], baseline_dir: Path | None, out_dir: Path, cfg: ProtocolConfig, dataset: pd.DataFrame, truth: pd.DataFrame, tag_groups: pd.DataFrame | None, user_tags: pd.DataFrame | None) -> Path`（产出 metrics.csv / report.md / consistency.json（轮次 > 1 时）/ heatmap.png（多模态实验时））
  - `06_evaluate.py --config <cfg> [--extra exp_name ...]`：读 `results/<exp>/annotations.parquet`（多轮取均值为预测）+ `results/baselines/baselines.parquet`，产出 `results/<exp>/evaluation/`（`metrics.csv`、`report.md`、多实验时 `heatmap.png`）

- [ ] **Step 1: 写失败测试**

`tests/test_report.py`：

```python
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
    table = metrics_table({"perfect": perfect[["movie_id", "tag_id", "score"]]},
                          truth, dataset, cfg, tag_groups=None, movie_ids=[2])
    assert len(table) == 1


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
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_report.py -v`
Expected: FAIL（ImportError）

- [ ] **Step 3: 实现 report.py**

**先调用 dataviz 技能再写 `modality_heatmap`**（含图表的代码须遵循其规范）。

`genometag/eval/report.py`：

```python
"""评估报告：指标表（CSV+markdown）、模态×标签类别热力图。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from genometag.config import ProtocolConfig
from genometag.eval.metrics import mae, ndcg_at_k, precision_at_k, rmse


def _wide(df: pd.DataFrame) -> pd.DataFrame:
    """(movie_id, tag_id, score) → movie × tag 宽矩阵。"""
    return df.pivot(index="movie_id", columns="tag_id", values="score")


def _evaluate_method(pred: pd.DataFrame, truth: pd.DataFrame,
                     cfg: ProtocolConfig) -> dict[str, float]:
    joined = pred.merge(truth, on=["movie_id", "tag_id"], suffixes=("_p", "_t"))
    p, t = joined["score_p"].to_numpy(), joined["relevance"].to_numpy()
    pw, tw = _wide(joined.rename(columns={"score_p": "score"})), _wide(
        joined.rename(columns={"relevance": "score"}))
    pk, nk = [], []
    for movie in pw.index:
        if movie in tw.index:
            pk.append(precision_at_k(pw.loc[movie].to_numpy(), tw.loc[movie].to_numpy(),
                                     cfg.top_k, cfg.relevance_threshold))
            nk.append(ndcg_at_k(pw.loc[movie].to_numpy(), tw.loc[movie].to_numpy(),
                                cfg.top_k, cfg.relevance_threshold))
    return {"rmse": rmse(p, t), "mae": mae(p, t),
            "precision@k": float(np.mean(pk)) if pk else 0.0,
            "ndcg@k": float(np.mean(nk)) if nk else 0.0}


def metrics_table(predictions: dict[str, pd.DataFrame], truth: pd.DataFrame,
                  dataset: pd.DataFrame, cfg: ProtocolConfig,
                  tag_groups: pd.DataFrame | None = None,
                  movie_ids: list[int] | None = None) -> pd.DataFrame:
    if movie_ids is not None:
        keep = set(movie_ids)
        truth = truth[truth["movie_id"].isin(keep)]
    rows = []
    for method, pred in predictions.items():
        if movie_ids is not None:
            pred = pred[pred["movie_id"].isin(set(movie_ids))]
        rows.append({"method": method, **_evaluate_method(pred, truth, cfg)})
    return pd.DataFrame(rows)


def modality_heatmap(rows: pd.DataFrame, out_png: Path) -> None:
    """模态 × 标签类别的 NDCG 热力图。实现前已按 dataviz 技能规范设计。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pivot = rows.pivot(index="modality_set", columns="tag_type", values="ndcg")
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pivot.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_xlabel("标签类别")
    ax.set_ylabel("模态组合")
    ax.set_title("NDCG：模态 × 标签类别")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="black" if v < 0.6 else "white")
    fig.colorbar(im, ax=ax, label="NDCG")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


report.py 顶部 import（最终版，覆盖 Step 3 开头给出的 import 块）：

```python
import json
from pathlib import Path

import numpy as np
import pandas as pd

from genometag.config import ProtocolConfig
from genometag.eval.metrics import (
    consistency_std, icc_2_1, mae, ndcg_at_k, precision_at_k, rmse,
)
from genometag.eval.protocol import cold_start_subset, tag_type_breakdown, temporal_split
```

`write_report`（追加在 modality_heatmap 之后）：

```python
def write_report(exp_dirs: list[Path], baseline_dir: Path | None, out_dir: Path,
                 cfg: ProtocolConfig, dataset: pd.DataFrame,
                 truth: pd.DataFrame, tag_groups: pd.DataFrame | None,
                 user_tags: pd.DataFrame | None) -> Path:
    """汇总多实验 + 基线 → evaluation/ 目录（metrics.csv、report.md、
    consistency.json、heatmap.png）。"""
    predictions: dict[str, pd.DataFrame] = {}
    for exp in exp_dirs:
        ann = pd.read_parquet(exp / "annotations.parquet")
        agg = (ann.dropna(subset=["score"])
               .groupby(["movie_id", "tag_id"], as_index=False)["score"].mean())
        modality = ann["modality_set"].iloc[0]
        predictions[f"MLLM[{modality}]"] = agg
    if baseline_dir and (baseline_dir / "baselines.parquet").exists():
        base = pd.read_parquet(baseline_dir / "baselines.parquet")
        for method, df in base.groupby("method"):
            predictions[str(method)] = df[["movie_id", "tag_id", "score"]]

    out_dir.mkdir(parents=True, exist_ok=True)
    # 评估电影集合 = 第一个实验实际标注到的电影（基线在同一集合上对比才公平）
    first_ann = pd.read_parquet(exp_dirs[0] / "annotations.parquet")
    movie_ids = sorted(first_ann["movie_id"].unique().tolist())

    sections: list[pd.DataFrame] = [metrics_table(
        predictions, truth, dataset, cfg, tag_groups, movie_ids).assign(subset="全部")]
    # 时序测试子集
    _, test_ids = temporal_split(dataset, cfg.temporal_split_year)
    sections.append(metrics_table(predictions, truth, dataset, cfg, tag_groups,
                                  [m for m in movie_ids if m in set(test_ids)]
                                  ).assign(subset=f"时序测试(≥{cfg.temporal_split_year})"))
    # 冷启动子集
    if user_tags is not None:
        cold = cold_start_subset(dataset, user_tags, cfg.cold_start_max_tags)
        sections.append(metrics_table(predictions, truth, dataset, cfg, tag_groups,
                                      [m for m in movie_ids if m in set(cold)]
                                      ).assign(subset="冷启动"))

    table = pd.concat(sections, ignore_index=True)
    table.to_csv(out_dir / "metrics.csv", index=False)
    md = ["# 评估报告\n", table.to_markdown(index=False), ""]

    # 一致性分析（spec §5.1）：轮次 > 1 时计算 ICC(2,1) 与平均轮次标准差
    n_runs = first_ann["run_id"].nunique()
    if n_runs > 1:
        wide_runs = first_ann.dropna(subset=["score"]).pivot_table(
            index=["movie_id", "tag_id"], columns="run_id", values="score")
        if wide_runs.shape[1] > 1 and len(wide_runs) > 1:
            matrix = wide_runs.to_numpy()
            consistency = {
                "n_runs": int(n_runs),
                "icc_2_1": icc_2_1(matrix),
                "mean_std": float(np.mean(consistency_std(matrix))),
            }
            (out_dir / "consistency.json").write_text(
                json.dumps(consistency, ensure_ascii=False, indent=1), encoding="utf-8")
            md.append(f"\n## 一致性（{n_runs} 轮调用）\n"
                      f"- ICC(2,1): {consistency['icc_2_1']:.3f}\n"
                      f"- 平均轮次标准差: {consistency['mean_std']:.3f}\n")
    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")

    # 热力图：多模态实验 + tag 类型信息时生成
    if len(exp_dirs) > 1 and tag_groups is not None:
        rows = []
        for exp in exp_dirs:
            ann = pd.read_parquet(exp / "annotations.parquet")
            agg = (ann.dropna(subset=["score"])
                   .groupby(["movie_id", "tag_id"], as_index=False)["score"].mean())
            for tag_type, tids in tag_type_breakdown(tag_groups).items():
                sub = agg[agg["tag_id"].isin(tids)]
                m = metrics_table({"m": sub}, truth, dataset, cfg, None, movie_ids)
                rows.append({"modality_set": ann["modality_set"].iloc[0],
                             "tag_type": tag_type, "ndcg": float(m["ndcg@k"].iloc[0])})
        modality_heatmap(pd.DataFrame(rows), out_dir / "heatmap.png")
    return out_dir
```

- [ ] **Step 4: 写脚本 06**

`scripts/06_evaluate.py`：

```python
"""06 · 评估：annotations + 基线 → 指标表 / 报告 / 热力图。"""
from __future__ import annotations

import argparse

import pandas as pd

from genometag.config import load_config, paths_for


def main(config_path: str = "configs/default.yaml",
         extra_exps: list[str] | None = None) -> int:
    config = load_config(config_path)
    paths = paths_for(config)
    from genometag.data.movielens import load_user_tags
    from genometag.eval.report import write_report
    from genometag.tags.grouping import load_tag_groups

    dataset = pd.read_parquet(paths.dataset)
    truth = pd.read_parquet(paths.genome_scores)
    tag_groups = load_tag_groups(paths.tag_groups)
    user_tags_path = paths.movielens / "tags.csv"
    user_tags = load_user_tags(user_tags_path) if user_tags_path.exists() else None

    exps = [paths.results] + [paths.root / "results" / e for e in (extra_exps or [])]
    exp_dirs = [e for e in exps if (e / "annotations.parquet").exists()]
    baseline_dir = paths.root / "results" / "baselines"
    out = write_report(exp_dirs, baseline_dir, paths.results / "evaluation",
                       config.protocol, dataset, truth, tag_groups, user_tags)
    print(f"[06] 评估完成 → {out}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--extra", nargs="*", default=[],
                        help="参与对比的其他实验名（如 ablation/video-only）")
    raise SystemExit(main(parser.parse_args().config, parser.parse_args().extra))
```

- [ ] **Step 5: 运行测试**

Run: `.venv/Scripts/python -m pytest tests/test_report.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add genometag/eval/report.py scripts/06_evaluate.py tests/test_report.py
git commit -m "feat: 评估报告（指标表/时序/冷启动子集/一致性分析/模态热力图）+ 06 脚本"
```

---

### Task 18: 端到端冒烟

**Files:**
- Create: `scripts/run_smoke.py`
- Test: `tests/test_smoke.py`

**Interfaces:**
- Consumes: 全部脚本 main（Task 3、6、11、12、15、17）
- Produces: `run_smoke.main() -> int`（清空 `results/smoke/` 后以 `configs/smoke.yaml` 串联 00→06，断言关键产物存在）

- [ ] **Step 1: 写失败测试**

`tests/test_smoke.py`：

```python
"""端到端冒烟：合成样本 + mock，零网络零成本。"""
from scripts.run_smoke import main


def test_smoke_pipeline():
    assert main() == 0
```

（scripts 目录无 `__init__.py` 时 `from scripts.run_smoke import ...` 不生效——Step 3 会一并创建空 `scripts/__init__.py`。）

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_smoke.py -v`
Expected: FAIL（ModuleNotFoundError: scripts.run_smoke）

- [ ] **Step 3: 实现 run_smoke.py**

`scripts/run_smoke.py`（另建空 `scripts/__init__.py`）：

```python
"""端到端冒烟：configs/smoke.yaml（合成样本 + mock）串联 00→06。

用途：日常回归 + pytest 集成测试。幂等：先清空 results/smoke/。
"""
from __future__ import annotations

import importlib
import json
import shutil

from genometag.config import load_config, paths_for

STEPS = ["00_make_synthetic", "01_verify_data", "02_build_dataset", "03_group_tags",
         "04_run_annotation", "05_run_baselines", "06_evaluate"]


def main() -> int:
    cfg_path = "configs/smoke.yaml"
    config = load_config(cfg_path)
    paths = paths_for(config)
    if paths.results.exists():
        shutil.rmtree(paths.results)          # 幂等

    for step in STEPS:
        mod = importlib.import_module(f"scripts.{step}")
        rc = mod.main(cfg_path)
        if rc != 0:
            print(f"[smoke] {step} 失败 (rc={rc})")
            return 1
        print(f"[smoke] {step} ✓")

    # 关键产物断言
    checks = [
        paths.dataset,
        paths.tag_groups,
        paths.results / "annotations.parquet",
        paths.results / "run_manifest.json",
        paths.root / "results" / "baselines" / "baselines.parquet",
        paths.results / "evaluation" / "metrics.csv",
        paths.results / "evaluation" / "consistency.json",   # smoke n_runs=2 → 一致性产出
    ]
    for c in checks:
        if not c.exists():
            print(f"[smoke] 缺少产物: {c}")
            return 1
    manifest = json.loads((paths.results / "run_manifest.json").read_text(encoding="utf-8"))
    print(f"[smoke] PASS — tasks={manifest.get('tasks')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

注记：(1) `00_make_synthetic.py` 的 `main(config_path)` 默认参数就是 smoke 配置，统一传 `cfg_path` 没有签名问题。(2) `05_run_baselines` 的输出固定写 `results/baselines/`（与实验目录无关），冒烟会覆盖已有基线输出——**接受**：基线便宜可重跑，06 每次评估重新读取；真实跑批顺序（先 05 后 06）已在 README 注明。

- [ ] **Step 4: 运行测试**

Run: `.venv/Scripts/python -m pytest tests/test_smoke.py -v`
Expected: 1 passed（终端逐行打印 `[smoke] xx ✓`）

Run: `.venv/Scripts/python scripts/run_smoke.py`
Expected: 同上，exit 0

- [ ] **Step 5: 全量回归**

Run: `.venv/Scripts/python -m pytest tests/ -v`
Expected: 此前全部测试 + smoke 通过（无 skip 之外的红）

- [ ] **Step 6: Commit**

```bash
git add scripts/run_smoke.py scripts/__init__.py tests/test_smoke.py
git commit -m "feat: 端到端冒烟脚本（00→06 串联 + 产物断言）与集成测试"
```

---

### Task 19: 文档与配置完备

**Files:**
- Modify: `README.md`（完整版）
- Create: `docs/download-guide.md`、`docs/data-format.md`、`docs/research-plan.md`（副本）、`configs/ablation/{video-only,script-only,dialogue-only,video-script,full}.yaml`、`configs/baselines.yaml`
- Test: `tests/test_configs_load.py`

**Interfaces:**
- Consumes: `load_config`（Task 2）
- Produces: configs/ 下全部 YAML 可加载并通过校验；文档齐备

- [ ] **Step 1: 写失败测试**

`tests/test_configs_load.py`：

```python
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
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_configs_load.py -v`
Expected: FAIL（ablation 配置不足 8 个文件）

- [ ] **Step 3: 创建全部配置文件**

`configs/ablation/video-only.yaml`（其余四个仅 `modalities` 与 `exp_name` 不同）：

```yaml
exp_name: ablation-video-only
data: {source: real, seed: 42}
model:
  model: gpt-4o
  temperature: 0.7
  max_frames: 12
  script_token_budget: 4000
  dialogue_token_budget: 3000
annotation:
  n_runs: 3
  modalities: [video]
  few_shot_examples: 2
protocol: {temporal_split_year: 2022, relevance_threshold: 0.5, top_k: 10}
mock: false
```

五个文件：`video-only.yaml`（`[video]`）、`script-only.yaml`（`[script]`）、`dialogue-only.yaml`（`[dialogue]`）、`video-script.yaml`（`[video, script]`）、`full.yaml`（`[video, script, dialogue]`）；`exp_name` 分别为 `ablation-video-only` 等。

`configs/baselines.yaml`：

```yaml
exp_name: baselines
data: {source: real, seed: 42}
protocol: {temporal_split_year: 2022, relevance_threshold: 0.5, top_k: 10}
baselines: {run: [random, mean, genre, bert]}
mock: false
```

- [ ] **Step 4: 写文档**

`docs/download-guide.md`：

```markdown
# 数据下载指南

## MovieLens（含 genome 数据）

1. 访问 https://grouplens.org/datasets/movielens/ ，下载 **ml-latest.zip**（或 ml-25m.zip，genome 文件结构相同）
2. 解压到 `data/raw/movielens/ml-latest/`，确保以下文件在位：
   - `genome-scores.csv`（movieId, tagId, relevance）
   - `genome-tags.csv`（tagId, tag；共 1,128 个）
   - `movies.csv`、`links.csv`、`tags.csv`
3. 校验：`.venv/Scripts/python scripts/01_verify_data.py`

## MovieNet

1. 访问 https://movienet.site/ ，按其流程申请/下载（通常含元数据 CSV 与各模态数据的分块下载）
2. 解压并整理到 `data/raw/movienet/`，目标结构（**当前适配器假定的格式**，
   与实际发布有出入时改 `genometag/data/movienet.py` 顶部的常量与 `load_movienet`）：

   ```
   data/raw/movienet/
   ├── movies.csv            # movie_id,title,year,imdb_id
   ├── keyframes/<id>/*.jpg
   ├── scripts/<id>.txt      # 覆盖不足时字幕可作台词模态代理
   └── subtitles/<id>.srt
   ```

3. 校验：`01_verify_data.py` 会报告**年份分布**——spec §9 风险：MovieNet 可能不含
   2022 年后电影，届时需重新确定时序分割年份（改配置 `temporal_split_year` 一行）。
```

`docs/data-format.md`：

```markdown
# 数据产物 Schema

## data/interim/aligned.parquet
| 列 | 类型 | 说明 |
|----|------|------|
| movie_id | int | MovieLens movieId（全管道主键） |
| title | str | 去年份的标题 |
| year | int | 上映年 |
| genres | list[str] | |
| imdb_id | int | 规范化 IMDb 数字 ID |
| movienet_id | str | |
| match_method | str | imdb / title |

## data/interim/align_report.md
匹配率、歧义清单（同名同年）、未匹配清单。

## data/processed/dataset.parquet
上述列 + `keyframes: list[str]`（相对 movienet 根）、`script_path`、`subtitles_path`（相对路径或 null）。

## data/processed/genome_scores.parquet
`movie_id, tag_id, relevance`（仅对齐电影；ground truth）。

## data/processed/genome_tags.parquet
`tag_id, tag`。

## data/processed/tag_groups.json（入 git）
`[{tag_id, tag, group, tag_type: content|preference, definition}]`。

## results/<exp>/annotations.jsonl（append-only，续跑事实来源）
每行 `{"task_key": "movie|group|run", "status": "done|parse_failed|no_content",
"rows": [...], "usage": {...}}`。

## results/<exp>/annotations.parquet
`movie_id, tag_id, group, modality_set, run_id, score`。

## results/baselines/baselines.parquet
`movie_id, tag_id, score, method`。
```

`docs/research-plan.md`：复制上级目录 `../方向A_GenomeTag-MLLM_研究计划.md` 的完整内容（`cp ../方向A_GenomeTag-MLLM_研究计划.md docs/research-plan.md`）。

`README.md` 完整版：

```markdown
# GenomeTag-MLLM

MLLM 从电影多模态内容（关键帧 + 剧本 + 台词）预测 MovieLens genome tag 关联度评分。
研究问题、章节计划与文献定位见 `docs/research-plan.md`；框架设计见
`docs/superpowers/specs/`。

## 安装

​```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
# 可选（BERT 基线）：
.venv/Scripts/python -m pip install -r requirements-baselines.txt
cp .env.example .env   # 填入 OPENAI_API_KEY / OPENAI_BASE_URL
​```

## 验证安装（零成本）

​```bash
.venv/Scripts/python -m pytest tests/ -v      # 全部测试：无网络、无 key
.venv/Scripts/python scripts/run_smoke.py     # 端到端冒烟（合成样本 + mock）
​```

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
```

- [ ] **Step 5: 运行测试与冒烟**

Run: `.venv/Scripts/python -m pytest tests/test_configs_load.py -v`
Expected: 8+ passed

Run: `.venv/Scripts/python -m pytest tests/ -q`
Expected: 全绿

- [ ] **Step 6: Commit**

```bash
git add README.md docs/ configs/ tests/test_configs_load.py
git commit -m "docs: 下载指南/数据格式/研究计划副本 + 消融配置全套 + README 完整版"
```

---

### Task 20: 最终验证

**Files:**
- Modify: 无新文件（验证与收尾）

**Interfaces:**
- Consumes: 全部
- Produces: 验证记录（全测试通过 + 冒烟通过 + git 状态干净）

- [ ] **Step 1: 全量测试**

Run: `.venv/Scripts/python -m pytest tests/ -v`
Expected: 全部通过（BERT 测试 skipped 属正常），输出无 error

- [ ] **Step 2: 冒烟**

Run: `.venv/Scripts/python scripts/run_smoke.py`
Expected: `PASS`，exit 0

- [ ] **Step 3: 检查 spec 覆盖（人工核对清单）**

对照 spec 逐项确认：

- [ ] §3 数据层：movienet/movielens/align/sample 四模块 + 00/01/02 脚本
- [ ] §4 标注层：grouping/client/cache/prompts/pipeline/mock + 03/04 脚本；缓存 run_id 入键；mock 零网络；`--dry-run` 落地（Task 12）
- [ ] §5 评估层：metrics/protocol/report + 06 脚本；防泄露约束落实（few-shot/mean/genre/bert 四处）；一致性分析（ICC）在 write_report 中产出（Task 17）
- [ ] §5.3 四基线 + 05 脚本（bert 可选跳过）
- [ ] §6 冒烟路径 = pytest 集成测试
- [ ] §7 测试策略：单元 + 集成 + 无网络（BERT 显式豁免）
- [ ] §8 依赖分层：requirements / requirements-baselines
- [ ] 目录结构与 spec §2 一致

- [ ] **Step 4: git 收尾**

```bash
git status   # 确认无未跟踪的实现文件（data/、results/、.cache/ 应被 ignore）
git log --oneline   # 每任务一提交
```

---

## Self-Review 记录

（writing-plans 自审：以下为检查过程中发现并已在定稿中修复的问题记录）

1. **Spec 覆盖**：spec §2-§8 全部章节映射到 Task 1-19；§9 风险的"年份分布检查"落在 Task 6 的 `01_verify_data.py`。发现并修复三处覆盖缺口：(a) spec §4.4 要求 tag 一句话定义但 tag_groups.json schema 缺字段 → 已修订 spec（`definition` 字段），Task 3/11 同步；(b) spec §4.5 的 `--dry-run` 初稿未实现 → Task 12 补齐（`run_annotation(dry_run=True)` + 脚本 flag + 测试）；(c) spec §5.1/§5.5 的一致性分析（ICC）初稿无计算入口 → Task 17 的 `write_report` 在轮次 > 1 时产出 `consistency.json` 并有测试锁定。
2. **占位符扫描**：初稿 Task 8/11/12/17/18 含"草稿 + 实现注记修正"双版本代码——违反 No Placeholders 原则，已全部清理为单一最终版（`_JsonFormatUnsupported` 双路径、`tag_map` 残行、`write_report` 残尾、`zero0_placeholder` 导入、`__import__("pandas")` hack 均已消除）。
3. **类型一致性**：`LLMResponse`（Task 8）被 Task 9/11/12 消费；`TagDef/FewShot`（Task 10）被 Task 12 消费；`select_few_shots` 签名带 `tag_names` 参数且测试与实现一致；`build_dataset` 移入库层（align.py）供 Task 12 fixture 复用，Task 12 测试不再 monkeypatch；基线长表 schema `movie_id, tag_id, score, method` 在 Task 15/16/17 一致；`write_report` 签名与 Task 17 脚本 06 调用一致。
4. **Review Focus 五项**均已在对应任务测试中落地（见各测试文件名）；另补 `test_write_report_and_consistency` 覆盖主输出路径。
