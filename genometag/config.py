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
    extra_body: dict = field(default_factory=dict)  # 端点专有参数透传（如 DashScope {enable_thinking: false}）


@dataclass
class AnnotationConfig:
    n_runs: int = 3
    modalities: list[str] = field(default_factory=lambda: list(VALID_MODALITIES))
    few_shot_examples: int = 2
    workers: int = 1   # 并行线程数（1=串行）。任务相互独立，结果与串行等价（有测试钉死）


@dataclass
class ProtocolConfig:
    temporal_split_year: int = 2022
    relevance_threshold: float = 0.5
    top_k: int = 10
    cold_start_max_tags: int = 0


@dataclass
class BaselinesConfig:
    run: list[str] = field(default_factory=lambda: ["random", "mean", "genre", "bert"])


@dataclass
class ProbesConfig:
    """记忆探针 2（错配内容）参数——spec 2026-10-16 修订。"""
    n_pairs: int = 150          # 电影对总数（同类型/跨类型各半）


# 顶层合法键：未知键一律报错（防止拼错/过时配置被静默忽略）。
# 说明：旧 consistency 块已移除（原为死配置，会被静默忽略）——
# 其意图用 annotation.n_runs / data.movie_limit 表达即可。
KNOWN_TOP_LEVEL_KEYS = {
    "exp_name", "data", "model", "annotation", "protocol", "baselines", "probes",
    "mock",
}


@dataclass
class AppConfig:
    exp_name: str = "default"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    annotation: AnnotationConfig = field(default_factory=AnnotationConfig)
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    baselines: BaselinesConfig = field(default_factory=BaselinesConfig)
    probes: ProbesConfig = field(default_factory=ProbesConfig)
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
    if cfg.annotation.workers < 1:
        raise ValueError("annotation.workers 必须 ≥ 1（1=串行）")
    if not 0.0 < cfg.protocol.relevance_threshold < 1.0:
        raise ValueError("protocol.relevance_threshold 必须在 (0,1) 开区间")
    if cfg.protocol.top_k < 1:
        raise ValueError("protocol.top_k 必须 ≥ 1")
    if cfg.data.movie_limit is not None and cfg.data.movie_limit < 1:
        raise ValueError("data.movie_limit 为 None 或 ≥ 1")
    if not isinstance(cfg.model.extra_body, dict):
        raise ValueError("model.extra_body 必须是字典，如 {enable_thinking: false}")


def load_config(path: str | Path) -> AppConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    unknown = set(raw) - KNOWN_TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(
            f"配置含未知顶层键: {sorted(unknown)}（合法键: {sorted(KNOWN_TOP_LEVEL_KEYS)}）。"
            "已移除的 consistency 块用 annotation.n_runs / data.movie_limit 替代")
    cfg = AppConfig(
        exp_name=raw.get("exp_name", "default"),
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        annotation=AnnotationConfig(**raw.get("annotation", {})),
        protocol=ProtocolConfig(**raw.get("protocol", {})),
        baselines=BaselinesConfig(**raw.get("baselines", {})),
        probes=ProbesConfig(**raw.get("probes", {})),
        mock=bool(raw.get("mock", False)),
    )
    validate_config(cfg)
    return cfg
