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
