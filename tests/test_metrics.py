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
