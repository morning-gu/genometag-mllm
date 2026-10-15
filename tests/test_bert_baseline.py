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
