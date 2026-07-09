"""Shared binary-classification metrics for the HIGGS accuracy-parity benchmark.

Kept identical to docs/benchmarks/throughput/higgs/metrics.py so OpenNN,
PyTorch, and TensorFlow are all scored the same way.
"""

from __future__ import annotations

import math

import numpy as np


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    positives = y >= 0.5
    n_pos = int(positives.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(y.size, dtype=np.float64)
    sorted_scores = s[order]

    i = 0
    while i < y.size:
        j = i + 1
        while j < y.size and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j

    sum_pos_ranks = float(ranks[positives].sum())
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def binary_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    p = np.clip(p, 1.0e-7, 1.0 - 1.0e-7)

    accuracy = float(((p >= 0.5) == (y >= 0.5)).mean())
    log_loss = float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())
    auc = float(roc_auc(y, p))

    return {
        "test_accuracy": accuracy,
        "test_log_loss": log_loss,
        "test_roc_auc": auc,
    }
