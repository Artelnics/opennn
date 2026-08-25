"""Binary-classification metrics, scored the same way for every engine.

Step 3 of REORGANIZATION_PLAN.md. Lifted from the two copies that existed:
`throughput/higgs/metrics.py` and `quality/accuracy/metrics.py`.

This is a lift rather than a merge, but only just. The accuracy copy carried a
docstring claiming it was "kept identical to benchmarks/throughput/higgs/
metrics.py"; it was not. It had fallen 17 lines behind, missing
`passes_quality_gate` and `parse_optional_float` entirely. The shared portion
was verified byte-identical -- the accuracy copy is a strict prefix of the
HIGGS one -- so the HIGGS version is a pure superset and adopting it changes
no number either copy produced.

Worth recording because it is the failure mode the ledger exists to catch: a
comment asserting two files agree is not evidence that they do, and here the
assertion had been false for long enough that nobody noticed.
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

def passes_quality_gate(
    metrics: dict[str, float],
    min_accuracy: float | None,
    max_log_loss: float | None,
    min_auc: float | None,
) -> bool:
    if min_accuracy is not None and metrics["test_accuracy"] < min_accuracy:
        return False
    if max_log_loss is not None and metrics["test_log_loss"] > max_log_loss:
        return False
    if min_auc is not None:
        auc = metrics["test_roc_auc"]
        if math.isnan(auc) or auc < min_auc:
            return False
    return True

def parse_optional_float(value: str | None) -> float | None:
    if value is None or value == "" or value.lower() in {"none", "nan"}:
        return None
    return float(value)
