"""Dependency-light five-class metrics used in validation."""

from __future__ import annotations

import numpy as np


def confusion_matrix(
    targets: np.ndarray,
    predictions: np.ndarray,
    classes: int,
) -> np.ndarray:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (targets, predictions), 1)
    return matrix


def macro_f1(matrix: np.ndarray) -> float:
    true_positive = np.diag(matrix).astype(np.float64)
    precision_denominator = matrix.sum(axis=0)
    recall_denominator = matrix.sum(axis=1)
    precision = np.divide(
        true_positive,
        precision_denominator,
        out=np.zeros_like(true_positive),
        where=precision_denominator != 0,
    )
    recall = np.divide(
        true_positive,
        recall_denominator,
        out=np.zeros_like(true_positive),
        where=recall_denominator != 0,
    )
    denominator = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )
    return float(f1.mean())


def cohens_kappa(matrix: np.ndarray) -> float:
    total = matrix.sum()
    observed = np.trace(matrix) / total
    expected = float(matrix.sum(axis=0) @ matrix.sum(axis=1)) / total**2
    return float((observed - expected) / (1.0 - expected))


def multiclass_mcc(matrix: np.ndarray) -> float:
    total = matrix.sum()
    correct = np.trace(matrix)
    predicted = matrix.sum(axis=0)
    actual = matrix.sum(axis=1)
    numerator = correct * total - predicted @ actual
    denominator = np.sqrt(
        (total**2 - predicted @ predicted)
        * (total**2 - actual @ actual)
    )
    return float(numerator / denominator) if denominator else 0.0


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def macro_ovr_auc(targets: np.ndarray, probabilities: np.ndarray) -> float:
    aucs: list[float] = []
    for class_index in range(probabilities.shape[1]):
        positive = targets == class_index
        positive_count = int(positive.sum())
        negative_count = len(targets) - positive_count
        if positive_count == 0 or negative_count == 0:
            continue
        ranks = _average_ranks(probabilities[:, class_index])
        positive_rank_sum = ranks[positive].sum()
        auc = (
            positive_rank_sum
            - positive_count * (positive_count + 1) / 2.0
        ) / (positive_count * negative_count)
        aucs.append(float(auc))
    return float(np.mean(aucs))


def classification_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    predictions = probabilities.argmax(axis=1)
    matrix = confusion_matrix(targets, predictions, probabilities.shape[1])
    return {
        "accuracy": float((predictions == targets).mean()),
        "macro_f1": macro_f1(matrix),
        "macro_auc": macro_ovr_auc(targets, probabilities),
        "cohens_kappa_unweighted": cohens_kappa(matrix),
        "mcc": multiclass_mcc(matrix),
    }
