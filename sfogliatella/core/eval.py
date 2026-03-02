"""Evaluation metrics for regression, classification, and other tasks."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def smape(y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-8) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + eps
    return float(np.mean(np.abs(y_pred - y_true) / denom) * 100.0)


def mape(y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.mean(np.abs((y_pred - y_true) / (np.abs(y_true) + eps))) * 100.0)


def r2(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def regression_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_true).ravel()
    return {
        "mse":   mse(y_pred, y_true),
        "rmse":  rmse(y_pred, y_true),
        "mae":   mae(y_pred, y_true),
        "smape": smape(y_pred, y_true),
        "r2":    r2(y_pred, y_true),
    }


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def accuracy(logits_or_preds: np.ndarray, y_true: np.ndarray) -> float:
    """Works for both class predictions and logit arrays."""
    arr = np.asarray(logits_or_preds)
    if arr.ndim > 1:
        preds = np.argmax(arr, axis=-1)
    else:
        preds = (arr > 0.5).astype(int)
    return float(np.mean(preds == np.asarray(y_true).ravel()))


def binary_metrics(proba: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    proba = np.asarray(proba).ravel()
    y_true = np.asarray(y_true).ravel().astype(int)
    preds = (proba >= threshold).astype(int)
    tp = int(np.sum((preds == 1) & (y_true == 1)))
    fp = int(np.sum((preds == 1) & (y_true == 0)))
    fn = int(np.sum((preds == 0) & (y_true == 1)))
    tn = int(np.sum((preds == 0) & (y_true == 0)))
    acc = (tp + tn) / max(1, len(y_true))
    prec = tp / max(1, tp + fp)
    rec  = tp / max(1, tp + fn)
    f1   = 2 * prec * rec / max(1e-8, prec + rec)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def classification_metrics(
    logits: np.ndarray,
    y_true: np.ndarray,
    task: str = "binary",
    threshold: float = 0.5,
) -> Dict[str, float]:
    if task == "binary":
        proba = 1.0 / (1.0 + np.exp(-np.asarray(logits).ravel()))
        return binary_metrics(proba, y_true, threshold)
    else:
        acc = accuracy(logits, y_true)
        return {"accuracy": acc}


# ---------------------------------------------------------------------------
# Composite dispatcher
# ---------------------------------------------------------------------------

def compute_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    task: str = "regression",
    **kwargs,
) -> Dict[str, float]:
    if task == "regression":
        return regression_metrics(y_pred, y_true)
    elif task in ("classification", "binary"):
        return classification_metrics(y_pred, y_true, task="binary", **kwargs)
    elif task == "multiclass":
        return classification_metrics(y_pred, y_true, task="multiclass", **kwargs)
    else:
        # Fallback: regression metrics
        return regression_metrics(y_pred, y_true)
