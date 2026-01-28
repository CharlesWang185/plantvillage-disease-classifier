# src/metrics.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


ArrayLike = Union[List[int], np.ndarray]


def _to_numpy_int(x: ArrayLike) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr.astype(int)


def compute_accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Top-1 accuracy.
    """
    y_true = _to_numpy_int(y_true)
    y_pred = _to_numpy_int(y_pred)
    return float(accuracy_score(y_true, y_pred))


def compute_topk_accuracy(y_true: ArrayLike, y_logits: np.ndarray, k: int = 3) -> float:
    """
    Top-k accuracy for multiclass classification.

    Parameters:
    - y_true: shape (N,)
    - y_logits: shape (N, C) raw logits OR probabilities
    - k: top-k

    Returns:
    - float accuracy in [0,1]
    """
    y_true = _to_numpy_int(y_true)
    if y_logits.ndim != 2:
        raise ValueError(f"y_logits must have shape (N, C), got {y_logits.shape}")
    if k <= 0:
        raise ValueError("k must be positive")

    # indices of top-k predictions for each sample
    topk = np.argsort(y_logits, axis=1)[:, -k:]
    correct = np.any(topk == y_true[:, None], axis=1)
    return float(np.mean(correct))


def compute_classification_report(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    class_names: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Returns:
    - per_class_df: DataFrame with precision/recall/f1/support for each class
    - summary: dict with overall accuracy + macro/weighted averages
    """
    y_true = _to_numpy_int(y_true)
    y_pred = _to_numpy_int(y_pred)

    # classification_report can output a dict
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # Extract per-class rows (skip avg rows)
    # If class_names is provided, report_dict keys are class names (strings)
    # Otherwise, keys are string versions of indices: "0", "1", ...
    per_class_rows = {}
    for key, val in report_dict.items():
        if key in {"accuracy", "macro avg", "weighted avg"}:
            continue
        per_class_rows[key] = val

    per_class_df = pd.DataFrame(per_class_rows).T
    # Make column order nice
    per_class_df = per_class_df[["precision", "recall", "f1-score", "support"]]

    # Build summary
    summary = {
        "accuracy": float(report_dict.get("accuracy", accuracy_score(y_true, y_pred))),
        "macro_precision": float(report_dict["macro avg"]["precision"]),
        "macro_recall": float(report_dict["macro avg"]["recall"]),
        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "weighted_f1": float(report_dict["weighted avg"]["f1-score"]),
    }

    return per_class_df, summary


def compute_confusion_matrix(y_true: ArrayLike, y_pred: ArrayLike, num_classes: Optional[int] = None) -> np.ndarray:
    """
    Confusion matrix of shape (C, C) where rows=true classes and cols=pred classes.
    """
    y_true = _to_numpy_int(y_true)
    y_pred = _to_numpy_int(y_pred)

    labels = None
    if num_classes is not None:
        labels = list(range(num_classes))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return cm.astype(int)


def evaluate_predictions(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_logits: Optional[np.ndarray] = None,
    class_names: Optional[Sequence[str]] = None,
    topk: int = 3,
    num_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    One-stop evaluation helper.

    Returns a dict containing:
      - summary metrics (accuracy, macro_f1, etc.)
      - per_class_df (as DataFrame)
      - confusion_matrix (as numpy array)
      - optional topk accuracy if y_logits is provided
    """
    per_class_df, summary = compute_classification_report(y_true, y_pred, class_names=class_names)
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=num_classes)

    out: Dict[str, Any] = {
        "summary": summary,
        "per_class_df": per_class_df,
        "confusion_matrix": cm,
    }

    if y_logits is not None:
        out["topk_accuracy"] = compute_topk_accuracy(y_true, y_logits, k=topk)

    return out
