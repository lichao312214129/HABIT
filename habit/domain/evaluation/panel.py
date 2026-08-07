# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Binary classification metric panels used by model comparison.

Pure arrays in / dict out. NaN rows are dropped before scoring so pairwise
model overlays and threshold searches see the same cleaned subjects that the
v0.1 comparison workflow used via ``PredictionContainer.clean_nan``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from habit.domain.evaluation.metrics import (  # noqa: F401 — register builtins
    AccuracyMetric,
    AucMetric,
    F1ScoreMetric,
    HosmerLemeshowPValueMetric,
    NpvMetric,
    PpvMetric,
    SensitivityMetric,
    SpecificityMetric,
    SpiegelhalterZPValueMetric,
)
from habit.domain.evaluation.registry import MetricRegistry
from habit.exceptions import HABITAPIError

__all__ = [
    "CleanedPredictions",
    "clean_binary_predictions",
    "compute_classification_metrics",
]

#: Metric names written into comparison ``metrics.json`` (stable order).
_PANEL_METRIC_NAMES: Tuple[str, ...] = (
    "accuracy",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "f1_score",
    "auc",
    "hosmer_lemeshow_p_value",
    "spiegelhalter_z_p_value",
)


@dataclass(frozen=True)
class CleanedPredictions:
    """Finite-row prediction triple ready for metric / threshold routines."""

    y_true: np.ndarray
    y_prob: np.ndarray
    y_pred: np.ndarray


def clean_binary_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    *,
    threshold: float = 0.5,
) -> CleanedPredictions:
    """
    Drop NaN rows and materialise hard labels when missing.

    Args:
        y_true: Ground-truth labels.
        y_prob: Positive-class probabilities (1-D).
        y_pred: Optional hard labels; when omitted, ``y_prob >= threshold``.
        threshold: Fallback decision threshold when ``y_pred`` is absent.

    Returns:
        Cleaned arrays of equal length (possibly empty).

    Raises:
        HABITAPIError: When input lengths disagree before cleaning.
    """
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    if yt.shape[0] != yp.shape[0]:
        raise HABITAPIError(
            "clean_binary_predictions: y_true and y_prob length mismatch "
            f"({yt.shape[0]} vs {yp.shape[0]})."
        )
    if y_pred is None:
        yhat = (yp >= float(threshold)).astype(np.float64)
        mask = np.isfinite(yt) & np.isfinite(yp)
    else:
        yhat = np.asarray(y_pred, dtype=np.float64).reshape(-1)
        if yhat.shape[0] != yt.shape[0]:
            raise HABITAPIError(
                "clean_binary_predictions: y_pred length mismatch "
                f"({yhat.shape[0]} vs {yt.shape[0]})."
            )
        mask = np.isfinite(yt) & np.isfinite(yp) & np.isfinite(yhat)

    return CleanedPredictions(
        y_true=yt[mask],
        y_prob=yp[mask],
        y_pred=yhat[mask],
    )


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """
    Compute the standard binary classification metric panel.

    Args:
        y_true: Clean ground-truth labels.
        y_pred: Clean hard predictions.
        y_prob: Clean positive-class probabilities.

    Returns:
        Mapping of metric name -> float (``nan`` when a metric fails).
    """
    yt = np.asarray(y_true)
    yhat = np.asarray(y_pred)
    yp = np.asarray(y_prob)
    out: Dict[str, float] = {}
    for name in _PANEL_METRIC_NAMES:
        try:
            metric = MetricRegistry.create(name)
            value = metric(yt, yhat, yp)
            out[name] = float(value) if value is not None else float("nan")
        except Exception:  # noqa: BLE001 — match v0.1 soft-fail semantics
            out[name] = float("nan")
    return out
