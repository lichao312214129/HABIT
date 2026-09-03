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
"""Shared machinery for the built-in evaluation metrics."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from habit.exceptions import HABITAPIError

__all__ = ["confusion_matrix", "binary_class_scores"]


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Count label co-occurrences exactly like sklearn's ``confusion_matrix``.

    Rows index the true labels, columns the predicted ones, and both follow
    the sorted union of the observed labels -- the same convention sklearn
    uses when no explicit ``labels`` argument is given, so the v0.1 metric
    formulas keep their exact numerics without importing sklearn.

    Args:
        y_true: True class labels, one-dimensional.
        y_pred: Predicted class labels, aligned to ``y_true``.

    Returns:
        Tuple ``(cm, labels)`` with ``cm`` of shape ``(n_labels, n_labels)``
        and ``labels`` the sorted label values indexing it.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise HABITAPIError(
            f"y_true and y_pred must have the same shape; got "
            f"{y_true.shape} and {y_pred.shape}."
        )
    labels = np.union1d(y_true, y_pred)
    # np.searchsorted on the sorted union gives each sample's label index.
    true_index = np.searchsorted(labels, y_true)
    pred_index = np.searchsorted(labels, y_pred)
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    np.add.at(cm, (true_index, pred_index), 1)
    return cm, labels


def binary_class_scores(
    y_score: Optional[np.ndarray],
    *,
    owner: str,
) -> Optional[np.ndarray]:
    """
    Normalise the positive-class score vector for binary metrics.

    Args:
        y_score: Scores passed to a ``needs_proba`` metric: either a
            one-dimensional array, an ``(n, 1)`` column (flattened), or an
            ``(n, k)`` probability matrix.
        owner: Human-readable metric name for the error message.

    Returns:
        The one-dimensional positive-class scores, or ``None`` when
        ``y_score`` is a multi-column probability matrix (a genuinely
        multi-class problem, where the binary metric is undefined and the
        caller answers ``NaN``).

    Raises:
        HABITAPIError: If ``y_score`` is missing (the metric declared
            ``needs_proba``) or not one- or two-dimensional.
    """
    if y_score is None:
        raise HABITAPIError(
            f"metric.{owner} requires y_score (it is a probability/score "
            "metric); pass the positive-class probabilities."
        )
    scores = np.asarray(y_score, dtype=np.float64)
    if scores.ndim == 1:
        return scores
    if scores.ndim == 2 and scores.shape[1] == 1:
        return scores[:, 0]
    if scores.ndim == 2:
        return None
    raise HABITAPIError(
        f"metric.{owner} expected a 1-D or 2-D score array; got "
        f"{scores.ndim} dimensions."
    )
