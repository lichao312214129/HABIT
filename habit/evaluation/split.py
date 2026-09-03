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
"""Endpoint-aware train/validation splitting.

The split that keeps a survival analysis honest is stratification on the
EVENT indicator, not on time: the quantity a model must see in every fold is
the event rate, whereas splitting on follow-up duration would systematically
assign the long-surviving (mostly censored) patients to one side and bias the
C-index. For classification the same helper stratifies on the label; for
regression it falls back to an unstratified shuffle, because a continuous
response has no natural strata. This module centralises that single decision
so the CLI, the recipes and the tests all split the same way.

The functions live in the domain layer and operate on plain arrays, so they
stay free of configuration concepts and are reusable by any driver.
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np

from habit.exceptions import HABITAPIError
from habit.contracts.outcome import Outcome

__all__ = [
    "stratify_labels",
    "train_test_indices",
    "kfold_indices",
]


def stratify_labels(outcome: Optional[Outcome], frame) -> Optional[np.ndarray]:
    """
    Return the per-row label to stratify on, or ``None`` to skip stratifying.

    Args:
        outcome: The declared endpoint (or ``None`` for unsupervised tables).
        frame: Frame carrying the endpoint columns.

    Returns:
        The stratification key: the event indicator for survival (its boolean
        coding is itself the two strata), the class label for binary and
        multiclass endpoints, and ``None`` for continuous endpoints, which
        have no categorical strata.
    """
    if outcome is None:
        return None
    if outcome.task == "survival":
        # The boolean event mask IS the two strata: observed vs censored.
        return outcome.event_mask(frame).astype(int).to_numpy()
    if outcome.task in ("binary", "multiclass"):
        return frame[outcome.columns[0]].to_numpy()
    # Continuous endpoints have no strata; regressions split unstratified.
    return None


def _check_stratify_feasible(labels: np.ndarray, n_splits: int, owner: str) -> None:
    """Fail loudly when a stratum is too small for the requested folds."""
    _, counts = np.unique(labels, return_counts=True)
    if counts.size < 2:
        raise HABITAPIError(
            f"{owner}: stratification needs at least two distinct strata "
            f"(e.g. both events and censored rows); the data has only one."
        )
    if int(counts.min()) < n_splits:
        raise HABITAPIError(
            f"{owner}: the smallest stratum has {int(counts.min())} rows but "
            f"{n_splits} splits were requested, so at least one fold would "
            "hold none of it. Use fewer folds or more data."
        )


def train_test_indices(
    n_samples: int,
    *,
    test_size: float = 0.3,
    labels: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return train and test row indices, stratified when labels are given.

    Args:
        n_samples: Number of rows to split.
        test_size: Fraction assigned to the test side.
        labels: Stratification key from :func:`stratify_labels`; ``None``
            yields an unstratified shuffle.
        seed: Seed for reproducibility.

    Returns:
        Tuple ``(train_index, test_index)`` of integer row positions.

    Raises:
        HABITAPIError: If stratification is impossible (a stratum of one).
    """
    from sklearn.model_selection import train_test_split

    indices = np.arange(n_samples)
    if labels is not None:
        labels = np.asarray(labels)
        _, counts = np.unique(labels, return_counts=True)
        if counts.size < 2 or int(counts.min()) < 2:
            raise HABITAPIError(
                "train_test_indices: stratification needs at least two rows "
                "in each of two strata; the labels supplied do not allow it."
            )
    train_index, test_index = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    return np.asarray(train_index), np.asarray(test_index)


def kfold_indices(
    n_samples: int,
    *,
    n_splits: int = 5,
    labels: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Yield (train, validation) index pairs for K folds.

    Args:
        n_samples: Number of rows to split.
        n_splits: Number of folds.
        labels: Stratification key from :func:`stratify_labels`; ``None``
            yields a plain (unstratified) K-fold.
        seed: Seed for reproducibility.

    Yields:
        Tuple ``(train_index, validation_index)`` per fold.

    Raises:
        HABITAPIError: If a stratum is smaller than ``n_splits``.
    """
    if labels is not None:
        labels = np.asarray(labels)
        _check_stratify_feasible(labels, n_splits, owner="kfold_indices")
        from sklearn.model_selection import StratifiedKFold

        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        splits = splitter.split(np.zeros(n_samples), labels)
    else:
        from sklearn.model_selection import KFold

        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = splitter.split(np.arange(n_samples))
    for train_index, validation_index in splits:
        yield np.asarray(train_index), np.asarray(validation_index)
