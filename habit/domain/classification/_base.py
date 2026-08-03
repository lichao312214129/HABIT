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
"""Shared machinery for the built-in classifiers."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

from habit.api.exceptions import HABITAPIError
from habit.contracts.table import FeatureTable
from habit.domain.feature_selection._base import outcome_series

__all__ = ["SklearnClassifierBase"]


class SklearnClassifierBase:
    """
    Shared fit/predict bookkeeping for estimator-backed classifiers.

    The feature column set is captured at ``fit`` time and validated on every
    predict call, so a prediction table whose schema drifted from the training
    table fails loudly instead of silently predicting on misaligned columns.
    Stochastic estimators receive the seed set via :meth:`set_random_state`
    (v1.0 naming decisions: never a constructor parameter). Subclasses set
    ``_spec_name`` and implement :meth:`_build_estimator`.
    """

    _spec_name: str = ""

    def __init__(self) -> None:
        self._seed: Optional[int] = None
        self._estimator: Optional[Any] = None
        self._fit_columns: Tuple[str, ...] = ()
        self._classes: Optional[np.ndarray] = None

    def set_random_state(self, seed: int) -> None:
        """Set the random state used when the estimator is built at fit time."""
        self._seed = int(seed)

    def _build_estimator(self) -> Any:  # pragma: no cover - subclasses override
        """Construct the underlying estimator (lazy heavy imports live here)."""
        raise NotImplementedError

    def _predict_proba_matrix(self, X: pd.DataFrame) -> np.ndarray:
        """Return the class-probability matrix; overridden for SVM (LinearSVC)."""
        return self._estimator.predict_proba(X)

    def fit(self, table: FeatureTable) -> "SklearnClassifierBase":
        """Train on a table with an outcome column."""
        y = outcome_series(table, owner=f"classifier.{self._spec_name}")
        X = table.feature_matrix()
        self._estimator = self._build_estimator()
        self._estimator.fit(X, y)
        self._fit_columns = tuple(table.feature_columns)
        self._classes = np.asarray(self._estimator.classes_)
        return self

    def _matrix_for_predict(self, table: FeatureTable) -> pd.DataFrame:
        """Return the table's feature matrix restricted to the fit schema."""
        if self._estimator is None:
            raise HABITAPIError(
                f"classifier.{self._spec_name} must be fitted before predict."
            )
        missing = [
            column for column in self._fit_columns
            if column not in table.feature_columns
        ]
        if missing:
            raise HABITAPIError(
                f"classifier.{self._spec_name} was fitted on feature columns "
                f"{list(self._fit_columns)} but the table to predict does not "
                f"declare {missing} as feature columns. Apply the same "
                "upstream pipeline steps as at fit time."
            )
        return table.feature_matrix()[list(self._fit_columns)]

    def predict(self, table: FeatureTable) -> pd.Series:
        """Predict class labels indexed by the table's identifier columns."""
        X = self._matrix_for_predict(table)
        labels = self._estimator.predict(X)
        return pd.Series(
            np.asarray(labels),
            index=X.index,
            name=table.outcome_column or "prediction",
        )

    def predict_proba(self, table: FeatureTable) -> pd.DataFrame:
        """Predict class probabilities indexed by the identifier columns."""
        X = self._matrix_for_predict(table)
        proba = self._predict_proba_matrix(X)
        return pd.DataFrame(
            np.asarray(proba),
            index=X.index,
            columns=[str(label) for label in self._classes],
        )

    @property
    def spec(self):  # pragma: no cover - subclasses override
        raise NotImplementedError
