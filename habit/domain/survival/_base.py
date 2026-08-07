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
"""Shared machinery for the built-in survival models."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

from habit.exceptions import HABITAPIError
from habit.contracts.outcome import SurvivalOutcome
from habit.contracts.table import FeatureTable
from habit.domain.outcome_access import structured_survival_array
from habit.utils.estimator_utils import ComponentParamsMixin

__all__ = ["SurvivalModelBase"]


class SurvivalModelBase(ComponentParamsMixin):
    """
    Shared fit/predict bookkeeping for the built-in survival models.

    Handles the two things every implementation needs the same way: reading
    the validated survival target, and enforcing at predict time that the
    table carries exactly the feature columns seen at ``fit``. Subclasses set
    ``_spec_name`` and implement :meth:`_fit_backend`, :meth:`_risk_scores`
    and :meth:`_survival_probabilities`, which is where the lifelines /
    scikit-survival imports live (kept lazy so importing this module stays
    cheap, per the L3 layer rule).

    :class:`~habit.utils.estimator_utils.ComponentParamsMixin` adds the
    scikit-learn ``get_params``/``set_params``/``clone`` protocol, sourced
    from ``self._params`` so it cannot drift from ``spec.params``.
    """

    _spec_name: str = ""

    def __init__(self) -> None:
        self._seed: Optional[int] = None
        self._fit_columns: Tuple[str, ...] = ()
        #: The endpoint declaration captured at fit time, so predict can keep
        #: working on a feature-only table (which carries no outcome).
        self._outcome: Optional[SurvivalOutcome] = None

    def set_random_state(self, seed: int) -> None:
        """Set the random state used when the backend is built at fit time."""
        self._seed = int(seed)

    def fit(self, table: FeatureTable) -> "SurvivalModelBase":
        """Train on a table with a survival outcome."""
        target = structured_survival_array(table, owner=f"survival_model.{self._spec_name}")
        X = table.feature_matrix()
        self._outcome = table.outcome  # type: ignore[assignment]
        self._fit_columns = tuple(table.feature_columns)
        self._fit_backend(X, target)
        return self

    def _matrix_for_predict(self, table: FeatureTable) -> pd.DataFrame:
        """Return the table's feature matrix restricted to the fit schema."""
        if not self._fit_columns:
            raise HABITAPIError(
                f"survival_model.{self._spec_name} must be fitted before predict."
            )
        missing = [
            column for column in self._fit_columns
            if column not in table.feature_columns
        ]
        if missing:
            raise HABITAPIError(
                f"survival_model.{self._spec_name} was fitted on feature "
                f"columns {list(self._fit_columns)} but the table to predict "
                f"does not declare {missing} as feature columns. Apply the "
                "same upstream pipeline steps as at fit time."
            )
        return table.feature_matrix()[list(self._fit_columns)]

    def predict_risk(self, table: FeatureTable) -> pd.Series:
        """Predict per-subject risk scores (higher means shorter survival)."""
        X = self._matrix_for_predict(table)
        risk = np.asarray(self._risk_scores(X), dtype=np.float64)
        return pd.Series(risk, index=X.index, name="risk_score")

    def predict_survival_function(
        self, table: FeatureTable, times: np.ndarray
    ) -> pd.DataFrame:
        """Predict S(t|x) per subject at the requested times."""
        times = np.asarray(times, dtype=np.float64)
        if times.ndim != 1 or times.size == 0:
            raise HABITAPIError(
                "predict_survival_function expects a non-empty 1-D time grid."
            )
        if np.any(np.diff(times) < 0):
            raise HABITAPIError(
                "predict_survival_function expects ascending times."
            )
        X = self._matrix_for_predict(table)
        probability = np.asarray(self._survival_probabilities(X, times), dtype=np.float64)
        expected = (X.shape[0], times.size)
        if probability.shape != expected:
            raise HABITAPIError(
                f"survival_model.{self._spec_name} returned survival "
                f"probabilities of shape {probability.shape}; expected "
                f"{expected} (subjects x times)."
            )
        return pd.DataFrame(probability, index=X.index, columns=list(times))

    # -- Backend hooks (subclasses implement; lazy imports live here) --------

    def _fit_backend(
        self, X: pd.DataFrame, target: np.ndarray
    ) -> None:  # pragma: no cover - subclasses override
        """Fit the underlying estimator on the feature matrix and target."""
        raise NotImplementedError

    def _risk_scores(self, X: pd.DataFrame) -> np.ndarray:  # pragma: no cover
        """Return per-subject risk scores from the fitted backend."""
        raise NotImplementedError

    def _survival_probabilities(
        self, X: pd.DataFrame, times: np.ndarray
    ) -> np.ndarray:  # pragma: no cover - subclasses override
        """Return S(t|x) of shape (n_subjects, n_times)."""
        raise NotImplementedError

    @property
    def spec(self):  # pragma: no cover - subclasses override
        raise NotImplementedError
