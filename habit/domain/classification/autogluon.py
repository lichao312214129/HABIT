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
"""AutoGluon classifier (optional dependency, domain ``classifier``)."""

from __future__ import annotations

import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from habit.api.exceptions import HABITAPIError, OptionalDependencyError
from habit.contracts.table import FeatureTable
from habit.domain.classification.registry import ClassifierRegistry
from habit.domain.feature_selection._base import outcome_series
from habit.spec.specs import Spec

__all__ = ["AutogluonTabularClassifier", "AutogluonTabularClassifierParams"]


class AutogluonTabularClassifierParams(BaseModel):
    """Constructor parameters for :class:`AutogluonTabularClassifier`."""

    #: HABIT-level feature-importance mode (never forwarded to AutoGluon).
    feature_importance: str = "auto"
    #: Label column name inside the AutoGluon training frame; defaults to the
    #: table's outcome column at fit time.
    label: Optional[str] = None
    #: Keyword arguments for ``TabularPredictor(...)`` (the task definition).
    predictor: Dict[str, Any] = Field(default_factory=dict)
    #: Keyword arguments for ``TabularPredictor.fit(...)`` (the training run).
    fit: Dict[str, Any] = Field(default_factory=dict)


def _lazy_tabular_predictor() -> Any:
    """Import AutoGluon's TabularPredictor or raise a precise error."""
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as exc:
        raise OptionalDependencyError(
            "classifier.AutoGluonTabular requires the optional AutoGluon "
            "dependency; install 'HABIT[automl]' to use it."
        ) from exc
    return TabularPredictor


@ClassifierRegistry.register("AutoGluonTabular")
class AutogluonTabularClassifier:
    """
    AutoML classifier wrapping AutoGluon's ``TabularPredictor``.

    AutoGluon trains and stacks many candidate models under one ``fit``
    call; the wrapper keeps the two-part AutoGluon API (constructor =
    task definition, ``fit`` = training control) as two explicit parameter
    blocks. AutoGluon's ``fit`` accepts no random-state argument (v1.3+), so
    :meth:`set_random_state` seeds the global Python/NumPy RNGs before
    training, the same mitigation the v0.1 wrapper applied.
    """

    _spec_name = "AutoGluonTabular"

    def __init__(
        self,
        feature_importance: str = "auto",
        label: Optional[str] = None,
        predictor: Optional[Dict[str, Any]] = None,
        fit: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._feature_importance = str(feature_importance)
        self._label = label
        self._predictor_params: Dict[str, Any] = dict(predictor or {})
        self._fit_params: Dict[str, Any] = dict(fit or {})
        self._seed: Optional[int] = None
        self._predictor: Optional[Any] = None
        self._fit_columns: Tuple[str, ...] = ()
        self._classes: Optional[np.ndarray] = None

    def set_random_state(self, seed: int) -> None:
        """Set the seed applied to the global RNGs before AutoGluon fitting."""
        self._seed = int(seed)

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(
            name=self._spec_name,
            params={
                "feature_importance": self._feature_importance,
                "label": self._label,
                "predictor": dict(self._predictor_params),
                "fit": dict(self._fit_params),
            },
        )

    def fit(self, table: FeatureTable) -> "AutogluonTabularClassifier":
        """Train the AutoGluon predictor on a table with an outcome column."""
        tabular_predictor = _lazy_tabular_predictor()
        y = outcome_series(table, owner=f"classifier.{self._spec_name}")
        X = table.feature_matrix()
        label = self._label or table.outcome_column or "target"
        self._label = label
        train_data = X.reset_index(drop=True).copy()
        train_data[label] = np.asarray(y)

        predictor_params = dict(self._predictor_params)
        predictor_params["label"] = label
        self._predictor = tabular_predictor(**predictor_params)

        if self._seed is not None:
            random.seed(self._seed)
            np.random.seed(self._seed)
        self._predictor.fit(train_data=train_data, **self._fit_params)

        self._fit_columns = tuple(table.feature_columns)
        self._classes = np.asarray(self._predictor.classes_)
        return self

    def _frame_for_predict(self, table: FeatureTable) -> pd.DataFrame:
        """Return the table's feature matrix restricted to the fit schema."""
        if self._predictor is None:
            raise HABITAPIError(
                "classifier.AutoGluonTabular must be fitted before predict."
            )
        missing = [
            column for column in self._fit_columns
            if column not in table.feature_columns
        ]
        if missing:
            raise HABITAPIError(
                "classifier.AutoGluonTabular was fitted on feature columns "
                f"{list(self._fit_columns)} but the table to predict does not "
                f"declare {missing} as feature columns."
            )
        return table.feature_matrix()[list(self._fit_columns)]

    def predict(self, table: FeatureTable) -> pd.Series:
        """Predict class labels indexed by the table's identifier columns."""
        X = self._frame_for_predict(table)
        labels = self._predictor.predict(X.reset_index(drop=True))
        return pd.Series(
            np.asarray(labels),
            index=X.index,
            name=table.outcome_column or "prediction",
        )

    def predict_proba(self, table: FeatureTable) -> pd.DataFrame:
        """Predict class probabilities indexed by the identifier columns."""
        X = self._frame_for_predict(table)
        proba = self._predictor.predict_proba(X.reset_index(drop=True))
        # AutoGluon returns a DataFrame for multi-class and a Series for
        # binary positive-class probability; normalise to a full matrix.
        if isinstance(proba, pd.Series):
            proba = pd.DataFrame({str(self._classes[1]): proba})
            proba[str(self._classes[0])] = 1.0 - proba[str(self._classes[1])]
            proba = proba[[str(c) for c in self._classes]]
        return pd.DataFrame(
            np.asarray(proba),
            index=X.index,
            columns=[str(label) for label in self._classes],
        )


ClassifierRegistry.register_params_model(
    "AutoGluonTabular", AutogluonTabularClassifierParams
)
