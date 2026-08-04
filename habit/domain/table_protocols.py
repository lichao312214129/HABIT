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
"""Table-level machine-learning protocols (L3), operating on FeatureTable.

The five domain protocols (``habit.domain.protocols``) describe the habitat
imaging pipeline; the four protocols here describe the downstream tabular
machine-learning flow that consumes the resulting feature tables:

- :class:`TablePreprocessor` -- feature-table normalisation / filtering,
  sklearn-transformer semantics (``fit`` learns state, ``transform`` applies
  it). Fitted state lives on the instance so a fitted pipeline transforms
  prediction data with TRAINING statistics -- the structural answer to the
  train/predict leakage class of bugs.
- :class:`FeatureSelector` -- supervised or unsupervised column selection
  with the same fit/transform split.
- :class:`Classifier` -- outcome models. Named ``Classifier`` rather than
  ``Model`` to keep it distinct from :class:`~habit.contracts.habitat.HabitatModel`,
  the habitat-definition artefact (v1.0 naming decisions).
- :class:`Metric` -- evaluation functions with explicit probability needs.

All four take and return :class:`~habit.contracts.table.FeatureTable`, whose
explicit column roles (identifier / feature / outcome) are what make these
contracts checkable: an id column can never silently enter the model matrix.
"""

from __future__ import annotations

from typing import Optional, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd

from habit.contracts.table import FeatureTable
from habit.spec.specs import Spec

__all__ = [
    "TablePreprocessor",
    "FeatureSelector",
    "Classifier",
    "Regressor",
    "SurvivalModel",
    "Metric",
    "SurvivalMetric",
    "RegressionMetric",
]


@runtime_checkable
class TablePreprocessor(Protocol):
    """
    Learn and apply a feature-table transformation.

    sklearn transformer semantics: ``fit`` learns any statistics the
    transformation needs (means, min/max, kept columns, ...) and returns
    ``self``; ``transform`` applies them. Components are cohort-level at
    ``fit`` time (statistics cross subject boundaries) and row-parallel at
    ``transform`` time, which is exactly what a train/predict split needs.
    """

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""

    def fit(self, table: FeatureTable) -> "TablePreprocessor":
        """
        Learn the transformation state from a table.

        Args:
            table: Table whose feature columns provide the fit statistics.

        Returns:
            ``self``, fitted.
        """

    def transform(self, table: FeatureTable) -> FeatureTable:
        """
        Apply the fitted transformation.

        Args:
            table: Table to transform; must carry the feature columns seen
                at fit time.

        Returns:
            A new table with transformed feature columns and unchanged
            identifier/outcome columns.
        """


@runtime_checkable
class FeatureSelector(Protocol):
    """
    Learn and apply a feature-column subset.

    Same fit/transform split as :class:`TablePreprocessor`; ``transform``
    restricts the table to the columns selected at fit time, so prediction
    data is reduced with the TRAINING selection and never re-selected.
    """

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""

    def fit(
        self,
        table: FeatureTable,
        *,
        repeat_tables: Optional[Sequence[FeatureTable]] = None,
    ) -> "FeatureSelector":
        """
        Learn the feature subset from a table.

        Args:
            table: Table with feature columns and, for supervised selectors,
                an outcome column.
            repeat_tables: Optional repeated-measurement tables aligned to
                ``table`` by identifier columns, consumed only by
                stability-driven selectors (e.g. ICC test-retest filtering).

        Returns:
            ``self``, fitted.
        """

    def transform(self, table: FeatureTable) -> FeatureTable:
        """
        Restrict a table to the fitted feature subset.

        Args:
            table: Table carrying (at least) the selected feature columns.

        Returns:
            A new table with only the selected feature columns.
        """


@runtime_checkable
class Classifier(Protocol):
    """
    Outcome model over feature tables.

    Named ``Classifier`` (not ``Model``) so it can never be confused with
    :class:`~habit.contracts.habitat.HabitatModel`, the habitat-definition
    artefact. The feature column set is captured at ``fit`` time; ``predict``
    validates it, catching silent schema drift between training and
    prediction tables.
    """

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""

    def fit(self, table: FeatureTable) -> "Classifier":
        """
        Train on a table with an outcome column.

        Args:
            table: Training table with feature columns and a binary or
                categorical outcome column.

        Returns:
            ``self``, fitted.
        """

    def predict(self, table: FeatureTable) -> pd.Series:
        """
        Predict class labels for a table's rows.

        Args:
            table: Table carrying the feature columns seen at fit time.

        Returns:
            Predicted labels indexed by the table's identifier columns.
        """

    def predict_proba(self, table: FeatureTable) -> pd.DataFrame:
        """
        Predict class probabilities for a table's rows.

        Args:
            table: Table carrying the feature columns seen at fit time.

        Returns:
            Probability frame indexed by the identifier columns, one column
            per class.
        """


@runtime_checkable
class Regressor(Protocol):
    """
    Continuous-outcome model over feature tables.

    The sibling of :class:`Classifier` for a :class:`ContinuousOutcome`
    endpoint. Kept a distinct protocol rather than a flag on the classifier
    because evaluation is entirely different (R-squared, MAE vs AUC, and no
    probabilities), and because a model declaring the wrong family fails at
    ``fit`` -- where the endpoint type is checked -- rather than at scoring.
    """

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""

    def fit(self, table: FeatureTable) -> "Regressor":
        """
        Train on a table with a continuous outcome column.

        Args:
            table: Training table with feature columns and a continuous
                outcome column.

        Returns:
            ``self``, fitted.
        """

    def predict(self, table: FeatureTable) -> pd.Series:
        """
        Predict the continuous response for a table's rows.

        Args:
            table: Table carrying the feature columns seen at fit time.

        Returns:
            Predicted values indexed by the table's identifier columns.
        """


@runtime_checkable
class SurvivalModel(Protocol):
    """
    Right-censored time-to-event model over feature tables.

    Two predict methods are separated because they serve disjoint purposes.
    ``predict_risk`` answers "how does this patient rank against the others?"
    with a single score per subject -- higher means shorter expected survival
    -- and is all the C-index needs. ``predict_survival_function`` answers
    "what is the probability of surviving past time t?" per subject and per
    time point, which is what time-dependent AUC, the integrated Brier score,
    fixed-horizon calibration and the KM-style plot of predicted curves all
    require. A model may leave the latter unimplemented; evaluation drivers
    detect that and skip the function-based metrics and plots rather than
    failing.
    """

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""

    def fit(self, table: FeatureTable) -> "SurvivalModel":
        """
        Train on a table with a survival outcome.

        Args:
            table: Training table with feature columns and a
                :class:`~habit.contracts.outcome.SurvivalOutcome`.

        Returns:
            ``self``, fitted.
        """

    def predict_risk(self, table: FeatureTable) -> pd.Series:
        """
        Predict a per-subject risk score (higher means worse prognosis).

        Args:
            table: Table carrying the feature columns seen at fit time.

        Returns:
            Risk scores indexed by the table's identifier columns.
        """

    def predict_survival_function(
        self, table: FeatureTable, times: np.ndarray
    ) -> pd.DataFrame:
        """
        Predict S(t | x) for each subject at the requested times.

        Args:
            table: Table carrying the feature columns seen at fit time.
            times: One-dimensional grid of evaluation times (ascending), in
                the same units the outcome declares.

        Returns:
            Survival probabilities, one row per subject (indexed by the
            table's identifier columns) and one column per requested time
            (named by the time value).

        Raises:
            NotImplementedError: When the model only supports risk ranking.
        """


@runtime_checkable
class Metric(Protocol):
    """
    Evaluation metric with explicit input requirements.

    The ``needs_proba`` flag declares whether the metric consumes class
    probabilities/scores (AUC, calibration tests) or hard labels (accuracy,
    sensitivity); evaluation drivers use it instead of guessing from the
    metric name.
    """

    needs_proba: bool
    greater_is_better: bool

    @property
    def spec(self) -> Spec:
        """Return the metric specification."""

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute the metric value.

        Args:
            y_true: True class labels.
            y_pred: Predicted class labels.
            y_score: Probability/score of the positive class; required when
                ``needs_proba`` is true.

        Returns:
            The metric value (``NaN`` where the metric is undefined for the
            given inputs, e.g. calibration tests on multi-class problems).
        """


@runtime_checkable
class SurvivalMetric(Protocol):
    """
    Evaluation metric for a right-censored survival endpoint.

    Kept separate from :class:`Metric` because the inputs are categorically
    different: a survival metric consumes the follow-up time and the event
    indicator as GROUND TRUTH plus a per-subject prediction, whereas
    :class:`Metric` consumes class labels. Folding the two into one
    ``__call__`` would force a union of unrelated argument shapes and make
    "which inputs does this metric need?" unanswerable.

    Two prediction forms exist and the metric declares which it consumes via
    ``needs_survival_function``: risk-based metrics (C-index) need only the
    per-subject score, while function-based metrics (integrated Brier score,
    time-dependent AUC) need S(t|x) evaluated on a time grid.
    """

    #: True when the metric needs survival probabilities, False for risk-only.
    needs_survival_function: bool
    greater_is_better: bool

    @property
    def spec(self) -> Spec:
        """Return the metric specification."""

    def __call__(
        self,
        time: np.ndarray,
        event: np.ndarray,
        prediction: np.ndarray,
        times: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute the metric value.

        Args:
            time: Observed follow-up durations (float).
            event: Event indicators, True for observed events.
            prediction: Per-subject risk scores for risk-based metrics, or an
                ``(n_subjects, n_times)`` survival-probability matrix for
                function-based ones.
            times: The time points the columns of a function-based
                ``prediction`` were evaluated at. Required when
                ``needs_survival_function`` is true, because the probability
                columns are meaningless without their evaluation times.

        Returns:
            The metric value.
        """


@runtime_checkable
class RegressionMetric(Protocol):
    """
    Evaluation metric for a continuous endpoint.

    Separate from :class:`Metric` for the same reason as
    :class:`SurvivalMetric`: the ground truth is a quantity, not a class.
    """

    greater_is_better: bool

    @property
    def spec(self) -> Spec:
        """Return the metric specification."""

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """
        Compute the metric value.

        Args:
            y_true: True continuous responses.
            y_pred: Predicted continuous responses.

        Returns:
            The metric value.
        """
