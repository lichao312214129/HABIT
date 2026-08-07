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
"""Table-level scikit-learn interop adapters (L3).

HABIT's tabular components speak :class:`~habit.contracts.table.FeatureTable`
in and out: the outcome rides inside the table (which is what makes the
train/predict split leakage-safe), selectors drop COLUMNS rather than
producing an anonymous matrix, and probabilities come back as a frame keyed
by class label. scikit-learn's estimator API speaks ``(X, y)`` arrays and
returns bare ``ndarray``. This module is the single bridge between the two
vocabularies, so neither has to bend:

* :class:`FrameToTable` turns a plain frame back into a ``FeatureTable``
  using a STATIC column schema, which is what lets sklearn's
  cross-validation drivers slice ``X`` by row (a ``FeatureTable`` is a frozen
  dataclass and deliberately not row-indexable).
* :class:`TableTransformerEstimator` wraps a
  :class:`~habit.domain.table_protocols.TablePreprocessor` or
  :class:`~habit.domain.table_protocols.FeatureSelector`.
* :class:`TableClassifierEstimator`, :class:`TableRegressorEstimator` and
  :class:`TableSurvivalEstimator` wrap the three terminal outcome-model
  families, each carrying the sklearn mixin its family needs so that
  ``is_classifier`` / ``is_regressor`` (and therefore ``GridSearchCV``'s
  choice of splitter and default scorer) answer correctly.

The adapters live here, at L3, rather than in ``habit.compat`` because
:class:`~habit.domain.pipeline.TablePipeline` -- itself an
``sklearn.pipeline.Pipeline`` subclass -- is built out of them, and
``habit.compat`` is a frozen compatibility surface that must not grow new
capability. ``habit.compat.sklearn`` keeps thin deprecated aliases for the
two adapter names and the two factory functions it used to own; they stay
importable for all of v1.x.

They cannot live in ``habit.adapters`` (L1) either: they depend on
``habit.domain.table_protocols`` (L3), and the layering forbids an upward
import.

Typical use::

    from sklearn.model_selection import GridSearchCV

    from habit.domain.sklearn_interop import as_classifier, as_transformer

    pipe = Pipeline([
        ("scale", as_transformer(ZScorePreprocessor())),
        ("select", as_transformer(AnovaSelector(n_features_to_select=20))),
        ("model", as_classifier(LogisticRegressionClassifier())),
    ])
    pipe.fit(train_table)          # outcome rides inside the FeatureTable
    pipe.predict_proba(holdout_table)

sklearn imports are module-level here because every user of this module is
already inside an sklearn workflow; the surrounding domain packages keep
their lazy-import discipline.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin
from sklearn.exceptions import NotFittedError

from habit.exceptions import HABITAPIError
from habit.contracts.outcome import BinaryOutcome, MulticlassOutcome, Outcome
from habit.contracts.table import FeatureTable
from habit.domain.outcome_access import outcome_series
from habit.domain.table_protocols import (
    Classifier,
    FeatureSelector,
    Regressor,
    SurvivalModel,
    TablePreprocessor,
)
from habit.utils.log_utils import get_module_logger

__all__ = [
    "FrameToTable",
    "TableTransformerEstimator",
    "TableClassifierEstimator",
    "TableRegressorEstimator",
    "TableSurvivalEstimator",
    "as_transformer",
    "as_classifier",
    "as_regressor",
    "as_survival_model",
    "as_outcome_model",
    "step_consumes_repeat_tables",
    "wraps_outcome_model",
]

_logger = get_module_logger(__name__)

#: Width of the separator line closing one selector's log block. Kept equal to
#: the v0.1 / pre-refactor ``TablePipeline.fit`` banner so log files parsed by
#: existing tooling do not change shape.
_LOG_RULE_WIDTH = 80


def _require_table(X: Any) -> FeatureTable:
    """
    Validate that a pipeline step received a ``FeatureTable``.

    Args:
        X: The object an sklearn driver handed to this step.

    Returns:
        ``X`` itself, typed as a ``FeatureTable``.

    Raises:
        HABITAPIError: When ``X`` is anything else. The message names the
            :class:`FrameToTable` head step, because a bare frame reaching a
            table adapter almost always means that step is missing.
    """
    if not isinstance(X, FeatureTable):
        raise HABITAPIError(
            "Table adapters operate on habit FeatureTable objects; got "
            f"{type(X).__name__}. Chain them with other table-aware steps, or "
            "put a FrameToTable step at the head of the pipeline so a plain "
            "frame is rebuilt into a FeatureTable first."
        )
    return X


def step_consumes_repeat_tables(component: Any) -> bool:
    """
    Report whether a HABIT component's ``fit`` accepts ``repeat_tables``.

    Test-retest selectors (the ICC family) learn from aligned repeat
    measurement tables in addition to the primary training table. Everything
    else must NOT be handed the keyword, so the pipeline routes it only to
    the steps that declare it.

    Args:
        component: A ``TablePreprocessor`` / ``FeatureSelector`` instance.

    Returns:
        ``True`` when ``component.fit`` declares a ``repeat_tables``
        parameter, ``False`` when it does not (or has no introspectable
        signature).
    """
    try:
        signature = inspect.signature(component.fit)
    except (AttributeError, TypeError, ValueError):
        return False
    return "repeat_tables" in signature.parameters


# ---------------------------------------------------------------------------
# FrameToTable: the row-indexable entry point for sklearn CV drivers
# ---------------------------------------------------------------------------


class FrameToTable(TransformerMixin, BaseEstimator):
    """
    Rebuild a ``FeatureTable`` from a plain frame plus a static column schema.

    This is the step that makes HABIT's tabular pipelines usable with
    ``cross_val_score`` / ``GridSearchCV``. Those drivers slice ``X`` by row,
    and a :class:`~habit.contracts.table.FeatureTable` is a frozen dataclass
    with column semantics, not a row-indexable container -- passing one as
    ``X`` fails inside sklearn's own input validation. A plain
    ``DataFrame`` carrying the identifier columns, the feature columns and
    the outcome column(s) IS row-indexable, so the frame becomes ``X`` and
    this step restores the table contract at the head of the pipeline.

    The schema (which columns identify a row, and which endpoint the study
    has) is METADATA, not data: it does not change when rows are resampled,
    so it belongs on the estimator as a constructor parameter and travels
    through ``sklearn.base.clone`` untouched. Feature columns are everything
    else, so a selector dropping columns simply shrinks the next fold's
    feature set with no schema bookkeeping.

    A ``FeatureTable`` passed straight through (HABIT's own
    ``TablePipeline.fit(table)`` entry point) is returned unchanged: no frame
    round-trip, no dtype promotion, no column reordering. That is a numerical
    requirement, not an optimisation -- rebuilding a float32 table through
    ``DataFrame`` construction can shift the cohort statistics a later
    z-score learns.

    Args:
        id_columns: Columns identifying a row (e.g. ``("subject",)``). They
            are excluded from the feature block.
        outcome: The endpoint declaration. All four families are supported;
            a :class:`~habit.contracts.outcome.SurvivalOutcome` occupies two
            columns (time and event), and every column it names is excluded
            from the feature block. ``None`` builds an unlabelled table,
            which is what a pure inference pipeline needs.
        feature_columns: Explicit feature block. ``None`` (the default)
            derives it as "every column that is neither an identifier nor
            part of the outcome", which is what keeps the schema stable
            while upstream selectors shrink the frame.
    """

    def __init__(
        self,
        id_columns: Sequence[str] = (),
        outcome: Optional[Outcome] = None,
        feature_columns: Optional[Sequence[str]] = None,
    ) -> None:
        self.id_columns = id_columns
        self.outcome = outcome
        self.feature_columns = feature_columns

    @classmethod
    def from_table(cls, table: FeatureTable) -> "FrameToTable":
        """
        Read the column schema off an existing table.

        The convenience constructor for the sklearn cross-validation path:
        the schema a study's frames follow is the schema of its training
        table, so declaring it twice by hand is an opportunity to declare it
        differently.

        Args:
            table: Table whose column semantics later frames will follow.
                Only the schema is read; no row of data is retained.

        Returns:
            A transformer that rebuilds tables with ``table``'s identifier
            columns and endpoint. ``feature_columns`` is deliberately left
            derived rather than pinned, so an upstream selector shrinking
            the frame shrinks the feature block with it.
        """
        return cls(id_columns=tuple(table.id_columns), outcome=table.outcome)

    @property
    def declares_schema(self) -> bool:
        """
        Report whether any column schema was declared at all.

        A transformer with no identifier columns, no endpoint and no explicit
        feature block cannot turn a frame into a meaningful table -- every
        column, identifiers included, would become a feature. Callers use
        this to raise a precise error instead of silently modelling on an
        identifier column.
        """
        return bool(self.id_columns) or self.outcome is not None or bool(
            self.feature_columns
        )

    def fit(self, X: Any, y: Any = None) -> "FrameToTable":
        """
        Do nothing; the schema is declared, never learned.

        Args:
            X: Training frame or ``FeatureTable``; unused.
            y: Ignored, accepted for sklearn compatibility.

        Returns:
            ``self``.
        """
        return self

    def transform(self, X: Any) -> FeatureTable:
        """
        Return ``X`` as a ``FeatureTable``.

        Args:
            X: A ``FeatureTable`` (returned unchanged) or a ``DataFrame``
                holding the identifier, feature and outcome columns.

        Returns:
            The table to hand to the next step.

        Raises:
            HABITAPIError: When ``X`` is neither, or when a declared schema
                column is missing from the frame.
        """
        if isinstance(X, FeatureTable):
            return X
        if not isinstance(X, pd.DataFrame):
            raise HABITAPIError(
                "FrameToTable needs a pandas DataFrame (or a FeatureTable to "
                f"pass through); got {type(X).__name__}."
            )
        if not self.declares_schema:
            raise HABITAPIError(
                "FrameToTable was handed a plain frame but declares no column "
                "schema, so every column -- identifiers included -- would "
                "become a feature and the endpoint would be unknown. Declare "
                "it, e.g. FrameToTable.from_table(training_table)."
            )
        id_columns = tuple(str(column) for column in self.id_columns)
        reserved = set(id_columns)
        if self.outcome is not None:
            reserved.update(str(column) for column in self.outcome.columns)
        missing = [column for column in sorted(reserved) if column not in X.columns]
        if missing:
            raise HABITAPIError(
                f"FrameToTable declares schema columns {missing} that the "
                f"frame does not carry (frame columns: {list(X.columns)[:10]}). "
                "Pass the frame the schema was declared for, or update the "
                "FrameToTable step's id_columns / outcome."
            )
        if self.feature_columns is None:
            features = tuple(
                str(column) for column in X.columns if str(column) not in reserved
            )
        else:
            features = tuple(str(column) for column in self.feature_columns)
            absent = [column for column in features if column not in X.columns]
            if absent:
                raise HABITAPIError(
                    f"FrameToTable declares feature columns {absent} that the "
                    "frame does not carry."
                )
        return FeatureTable(
            frame=X.reset_index(drop=True),
            id_columns=id_columns,
            feature_columns=features,
            outcome=self.outcome,
        )


# ---------------------------------------------------------------------------
# TableTransformerEstimator: preprocessors and selectors
# ---------------------------------------------------------------------------


class TableTransformerEstimator(TransformerMixin, BaseEstimator):
    """
    Adapt a HABIT table transformation to the sklearn transformer API.

    ``X`` is a :class:`~habit.contracts.table.FeatureTable` in and out, so the
    adapter composes with other table-aware steps inside an sklearn
    ``Pipeline`` while the fitted state (training statistics, selected
    columns) follows sklearn's clone/fit lifecycle.

    Args:
        component: A :class:`~habit.domain.table_protocols.TablePreprocessor`
            or :class:`~habit.domain.table_protocols.FeatureSelector`
            implementation.
        copy_on_fit: When ``True`` (the default, and the behaviour every
            standalone user of :func:`as_transformer` has always had), the
            wrapped component is deep-copied at ``fit`` time so the instance
            the caller passed is never mutated. When ``False`` the component
            is fitted IN PLACE, which is what
            :class:`~habit.domain.pipeline.TablePipeline` needs: its
            ``components`` property, its ``save()`` artefact and every
            reporting call site read fitted state off the very objects the
            pipeline was constructed with. ``sklearn.base.clone`` already
            gives each cross-validation fold its own component, so in-place
            fitting inside a pipeline cannot leak across folds.
        selector_step_index: 1-based position of this step among the
            pipeline's FEATURE SELECTORS, or ``None`` to log nothing. The
            per-step "features before / after / removed" report used to live
            in ``TablePipeline.fit``; ``sklearn.pipeline.Pipeline._fit`` has
            no hook there, so the report moved here, where the step itself
            knows its own numbers. ``None`` for preprocessors keeps the log
            identical to the pre-refactor output, which only ever reported
            selectors.

    Attributes:
        component_: The fitted component (the deep copy when ``copy_on_fit``
            is set, otherwise ``component`` itself).
    """

    def __init__(
        self,
        component: Any,
        copy_on_fit: bool = True,
        selector_step_index: Optional[int] = None,
    ) -> None:
        self.component = component
        self.copy_on_fit = copy_on_fit
        self.selector_step_index = selector_step_index

    def fit(
        self, X: Any, y: Any = None, **fit_params: Any
    ) -> "TableTransformerEstimator":
        """
        Fit the wrapped component on a training table.

        Args:
            X: Training ``FeatureTable``.
            y: Ignored (supervised selectors read the table's outcome
                column); accepted for sklearn compatibility.
            **fit_params: Forwarded to the component's ``fit`` (e.g.
                ``repeat_tables=`` for ICC-driven selectors).

        Returns:
            ``self``, fitted.

        Raises:
            HABITAPIError: When ``X`` is not a ``FeatureTable`` or the
                wrapped object implements neither table protocol.
        """
        table = _require_table(X)
        if not isinstance(self.component, (TablePreprocessor, FeatureSelector)):
            raise HABITAPIError(
                "TableTransformerEstimator wraps a TablePreprocessor or "
                f"FeatureSelector; got {type(self.component).__name__}."
            )
        self._log_before(table)
        self.component_ = (
            copy.deepcopy(self.component) if self.copy_on_fit else self.component
        )
        self.component_.fit(table, **fit_params)
        return self

    def fit_transform(self, X: Any, y: Any = None, **fit_params: Any) -> FeatureTable:
        """
        Fit on a table and return the transformed table in one pass.

        This is what ``sklearn.pipeline.Pipeline`` calls on intermediate
        steps, and the only place both the pre-step and post-step feature
        counts are known -- so it is where the selector report is closed.

        Args:
            X: Training ``FeatureTable``.
            y: Ignored; accepted for sklearn compatibility.
            **fit_params: Forwarded to the component's ``fit``.

        Returns:
            The transformed ``FeatureTable``.
        """
        table = _require_table(X)
        n_before = len(table.feature_columns)
        self.fit(X, y, **fit_params)
        transformed = self.component_.transform(table)
        self._log_after(n_before, len(transformed.feature_columns))
        return transformed

    def transform(self, X: Any) -> FeatureTable:
        """
        Apply the fitted transformation.

        Args:
            X: Table carrying the fit-time feature columns.

        Returns:
            The transformed ``FeatureTable``.

        Raises:
            NotFittedError: If called before :meth:`fit`.
        """
        if not hasattr(self, "component_"):
            raise NotFittedError(
                "This TableTransformerEstimator is not fitted yet; call fit first."
            )
        return self.component_.transform(_require_table(X))

    # -- logging ---------------------------------------------------------

    def _log_before(self, table: FeatureTable) -> None:
        """Log the opening lines of one selector's report."""
        if self.selector_step_index is None:
            return
        spec = self.component.spec
        _logger.info(
            "Step %s: Applying '%s' feature selection",
            self.selector_step_index,
            str(spec.name),
        )
        _logger.info("  Parameters: %s", dict(spec.params))
        _logger.info("  Features before this step: %s", len(table.feature_columns))

    def _log_after(self, n_before: int, n_after: int) -> None:
        """Log the closing lines of one selector's report."""
        if self.selector_step_index is None:
            return
        _logger.info("  Features after this step: %s", n_after)
        _logger.info("  Number of features removed: %s", n_before - n_after)
        _logger.info("-" * _LOG_RULE_WIDTH)


# ---------------------------------------------------------------------------
# Terminal outcome models: classifier / regressor / survival
# ---------------------------------------------------------------------------


class _TableModelEstimatorBase(BaseEstimator):
    """
    Shared plumbing for the three terminal outcome-model adapters.

    The outcome rides inside the ``FeatureTable`` (HABIT's leakage-safe
    convention). A separate ``y`` is therefore only ever used to FILL a
    missing outcome column, and is cross-checked against an existing one so
    a misaligned ``y`` fails loudly instead of silently training against the
    wrong targets.

    Args:
        component: The HABIT outcome model to wrap.
        copy_on_fit: See :class:`TableTransformerEstimator`.
    """

    #: Protocol the wrapped component must satisfy; set by each subclass.
    _protocol: Any = None
    #: Human-readable family name used in error messages.
    _family: str = "outcome model"

    def __init__(self, component: Any, copy_on_fit: bool = True) -> None:
        self.component = component
        self.copy_on_fit = copy_on_fit

    def _prepared_table(self, X: Any, y: Any) -> FeatureTable:
        """
        Validate the component and return the training table to fit on.

        Args:
            X: Training ``FeatureTable``.
            y: Optional targets, used only when the table declares no
                outcome.

        Returns:
            The table the component will be fitted on.

        Raises:
            HABITAPIError: On a wrong component family, a missing outcome
                with no ``y``, or a ``y``/outcome disagreement.
        """
        table = _require_table(X)
        if not isinstance(self.component, self._protocol):
            raise HABITAPIError(
                f"{type(self).__name__} wraps a HABIT {self._family}; got "
                f"{type(self.component).__name__}."
            )
        if table.outcome is not None and y is not None:
            declared = outcome_series(
                table, owner=f"{type(self).__name__}.fit"
            ).to_numpy()
            if not np.array_equal(np.asarray(y), declared):
                raise HABITAPIError(
                    "y disagrees with the table's outcome column; refusing "
                    "to train on ambiguous labels."
                )
        if table.outcome is None:
            if y is None:
                raise HABITAPIError(
                    "Training table has no outcome column and no y was given."
                )
            table = _attach_outcome(table, y)
        return table

    def _fitted_component(self) -> Any:
        """Return the fitted component, or raise sklearn's not-fitted error."""
        if not hasattr(self, "component_"):
            raise NotFittedError(
                f"This {type(self).__name__} is not fitted yet; call fit first."
            )
        return self.component_

    def _bind(self, table: FeatureTable) -> Any:
        """Copy (or adopt) the component, fit it, and return it."""
        self.component_ = (
            copy.deepcopy(self.component) if self.copy_on_fit else self.component
        )
        self.component_.fit(table)
        # The feature block the terminal model was actually trained on. It is
        # the output of the whole transformation chain, and the only place it
        # is observable after the fact, so the pipeline reads its
        # ``fit_output_columns`` (recorded in the ``.habitpipeline`` manifest)
        # from here rather than re-running the chain.
        self.feature_columns_ = tuple(table.feature_columns)
        return self.component_


class TableClassifierEstimator(ClassifierMixin, _TableModelEstimatorBase):
    """
    Adapt a HABIT :class:`~habit.domain.table_protocols.Classifier` to sklearn.

    ``ClassifierMixin`` is what makes ``sklearn.base.is_classifier`` answer
    ``True``, which in turn is what makes ``GridSearchCV`` pick a stratified
    splitter and the accuracy scorer by default.

    Args:
        component: A HABIT ``Classifier`` implementation.
        copy_on_fit: See :class:`TableTransformerEstimator`.

    Attributes:
        component_: The fitted classifier.
        classes_: Class labels in the fitted classifier's own order. HABIT
            classifiers label their probability FRAME columns with
            ``str(label)``, but sklearn's scorers compare ``classes_``
            against the ``y`` they were handed, so the labels are recovered
            in the endpoint's own dtype whenever the string round-trip is
            unambiguous (it is for every integer / string endpoint). Without
            that, ``scoring="roc_auc"`` on an integer 0/1 endpoint would fail
            to match ``classes_`` and report an error rather than a score.
        proba_columns_: The probability frame's column labels, in the fitted
            classifier's order. Column alignment always goes through these,
            so ``classes_`` never has to be the thing that indexes a frame.
    """

    _protocol = Classifier
    _family = "Classifier"

    def fit(self, X: Any, y: Any = None) -> "TableClassifierEstimator":
        """
        Train the wrapped classifier.

        Args:
            X: Training ``FeatureTable`` with an outcome column (or pass
                ``y`` to attach one).
            y: Optional outcome values. When the table already carries an
                outcome, ``y`` must agree with it exactly.

        Returns:
            ``self``, fitted.
        """
        table = self._prepared_table(X, y)
        component = self._bind(table)
        # Capturing the labels from the FITTED classifier (via a one-row
        # probe) guarantees predict_proba stays column-aligned without
        # reaching into private state.
        probe = dataclasses.replace(table, frame=table.frame.head(1))
        self.proba_columns_ = tuple(component.predict_proba(probe).columns)
        self.classes_ = _native_class_labels(table, self.proba_columns_)
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels for a table's rows."""
        return self._fitted_component().predict(_require_table(X)).to_numpy()

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities, columns aligned to ``classes_``."""
        proba = self._fitted_component().predict_proba(_require_table(X))
        aligned = proba.reindex(columns=list(self.proba_columns_))
        if aligned.isna().any().any():
            raise HABITAPIError(
                "Classifier probability columns do not cover the classes "
                "seen at fit time."
            )
        return aligned.to_numpy(dtype=float)

    def score(self, X: Any, y: Any = None, sample_weight: Any = None) -> float:
        """
        Return accuracy on a table.

        Differs from ``ClassifierMixin.score`` in one deliberate way: when
        ``y`` is omitted the table's own outcome column supplies the truth,
        which is the natural call inside a FeatureTable-carrying pipeline.

        Args:
            X: ``FeatureTable`` to score.
            y: True labels; falls back to the table's outcome column.
            sample_weight: Optional per-row weights.

        Returns:
            Mean accuracy.
        """
        from sklearn.metrics import accuracy_score

        if y is None:
            y = _truth_from_table(X, owner=f"{type(self).__name__}.score")
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


class TableRegressorEstimator(RegressorMixin, _TableModelEstimatorBase):
    """
    Adapt a HABIT :class:`~habit.domain.table_protocols.Regressor` to sklearn.

    ``RegressorMixin`` makes ``sklearn.base.is_regressor`` answer ``True``,
    so ``GridSearchCV`` picks a plain K-fold splitter and the R-squared
    scorer instead of stratifying on a continuous endpoint.

    Args:
        component: A HABIT ``Regressor`` implementation.
        copy_on_fit: See :class:`TableTransformerEstimator`.
    """

    _protocol = Regressor
    _family = "Regressor"

    def fit(self, X: Any, y: Any = None) -> "TableRegressorEstimator":
        """
        Train the wrapped regressor.

        Args:
            X: Training ``FeatureTable`` with a continuous outcome column.
            y: Optional outcome values; cross-checked against the table.

        Returns:
            ``self``, fitted.
        """
        self._bind(self._prepared_table(X, y))
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict the continuous outcome for a table's rows."""
        return self._fitted_component().predict(_require_table(X)).to_numpy()

    def score(self, X: Any, y: Any = None, sample_weight: Any = None) -> float:
        """
        Return the coefficient of determination on a table.

        Args:
            X: ``FeatureTable`` to score.
            y: True values; falls back to the table's outcome column.
            sample_weight: Optional per-row weights.

        Returns:
            R-squared.
        """
        from sklearn.metrics import r2_score

        if y is None:
            y = _truth_from_table(X, owner=f"{type(self).__name__}.score")
        return r2_score(y, self.predict(X), sample_weight=sample_weight)


class TableSurvivalEstimator(_TableModelEstimatorBase):
    """
    Adapt a HABIT :class:`~habit.domain.table_protocols.SurvivalModel`.

    Deliberately carries NO sklearn mixin: a right-censored endpoint is
    neither a classification nor a regression target, so claiming either
    family would make ``GridSearchCV`` stratify on the wrong thing and score
    with a metric that cannot see censoring. ``predict`` returns the risk
    score, which is what a concordance-style scorer needs;
    ``predict_survival_function`` exposes the per-time-point curves the
    integrated Brier score and time-dependent AUC require.

    Args:
        component: A HABIT ``SurvivalModel`` implementation.
        copy_on_fit: See :class:`TableTransformerEstimator`.
    """

    _protocol = SurvivalModel
    _family = "SurvivalModel"

    def fit(self, X: Any, y: Any = None) -> "TableSurvivalEstimator":
        """
        Train the wrapped survival model.

        Args:
            X: Training ``FeatureTable`` with a survival outcome (time and
                event columns).
            y: Ignored. A survival endpoint spans two columns, so it can
                only be declared on the table; accepted for sklearn
                compatibility.

        Returns:
            ``self``, fitted.
        """
        table = _require_table(X)
        if not isinstance(self.component, SurvivalModel):
            raise HABITAPIError(
                f"{type(self).__name__} wraps a HABIT SurvivalModel; got "
                f"{type(self.component).__name__}."
            )
        if table.outcome is None:
            raise HABITAPIError(
                "A survival model needs a table declaring its time and event "
                "columns; this table declares no outcome."
            )
        self._bind(table)
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict the per-row risk score (higher means shorter survival)."""
        return self._fitted_component().predict_risk(_require_table(X)).to_numpy()

    def predict_survival_function(
        self, X: Any, times: np.ndarray
    ) -> pd.DataFrame:
        """
        Predict per-row survival probabilities on a time grid.

        Args:
            X: ``FeatureTable`` to predict.
            times: Ascending 1-D grid of evaluation times.

        Returns:
            Survival probabilities, one row per subject, one column per time.
        """
        return self._fitted_component().predict_survival_function(
            _require_table(X), times
        )


def _native_class_labels(
    table: FeatureTable, proba_columns: Sequence[Any]
) -> np.ndarray:
    """
    Recover the class labels in the endpoint's own dtype.

    A HABIT classifier's probability frame is keyed by ``str(label)``, which
    is what makes the frame readable but loses the endpoint's dtype. sklearn
    compares ``estimator.classes_`` against the ``y`` a scorer was handed, so
    an integer 0/1 endpoint reported as ``["0", "1"]`` breaks
    ``scoring="roc_auc"`` (and every other label-aware scorer) with a
    confusing "labels not in classes_" error.

    Args:
        table: The training table, whose outcome column carries the labels in
            their native dtype.
        proba_columns: The probability frame's column labels, in the fitted
            classifier's own order.

    Returns:
        np.ndarray: The labels in native dtype, ordered to match
        ``proba_columns``, when the ``str()`` round-trip is unambiguous;
        otherwise ``proba_columns`` as-is, since a wrong dtype is better than
        a wrong ORDER (order decides which column is the positive class).
    """
    columns = [str(column) for column in proba_columns]
    if table.outcome is None:
        return np.asarray(columns)
    try:
        observed = np.unique(
            outcome_series(table, owner="TableClassifierEstimator.fit").to_numpy()
        )
    except Exception:  # pragma: no cover - non-label endpoints
        return np.asarray(columns)
    by_text = {str(label): label for label in observed}
    if len(by_text) != len(observed) or not set(columns) <= set(by_text):
        # Either two distinct labels share a string form, or the fitted
        # classifier knows a class the training rows never showed. Neither is
        # safe to re-map, so keep the classifier's own labels.
        return np.asarray(columns)
    return np.asarray([by_text[column] for column in columns])


def _truth_from_table(X: Any, *, owner: str) -> np.ndarray:
    """
    Read the ground truth out of a table for a ``score`` call.

    Args:
        X: The ``FeatureTable`` being scored.
        owner: Dotted name used in the error message.

    Returns:
        The outcome values as an array.

    Raises:
        HABITAPIError: When the table declares no outcome.
    """
    table = _require_table(X)
    if table.outcome is None:
        raise HABITAPIError(
            "score needs y or a table carrying an outcome column."
        )
    return outcome_series(table, owner=owner).to_numpy()


def _attach_outcome(table: FeatureTable, y: Any) -> FeatureTable:
    """
    Return a copy of ``table`` with ``y`` attached as its outcome column.

    Args:
        table: Table without an outcome.
        y: Target values, one per row.

    Returns:
        A table declaring the freshly attached outcome.

    Raises:
        HABITAPIError: On a length mismatch.
    """
    values = np.asarray(y)
    if values.shape[0] != len(table.frame):
        raise HABITAPIError(
            f"y has {values.shape[0]} entries but the table has "
            f"{len(table.frame)} rows."
        )
    column = "outcome"
    while column in table.frame.columns:
        column = f"habit_{column}"
    frame = table.frame.copy()
    frame[column] = values
    # sklearn passes bare labels, so the endpoint family is inferred from
    # them; the positive class follows sklearn's own convention that the
    # greater of two labels is positive.
    labels = np.unique(values)
    outcome: Outcome
    if labels.size <= 2:
        outcome = BinaryOutcome(column, positive_label=labels[-1])
    else:
        outcome = MulticlassOutcome(column, classes=tuple(labels))
    return FeatureTable(
        frame=frame,
        id_columns=table.id_columns,
        feature_columns=table.feature_columns,
        outcome=outcome,
        provenance=table.provenance,
    )


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def as_transformer(component: Any, **options: Any) -> TableTransformerEstimator:
    """
    Wrap a HABIT table preprocessor/selector as a sklearn transformer.

    Args:
        component: ``TablePreprocessor`` or ``FeatureSelector`` implementation.
        **options: Forwarded to :class:`TableTransformerEstimator`
            (``copy_on_fit`` / ``selector_step_index``).

    Returns:
        The configured adapter.
    """
    return TableTransformerEstimator(component, **options)


def as_classifier(component: Any, **options: Any) -> TableClassifierEstimator:
    """
    Wrap a HABIT classifier as a sklearn classifier.

    Args:
        component: ``Classifier`` implementation.
        **options: Forwarded to :class:`TableClassifierEstimator`
            (``copy_on_fit``).

    Returns:
        The configured adapter.
    """
    return TableClassifierEstimator(component, **options)


def as_regressor(component: Any, **options: Any) -> TableRegressorEstimator:
    """
    Wrap a HABIT regressor as a sklearn regressor.

    Args:
        component: ``Regressor`` implementation.
        **options: Forwarded to :class:`TableRegressorEstimator`
            (``copy_on_fit``).

    Returns:
        The configured adapter.
    """
    return TableRegressorEstimator(component, **options)


def as_survival_model(component: Any, **options: Any) -> TableSurvivalEstimator:
    """
    Wrap a HABIT survival model as an sklearn-compatible estimator.

    Args:
        component: ``SurvivalModel`` implementation.
        **options: Forwarded to :class:`TableSurvivalEstimator`
            (``copy_on_fit``).

    Returns:
        The configured adapter.
    """
    return TableSurvivalEstimator(component, **options)


def wraps_outcome_model(estimator: Any) -> bool:
    """
    Report whether ``estimator`` is one of the terminal outcome-model adapters.

    The single place that answers "is this pipeline step the terminal model?".
    A pipeline needs it in two directions -- to EXCLUDE the model when
    listing transformation components, and to find it when reporting
    ``pipeline.model`` -- and answering it in two places would eventually
    answer it differently.

    Args:
        estimator: Any pipeline step.

    Returns:
        bool: ``True`` for a :class:`TableClassifierEstimator`,
        :class:`TableRegressorEstimator` or :class:`TableSurvivalEstimator`.
    """
    return isinstance(estimator, _TableModelEstimatorBase)


def as_outcome_model(
    component: Any, **options: Any
) -> Union[
    TableClassifierEstimator, TableRegressorEstimator, TableSurvivalEstimator
]:
    """
    Wrap ANY terminal outcome model in the adapter matching its family.

    One dispatcher rather than three call sites, because the family a model
    belongs to is a property of the model, and a pipeline builder that had
    to decide would eventually decide differently from the evaluator.
    Survival is checked first: a survival model is the only family with two
    predict verbs, and the protocols are structural, so a broader match
    could otherwise win.

    Args:
        component: A ``Classifier`` / ``Regressor`` / ``SurvivalModel``.
        **options: Forwarded to the chosen adapter (``copy_on_fit``).

    Returns:
        The configured adapter.

    Raises:
        HABITAPIError: When ``component`` implements none of the three
            terminal-model protocols.
    """
    if isinstance(component, SurvivalModel):
        return TableSurvivalEstimator(component, **options)
    if isinstance(component, Classifier):
        return TableClassifierEstimator(component, **options)
    if isinstance(component, Regressor):
        return TableRegressorEstimator(component, **options)
    raise HABITAPIError(
        "A terminal outcome model must implement the Classifier, Regressor or "
        f"SurvivalModel protocol; {type(component).__name__} implements none. "
        "A classifier needs fit/predict/predict_proba, a regressor "
        "fit/predict, and a survival model fit/predict_risk."
    )
