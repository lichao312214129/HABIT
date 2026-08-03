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
"""scikit-learn interop: HABIT components as genuine ``BaseEstimator`` adapters.

The domain layer deliberately does NOT follow sklearn conventions:
``HabitatModelFitter.fit`` returns a NEW :class:`~habit.contracts.habitat.HabitatModel`
artefact rather than ``self`` (the habitat definition is a scientific product,
not internal estimator state), and components are seeded through
``set_random_state`` rather than a ``random_state`` constructor argument. This
module bridges that divergence instead of letting the domain API bend to
sklearn -- the ``*Estimator`` name, reserved by the v1.0 naming decisions for
exactly these adapters, signals objects whose ``fit`` really returns ``self``
and that are ``sklearn.base.clone``-able.

Typical use::

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline

    from habit.compat.sklearn import as_estimator

    pipe = Pipeline([
        ("habitats", as_estimator(habitat_spec, random_seed=42)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    search = GridSearchCV(pipe, {"habitats__n_habitats": [3, 4, 5]},
                          scoring="roc_auc", cv=5)
    search.fit(cohort, labels)   # cohort: a Sequence of habit Subject

The table-ML adapters keep ``FeatureTable`` semantics inside an sklearn
pipeline::

    from habit.compat.sklearn import as_classifier, as_transformer

    pipe = Pipeline([
        ("scale", as_transformer(ZScorePreprocessor())),
        ("select", as_transformer(AnovaSelector(top_n=20))),
        ("model", as_classifier(LogisticRegressionClassifier())),
    ])
    pipe.fit(train_table)          # outcome rides inside the FeatureTable
    pipe.predict_proba(holdout_table)

Note on cross-validation splits: sklearn's ``_safe_indexing`` handles any
sequence of subjects (including :class:`~habit.contracts.subject.Cohort`), so
``GridSearchCV`` works directly on cohorts. A bare ``FeatureTable`` is NOT
row-indexable by design, so CV drivers over tables should split the frame
first (or use ``habit.recipes.cross_validate`` when it lands).
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.exceptions import NotFittedError

from habit.api.exceptions import HABITAPIError
from habit.contracts.subject import Cohort, Subject
from habit.contracts.table import FeatureTable
from habit.domain.habitat_features import HabitatFeatureExtractorRegistry
from habit.domain.habitat_model import HabitatModelFitterRegistry
from habit.domain.pipeline import SubjectPipeline, voxel_units
from habit.domain.protocols import Seedable
from habit.domain.supervoxel import SupervoxelizerRegistry
from habit.domain.table_protocols import Classifier, FeatureSelector, TablePreprocessor
from habit.domain.voxel_features import VoxelFeatureExtractorRegistry
from habit.spec.specs import HabitatSpec
from habit.utils.progress_utils import CustomTqdm

__all__ = [
    "HabitatFeaturesEstimator",
    "TableTransformerEstimator",
    "TableClassifierEstimator",
    "as_estimator",
    "as_transformer",
    "as_classifier",
]


def _iter_with_progress(
    items: Sequence[Any], *, enabled: bool, desc: str
) -> Iterator[Any]:
    """
    Yield ``items``, showing a HABIT-standard progress bar when enabled.

    Args:
        items: Sequence to iterate.
        enabled: Whether to wrap the iteration in ``CustomTqdm``.
        desc: Progress-bar label.

    Yields:
        The items, in order.
    """
    if not enabled:
        yield from items
        return
    bar = CustomTqdm(total=len(items), desc=desc)
    try:
        for item in items:
            yield item
            bar.update(1)
    finally:
        bar.close()


class HabitatFeaturesEstimator(BaseEstimator, TransformerMixin):
    """
    Turn a cohort of subjects into a habitat feature matrix, sklearn-style.

    The adapter owns the whole habitat computation declared by a
    :class:`~habit.spec.specs.HabitatSpec`: per-subject voxel features and
    supervoxels, the cohort-level habitat model fit, per-subject assignment
    and habitat feature extraction. ``fit`` learns the habitat definition on
    the TRAINING cohort only; ``transform`` projects any later cohort onto
    that fixed definition and always returns the fit-time column layout, so
    cross-validation folds can never leak into the habitat definition.

    Args:
        spec: The habitat analysis to run.
        n_habitats: Optional override of the fitter's habitat count. Exposed
            as a top-level constructor parameter so ``GridSearchCV`` can tune
            it as ``habitats__n_habitats``.
        n_supervoxels: Optional override of the supervoxelizer's segment
            count (two-step designs only).
        random_seed: Optional override of ``spec.random_seed``; applied to
            every :class:`~habit.domain.protocols.Seedable` component through
            their ``set_random_state``.
        verbose: Show per-subject progress bars during fit/transform.

    Attributes:
        model_: The fitted :class:`~habit.contracts.habitat.HabitatModel`.
        spec_: The effective spec after applying constructor overrides.
        feature_names_in_: Feature column names captured at fit time.
    """

    def __init__(
        self,
        spec: HabitatSpec,
        *,
        n_habitats: Optional[int] = None,
        n_supervoxels: Optional[int] = None,
        random_seed: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        self.spec = spec
        self.n_habitats = n_habitats
        self.n_supervoxels = n_supervoxels
        self.random_seed = random_seed
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Spec resolution and component construction
    # ------------------------------------------------------------------

    def _effective_spec(self) -> HabitatSpec:
        """Return the spec with constructor overrides folded in."""
        if not isinstance(self.spec, HabitatSpec):
            raise HABITAPIError(
                "HabitatFeaturesEstimator.spec must be a HabitatSpec; got "
                f"{type(self.spec).__name__}."
            )
        effective = self.spec
        if self.n_habitats is not None:
            fitter_spec = effective.habitat_model_fitter
            effective = dataclasses.replace(
                effective,
                habitat_model_fitter=dataclasses.replace(
                    fitter_spec,
                    params={**fitter_spec.params, "n_habitats": self.n_habitats},
                ),
            )
        if self.n_supervoxels is not None:
            if effective.supervoxelizer is None:
                raise HABITAPIError(
                    "n_supervoxels overrides require a supervoxelizer in the "
                    "spec (two-step design); this spec clusters voxels directly."
                )
            svx_spec = effective.supervoxelizer
            effective = dataclasses.replace(
                effective,
                supervoxelizer=dataclasses.replace(
                    svx_spec,
                    params={**svx_spec.params, "n_supervoxels": self.n_supervoxels},
                ),
            )
        if self.random_seed is not None:
            effective = dataclasses.replace(
                effective, random_seed=int(self.random_seed)
            )
        return effective

    @staticmethod
    def _create_components(
        effective: HabitatSpec,
    ) -> Tuple[Any, Any, Any, Tuple[Any, ...]]:
        """Instantiate the pipeline components declared by the spec."""
        voxel_extractor = VoxelFeatureExtractorRegistry.create(
            effective.voxel_feature_extractor.name,
            **effective.voxel_feature_extractor.params,
        )
        supervoxelizer = None
        if effective.supervoxelizer is not None:
            supervoxelizer = SupervoxelizerRegistry.create(
                effective.supervoxelizer.name, **effective.supervoxelizer.params
            )
        fitter = HabitatModelFitterRegistry.create(
            effective.habitat_model_fitter.name, **effective.habitat_model_fitter.params
        )
        extractors = tuple(
            HabitatFeatureExtractorRegistry.create(
                feature_spec.name, **feature_spec.params
            )
            for feature_spec in effective.habitat_features
        )
        if not extractors:
            raise HABITAPIError(
                "HabitatSpec.habitat_features is empty; the estimator's whole "
                "purpose is producing a feature matrix, so declare at least "
                "one habitat feature family."
            )
        if effective.random_seed is not None:
            for component in (voxel_extractor, supervoxelizer, fitter, *extractors):
                if isinstance(component, Seedable):
                    component.set_random_state(effective.random_seed)
        return voxel_extractor, supervoxelizer, fitter, extractors

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(self, X: Any, y: Any = None) -> "HabitatFeaturesEstimator":
        """
        Learn the habitat definition from the training cohort.

        Args:
            X: Iterable of :class:`~habit.contracts.subject.Subject`
                (a ``Cohort`` keeps its name/metadata in the model card).
            y: Ignored -- habitat fitting is unsupervised. Accepted for
                sklearn pipeline compatibility.

        Returns:
            ``self``, fitted.
        """
        subjects, cohort = self._subjects_from(X)
        self._fit_components(subjects, cohort)
        # Feature columns are fully determined by the fitted model plus the
        # extractor specs, so one subject suffices to capture the layout.
        first = self._extract_one(subjects[0])
        self.feature_names_in_ = np.asarray(first.feature_columns, dtype=object)
        return self

    def fit_transform(self, X: Any, y: Any = None, **fit_params: Any) -> np.ndarray:
        """
        Fit on the cohort and return its habitat feature matrix in one pass.

        Overriding :meth:`fit` + :meth:`transform` avoids computing the
        per-subject pipeline twice for the training cohort, which is what
        ``sklearn.pipeline.Pipeline`` calls on intermediate steps.
        """
        subjects, cohort = self._subjects_from(X)
        self._fit_components(subjects, cohort)
        first = self._extract_one(subjects[0])
        self.feature_names_in_ = np.asarray(first.feature_columns, dtype=object)
        matrix = self._transform_subjects(subjects, first_table=first)
        return matrix.to_numpy(dtype=float)

    def transform(self, X: Any) -> np.ndarray:
        """
        Project a cohort onto the fitted habitat definition.

        Args:
            X: Iterable of :class:`~habit.contracts.subject.Subject`.

        Returns:
            Feature matrix of shape ``(n_subjects, n_features)`` with the
            fit-time column layout, rows in input subject order.

        Raises:
            NotFittedError: If called before :meth:`fit`.
            HABITAPIError: If the computed table lacks fit-time columns.
        """
        self._check_fitted()
        subjects, _ = self._subjects_from(X)
        return self._transform_subjects(subjects).to_numpy(dtype=float)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return the habitat feature names produced by :meth:`transform`."""
        self._check_fitted()
        return np.asarray(self.feature_names_in_, dtype=object)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _subjects_from(X: Any) -> Tuple[List[Subject], Cohort]:
        """Normalise ``X`` into a subject list plus a cohort for fingerprinting."""
        if isinstance(X, Cohort):
            subjects = list(X)
            cohort = X
        else:
            try:
                subjects = list(X)
            except TypeError as exc:
                raise HABITAPIError(
                    "HabitatFeaturesEstimator expects an iterable of Subject; "
                    f"got {type(X).__name__}."
                ) from exc
            cohort = None  # type: ignore[assignment]
        if not subjects:
            raise HABITAPIError("HabitatFeaturesEstimator received no subjects.")
        for item in subjects:
            if not isinstance(item, Subject):
                raise HABITAPIError(
                    "HabitatFeaturesEstimator expects an iterable of Subject; "
                    f"found {type(item).__name__}."
                )
        if cohort is None:
            # The fitter only uses the cohort for its non-identifiable
            # fingerprint, so a derived wrapper loses no information.
            cohort = Cohort(subjects, name="fit")
        return subjects, cohort

    def _fit_components(self, subjects: List[Subject], cohort: Cohort) -> None:
        """Run the cohort-level fit and bind fitted state to ``self``."""
        effective = self._effective_spec()
        voxel_extractor, supervoxelizer, fitter, extractors = self._create_components(
            effective
        )
        units = []
        for subject in _iter_with_progress(
            subjects, enabled=self.verbose, desc="Fit: voxel->units"
        ):
            field = voxel_extractor(subject)
            units.append(
                supervoxelizer(field) if supervoxelizer is not None else voxel_units(field)
            )
        model = fitter.fit(units, cohort=cohort)
        assigner = model.assigner(
            effective.habitat_assigner.name, **effective.habitat_assigner.params
        )
        if effective.random_seed is not None and isinstance(assigner, Seedable):
            assigner.set_random_state(effective.random_seed)
        self.model_ = model
        self.spec_ = effective
        self._voxel_extractor = voxel_extractor
        self._supervoxelizer = supervoxelizer
        self._assigner = assigner
        self._extractors = extractors

    def _extract_one(self, subject: Subject) -> FeatureTable:
        """Run the full per-subject chain and return its one-row table."""
        pipeline = SubjectPipeline(
            voxel_feature_extractor=self._voxel_extractor,
            supervoxelizer=self._supervoxelizer,
            habitat_assigner=self._assigner,
        )
        return pipeline.extract_features(subject, self._extractors)

    def _transform_subjects(
        self, subjects: List[Subject], *, first_table: Optional[FeatureTable] = None
    ) -> pd.DataFrame:
        """
        Compute the feature matrix for ``subjects`` in fit-time layout.

        Args:
            subjects: Subjects to process, in order.
            first_table: Precomputed table for ``subjects[0]`` (the
                fit_transform one-pass path already ran that subject).

        Returns:
            Frame indexed by subject id with exactly the fit-time columns.
        """
        rows: List[pd.DataFrame] = []
        remaining = subjects
        if first_table is not None:
            rows.append(first_table.feature_matrix())
            remaining = subjects[1:]
        for subject in _iter_with_progress(
            remaining, enabled=self.verbose, desc="Habitat features"
        ):
            rows.append(self._extract_one(subject).feature_matrix())
        combined = pd.concat(rows)
        missing = [c for c in self.feature_names_in_ if c not in combined.columns]
        if missing:
            raise HABITAPIError(
                "Transformed table lacks fit-time feature columns "
                f"{missing}; the habitat feature layout drifted."
            )
        return combined.loc[:, list(self.feature_names_in_)]

    def _check_fitted(self) -> None:
        """Guard sklearn's "no prediction before fitting" contract."""
        if not hasattr(self, "model_"):
            raise NotFittedError(
                "This HabitatFeaturesEstimator is not fitted yet; call fit "
                "with a training cohort first."
            )


def as_estimator(spec: HabitatSpec, **overrides: Any) -> HabitatFeaturesEstimator:
    """
    Wrap a :class:`~habit.spec.specs.HabitatSpec` as a sklearn transformer.

    Args:
        spec: The habitat analysis to run.
        **overrides: Forwarded to :class:`HabitatFeaturesEstimator`
            (``n_habitats`` / ``n_supervoxels`` / ``random_seed`` /
            ``verbose``).

    Returns:
        The configured estimator, ready for ``Pipeline`` / ``GridSearchCV``.
    """
    return HabitatFeaturesEstimator(spec, **overrides)


def _require_table(X: Any) -> FeatureTable:
    """Validate that a pipeline step received a ``FeatureTable``."""
    if not isinstance(X, FeatureTable):
        raise HABITAPIError(
            "Table adapters operate on habit FeatureTable objects; got "
            f"{type(X).__name__}. Chain them with other table-aware steps."
        )
    return X


class TableTransformerEstimator(BaseEstimator, TransformerMixin):
    """
    Adapt a HABIT table transformation to the sklearn transformer API.

    ``X`` is a :class:`~habit.contracts.table.FeatureTable` in and out, so the
    adapter composes with other table-aware steps inside an sklearn
    ``Pipeline`` while the fitted state (training statistics, selected
    columns) follows sklearn's clone/fit lifecycle. The wrapped component is
    deep-copied at fit time, so the instance the user passed is never mutated.

    Args:
        component: A :class:`~habit.domain.table_protocols.TablePreprocessor`
            or :class:`~habit.domain.table_protocols.FeatureSelector`
            implementation.
    """

    def __init__(self, component: Any) -> None:
        self.component = component

    def fit(self, X: Any, y: Any = None, **fit_params: Any) -> "TableTransformerEstimator":
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
        """
        table = _require_table(X)
        if not isinstance(self.component, (TablePreprocessor, FeatureSelector)):
            raise HABITAPIError(
                "TableTransformerEstimator wraps a TablePreprocessor or "
                f"FeatureSelector; got {type(self.component).__name__}."
            )
        self.component_ = copy.deepcopy(self.component)
        self.component_.fit(table, **fit_params)
        return self

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


class TableClassifierEstimator(BaseEstimator, ClassifierMixin):
    """
    Adapt a HABIT :class:`~habit.domain.table_protocols.Classifier` to sklearn.

    The outcome rides inside the ``FeatureTable`` (the leakage-safe HABIT
    convention); a separate ``y`` is only used to fill a missing outcome
    column, and is cross-checked against an existing one so a misaligned
    ``y`` fails loudly instead of silently training on the wrong labels.

    Args:
        component: A HABIT ``Classifier`` implementation.
    """

    def __init__(self, component: Any) -> None:
        self.component = component

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

        Raises:
            HABITAPIError: On missing outcome or a ``y``/outcome mismatch.
        """
        table = _require_table(X)
        if not isinstance(self.component, Classifier):
            raise HABITAPIError(
                "TableClassifierEstimator wraps a HABIT Classifier; got "
                f"{type(self.component).__name__}."
            )
        if table.outcome_column is not None and y is not None:
            outcome = table.frame[table.outcome_column].to_numpy()
            if not np.array_equal(np.asarray(y), outcome):
                raise HABITAPIError(
                    "y disagrees with the table's outcome column; refusing "
                    "to train on ambiguous labels."
                )
        if table.outcome_column is None:
            if y is None:
                raise HABITAPIError(
                    "Training table has no outcome column and no y was given."
                )
            table = self._attach_outcome(table, y)
        self.component_ = copy.deepcopy(self.component)
        self.component_.fit(table)
        # HABIT classifiers label probability columns by class (e.g. "0"/"1");
        # capturing the labels from the fitted classifier itself guarantees
        # predict_proba stays column-aligned without private-state access.
        probe = dataclasses.replace(table, frame=table.frame.head(1))
        self.classes_ = np.asarray(self.component_.predict_proba(probe).columns)
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels for a table's rows."""
        self._check_fitted()
        return self.component_.predict(_require_table(X)).to_numpy()

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities, columns aligned to ``classes_``."""
        self._check_fitted()
        proba = self.component_.predict_proba(_require_table(X))
        aligned = proba.reindex(columns=list(self.classes_))
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
            table = _require_table(X)
            if table.outcome_column is None:
                raise HABITAPIError(
                    "score needs y or a table carrying an outcome column."
                )
            y = table.frame[table.outcome_column].to_numpy()
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

    @staticmethod
    def _attach_outcome(table: FeatureTable, y: Any) -> FeatureTable:
        """Return a copy of ``table`` with ``y`` attached as outcome column."""
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
        return FeatureTable(
            frame=frame,
            id_columns=table.id_columns,
            feature_columns=table.feature_columns,
            outcome_column=column,
            provenance=table.provenance,
        )

    def _check_fitted(self) -> None:
        """Guard sklearn's "no prediction before fitting" contract."""
        if not hasattr(self, "component_"):
            raise NotFittedError(
                "This TableClassifierEstimator is not fitted yet; call fit first."
            )


def as_transformer(component: Any) -> TableTransformerEstimator:
    """
    Wrap a HABIT table preprocessor/selector as a sklearn transformer.

    Args:
        component: ``TablePreprocessor`` or ``FeatureSelector`` implementation.

    Returns:
        The configured adapter.
    """
    return TableTransformerEstimator(component)


def as_classifier(component: Any) -> TableClassifierEstimator:
    """
    Wrap a HABIT classifier as a sklearn classifier.

    Args:
        component: ``Classifier`` implementation.

    Returns:
        The configured adapter.
    """
    return TableClassifierEstimator(component)
