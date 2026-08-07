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

The TABLE-level adapters moved up to the domain layer in
:mod:`habit.domain.sklearn_interop`, because
:class:`~habit.domain.pipeline.TablePipeline` is now itself an
``sklearn.pipeline.Pipeline`` built out of them and this module is a frozen
compatibility surface that must not grow new capability. The four names this
module used to own stay importable here as thin deprecated aliases for all of
v1.x::

    from habit.domain.sklearn_interop import as_classifier, as_transformer

    pipe = Pipeline([
        ("scale", as_transformer(ZScorePreprocessor())),
        ("select", as_transformer(AnovaSelector(n_features_to_select=20))),
        ("model", as_classifier(LogisticRegressionClassifier())),
    ])
    pipe.fit(train_table)          # outcome rides inside the FeatureTable
    pipe.predict_proba(holdout_table)

Note on cross-validation splits: sklearn's ``_safe_indexing`` handles any
sequence of subjects (including :class:`~habit.contracts.subject.Cohort`), so
``GridSearchCV`` works directly on cohorts. A bare ``FeatureTable`` is NOT
row-indexable by design; a CV driver over tables passes the FRAME as ``X``
and rebuilds the table in a
:class:`~habit.domain.sklearn_interop.FrameToTable` head step (which is what
``TablePipeline`` does), or uses ``habit.recipes.cross_validate``.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

from habit.exceptions import HABITAPIError
from habit.contracts.subject import Cohort, Subject
from habit.contracts.table import FeatureTable
from habit.domain.assembly import HabitatComponents, build_habitat_components
from habit.domain.protocols import Seedable
from habit.domain.sklearn_interop import (
    TableClassifierEstimator as _TableClassifierEstimator,
    TableTransformerEstimator as _TableTransformerEstimator,
    as_classifier as _as_classifier,
    as_transformer as _as_transformer,
)
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

#: Text appended to every deprecation message of this module, so a user who
#: sees one knows both where the symbol went and how long the alias lives.
_MOVED_NOTE = (
    "It moved to habit.domain.sklearn_interop when TablePipeline became an "
    "sklearn Pipeline subclass; the alias here is kept for all of v1.x."
)


def _warn_moved(old: str, new: str) -> None:
    """
    Emit the module's standard "symbol moved" deprecation warning.

    Args:
        old: Fully-qualified old name, as written in user code.
        new: Fully-qualified replacement to import instead.
    """
    warnings.warn(
        f"{old} is deprecated; use {new}. {_MOVED_NOTE}",
        DeprecationWarning,
        stacklevel=3,
    )


class TableTransformerEstimator(_TableTransformerEstimator):
    """
    Deprecated alias of
    :class:`habit.domain.sklearn_interop.TableTransformerEstimator`.

    Subclassing rather than re-binding keeps ``isinstance`` checks written
    against either name true, so existing code that tests the type keeps
    working while the warning points at the new import path. Kept through
    v1.x.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _warn_moved(
            "habit.compat.sklearn.TableTransformerEstimator",
            "habit.domain.sklearn_interop.TableTransformerEstimator",
        )
        super().__init__(*args, **kwargs)


class TableClassifierEstimator(_TableClassifierEstimator):
    """
    Deprecated alias of
    :class:`habit.domain.sklearn_interop.TableClassifierEstimator`.

    Kept through v1.x; see :class:`TableTransformerEstimator` for why this is
    a subclass rather than a re-binding.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _warn_moved(
            "habit.compat.sklearn.TableClassifierEstimator",
            "habit.domain.sklearn_interop.TableClassifierEstimator",
        )
        super().__init__(*args, **kwargs)


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


class HabitatFeaturesEstimator(TransformerMixin, BaseEstimator):
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
    def _create_components(effective: HabitatSpec) -> HabitatComponents:
        """Build the spec's components, then enforce the estimator's own
        requirement that at least one habitat feature family is declared.

        Construction itself lives in ``habit.domain.assembly``, so this
        adapter and the recipe layer cannot drift apart.
        """
        components = build_habitat_components(effective)
        if not components.extractors:
            raise HABITAPIError(
                "HabitatSpec.habitat_features is empty; the estimator's whole "
                "purpose is producing a feature matrix, so declare at least "
                "one habitat feature family."
            )
        return components

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
        units = self._fit_components(subjects, cohort)
        # Feature columns come from assigning the already-computed units of
        # one subject -- never a second Stage-1 extraction.
        first = self._describe_from_units(subjects[0], units[0])
        self.feature_names_in_ = np.asarray(first.feature_columns, dtype=object)
        return self

    def fit_transform(self, X: Any, y: Any = None, **fit_params: Any) -> np.ndarray:
        """
        Fit on the cohort and return its habitat feature matrix in one pass.

        Stage-1 clustering units are computed once per subject during fit and
        reused for assignment / habitat features. This is what
        ``sklearn.pipeline.Pipeline`` calls on intermediate steps; without
        the reuse, voxel radiomics (and any GPU workers) would run twice.
        """
        subjects, cohort = self._subjects_from(X)
        units = self._fit_components(subjects, cohort)
        first = self._describe_from_units(subjects[0], units[0])
        self.feature_names_in_ = np.asarray(first.feature_columns, dtype=object)
        matrix = self._transform_from_units(subjects, units, first_table=first)
        return matrix.to_numpy(dtype=float)

    def transform(self, X: Any) -> np.ndarray:
        """
        Project a cohort onto the fitted habitat definition.

        Predict-path: Stage-1 is recomputed from each subject's images.
        Training cohorts that still have in-memory units should call
        :meth:`fit_transform` instead of ``fit`` then ``transform``.

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

    def _fit_components(
        self, subjects: List[Subject], cohort: Cohort
    ) -> List[Any]:
        """
        Run the cohort-level fit and bind fitted state to ``self``.

        Returns:
            Per-subject clustering units BEFORE cohort preprocessing, in
            ``subjects`` order. Callers reuse these for assignment so
            Stage-1 (voxel / supervoxel features) is not paid twice.
        """
        effective = self._effective_spec()
        components = self._create_components(effective)
        # The SAME pipeline object produces units at fit time and at predict
        # time; only the assigner differs. Reimplementing the stages here is
        # how train/predict pipelines silently diverge.
        units_pipeline = components.pipeline(assigner=None)
        raw_units = [
            units_pipeline.units(subject)
            for subject in _iter_with_progress(
                subjects, enabled=self.verbose, desc="Fit: voxel->units"
            )
        ]
        cohort_chain = components.cohort_chain
        fit_units: List[Any] = list(raw_units)
        if cohort_chain is not None:
            # Cohort-level statistics come from the pooled TRAINING units and
            # nothing else; this is the one leakage-sensitive step in habitat
            # definition. Transformed copies are used only for fitting;
            # assignment re-applies the fitted chain to ``raw_units``.
            pooled = pd.concat(
                [unit.feature_frame() for unit in raw_units], ignore_index=True
            )
            cohort_chain.fit(pooled)
            fit_units = [
                unit.with_feature_frame(
                    cohort_chain.transform(unit.feature_frame()),
                    produced_by="feature_preprocessing.cohort",
                    spec_fingerprint=cohort_chain.spec.fingerprint(),
                )
                for unit in raw_units
            ]
        model = fitter_model = components.fitter.fit(fit_units, cohort=cohort)
        if cohort_chain is not None:
            # The centroids only mean something in the preprocessed feature
            # space, so the space travels with the model.
            model = fitter_model.with_cohort_preprocessing(
                cohort_chain.state, cohort_chain.spec.to_dict()
            )
        assigner = model.assigner(
            effective.habitat_assigner.name, **effective.habitat_assigner.params
        )
        if effective.random_seed is not None and isinstance(assigner, Seedable):
            assigner.set_random_state(effective.random_seed)
        self.model_ = model
        self.spec_ = effective
        self._components = components
        self._assigner = assigner
        self._extractors = components.extractors
        return raw_units

    def _fitted_pipeline(self) -> Any:
        """Return the subject pipeline with the fitted assigner attached."""
        return self._components.pipeline(assigner=self._assigner)

    def _describe_from_units(self, subject: Subject, units: Any) -> FeatureTable:
        """
        Assign habitats from precomputed units and extract habitat features.

        Args:
            subject: Owning subject (images for habitat-level descriptors).
            units: Clustering units from the fit-time Stage-1 pass.

        Returns:
            One-row habitat feature table for ``subject``.
        """
        _, table, _ = self._fitted_pipeline().label_and_describe(
            subject, units, self._extractors
        )
        if table is None:
            raise HABITAPIError(
                "HabitatFeaturesEstimator requires habitat feature extractors; "
                "label_and_describe returned no table."
            )
        return table

    def _extract_one(self, subject: Subject) -> FeatureTable:
        """Predict-path: recompute Stage-1 from images, then describe."""
        return self._fitted_pipeline().extract_features(subject, self._extractors)

    def _align_feature_matrix(self, combined: pd.DataFrame) -> pd.DataFrame:
        """Enforce the fit-time column layout on a stacked feature matrix."""
        missing = [c for c in self.feature_names_in_ if c not in combined.columns]
        if missing:
            raise HABITAPIError(
                "Transformed table lacks fit-time feature columns "
                f"{missing}; the habitat feature layout drifted."
            )
        return combined.loc[:, list(self.feature_names_in_)]

    def _transform_from_units(
        self,
        subjects: List[Subject],
        units: Sequence[Any],
        *,
        first_table: Optional[FeatureTable] = None,
    ) -> pd.DataFrame:
        """
        Build the feature matrix by assigning precomputed clustering units.

        Args:
            subjects: Subjects in the same order as ``units``.
            units: Stage-1 units from :meth:`_fit_components`.
            first_table: Optional precomputed table for ``subjects[0]``.

        Returns:
            Frame indexed by subject id with exactly the fit-time columns.
        """
        if len(subjects) != len(units):
            raise HABITAPIError(
                "HabitatFeaturesEstimator: subjects and units length mismatch "
                f"({len(subjects)} vs {len(units)})."
            )
        rows: List[pd.DataFrame] = []
        start = 0
        if first_table is not None:
            rows.append(first_table.feature_matrix())
            start = 1
        for subject, subject_units in _iter_with_progress(
            list(zip(subjects[start:], units[start:])),
            enabled=self.verbose,
            desc="Habitat features",
        ):
            rows.append(
                self._describe_from_units(subject, subject_units).feature_matrix()
            )
        return self._align_feature_matrix(pd.concat(rows))

    def _transform_subjects(
        self, subjects: List[Subject], *, first_table: Optional[FeatureTable] = None
    ) -> pd.DataFrame:
        """
        Compute the feature matrix for ``subjects`` from images (predict path).

        Args:
            subjects: Subjects to process, in order.
            first_table: Precomputed table for ``subjects[0]`` when the
                caller already labelled that subject.

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
        return self._align_feature_matrix(pd.concat(rows))

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



def as_transformer(component: Any, **options: Any) -> _TableTransformerEstimator:
    """
    Deprecated alias of
    :func:`habit.domain.sklearn_interop.as_transformer` (kept through v1.x).

    Args:
        component: ``TablePreprocessor`` or ``FeatureSelector`` implementation.
        **options: Forwarded verbatim to the domain-layer factory.

    Returns:
        The configured adapter, of the domain-layer type.
    """
    _warn_moved(
        "habit.compat.sklearn.as_transformer()",
        "habit.domain.sklearn_interop.as_transformer()",
    )
    return _as_transformer(component, **options)


def as_classifier(component: Any, **options: Any) -> _TableClassifierEstimator:
    """
    Deprecated alias of
    :func:`habit.domain.sklearn_interop.as_classifier` (kept through v1.x).

    Args:
        component: ``Classifier`` implementation.
        **options: Forwarded verbatim to the domain-layer factory.

    Returns:
        The configured adapter, of the domain-layer type.
    """
    _warn_moved(
        "habit.compat.sklearn.as_classifier()",
        "habit.domain.sklearn_interop.as_classifier()",
    )
    return _as_classifier(component, **options)
