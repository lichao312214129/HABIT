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
"""Pipelines: subject-level and table-level component composition."""

from __future__ import annotations

import inspect
import io
import json
import pickle
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from habit.exceptions import CompatibilityError, HABITAPIError
from habit.contracts.habitat import HabitatMap, Supervoxelization, VoxelFeatureField
from habit.contracts.subject import Subject
from habit.contracts.table import FeatureTable
from habit.domain.outcome_access import outcome_series, survival_target
from habit.domain.protocols import (
    CohortFeaturePreprocessor,
    HabitatAssigner,
    HabitatFeatureExtractor,
    Seedable,
    SubjectFeaturePreprocessor,
    SupervoxelFeatureExtractor,
    Supervoxelizer,
    VoxelFeatureExtractor,
)
from habit.domain.table_protocols import (
    Classifier,
    FeatureSelector,
    Metric,
    RegressionMetric,
    Regressor,
    SurvivalMetric,
    SurvivalModel,
    TablePreprocessor,
)
from habit._version import __version__ as _habit_version
from habit.spec.specs import Spec

__all__ = ["SubjectPipeline", "TablePipeline", "voxel_units"]


def voxel_units(field: VoxelFeatureField) -> Supervoxelization:
    """
    Wrap a voxel feature field as single-voxel clustering units.

    The one-step and direct-pooling designs cluster voxels directly, with no
    supervoxel step. Representing each voxel as a one-voxel
    ``Supervoxelization`` keeps the assigner contract uniform instead of
    giving assigners a second input type to handle. It is also the building
    block external code (e.g. ``habit.compat.sklearn``) needs to drive a
    one-step design outside a ``SubjectPipeline``.

    Args:
        field: Per-voxel features for one subject.

    Returns:
        A partition in which every ROI voxel is its own unit.
    """
    n_voxels = field.values.shape[0]
    labels = np.zeros(tuple(int(v) for v in field.geometry.shape), dtype=np.int32)
    unit_ids = np.arange(1, n_voxels + 1, dtype=np.int32)
    labels[tuple(field.voxel_index.T)] = unit_ids
    features = pd.DataFrame(field.values, columns=list(field.feature_names))
    features.index = pd.Index(unit_ids, name="supervoxel")
    provenance = field.provenance.derive(
        produced_by="pipeline.voxel_units",
        spec_fingerprint="",
    )
    return Supervoxelization(
        subject_id=field.subject_id,
        label_array=labels,
        features=features,
        geometry=field.geometry,
        provenance=provenance,
    )


class SubjectPipeline:
    """
    The subject-level chain composed into a single callable.

    HABIT's answer to ``monai.transforms.Compose``. A generic ``Compose``
    cannot be reused directly because HABIT's steps are heterogeneously typed
    -- ``Subject -> VoxelFeatureField -> Supervoxelization -> HabitatMap`` --
    and erasing those types would discard exactly the contracts that make
    the design checkable.

    A fitted :class:`~habit.contracts.habitat.HabitatModel` plus a
    ``SubjectPipeline`` is precisely the pair a study publishes for external
    validation: the definition, and the procedure that applies it.

    Args:
        voxel_feature_extractor: Step producing per-voxel features.
        supervoxelizer: Step producing supervoxels. ``None`` clusters voxels
            directly, which is what the one-step and direct-pooling
            designs do.
        habitat_assigner: Step assigning habitat labels, already bound to a
            fitted model. ``None`` builds a FIT-TIME pipeline: :meth:`units`
            works, :meth:`__call__` does not. Cohort-level fitting needs
            exactly that, and sharing this class rather than reimplementing
            the stages is what guarantees a model is applied to units produced
            the same way it was fitted on.
        supervoxel_feature_extractor: Optional step describing the
            supervoxels. ``None`` keeps the feature means the supervoxelizer
            attached, which is the v0.1 default; a
            ``supervoxel_radiomics`` extractor replaces them with texture
            features. Ignored when ``supervoxelizer`` is ``None``, since a
            single voxel has no region to describe -- mirroring v0.1, where
            the one-step design ignores the ``supervoxel_level`` block.
        voxel_feature_preprocessor: Optional stateless preprocessing of the
            voxel features, applied BEFORE supervoxelisation. This is v0.1's
            ``preprocessing_for_subject_level``, and its position matters:
            normalising each subject before its ROI is partitioned is what
            keeps supervoxel boundaries from tracking scanner intensity scale.
        supervoxel_feature_preprocessor: Optional stateless preprocessing of
            the supervoxel features. The slot v0.1 lacked entirely -- per
            supervoxel radiomics had no way to be normalised within a subject
            before cohort pooling. Requires a supervoxelizer, for the same
            reason as ``supervoxel_feature_extractor``.
        cohort_feature_preprocessor: Optional FITTED cohort-level chain,
            applied last, immediately before assignment. Required whenever
            the habitat model was fitted on cohort-preprocessed units:
            omitting it would feed the assigner a feature space different
            from the one the model was defined in, and it would still return
            plausible-looking labels.
    """

    def __init__(
        self,
        voxel_feature_extractor: VoxelFeatureExtractor,
        supervoxelizer: Optional[Supervoxelizer],
        habitat_assigner: Optional[HabitatAssigner],
        supervoxel_feature_extractor: Optional[SupervoxelFeatureExtractor] = None,
        voxel_feature_preprocessor: Optional[SubjectFeaturePreprocessor] = None,
        supervoxel_feature_preprocessor: Optional[SubjectFeaturePreprocessor] = None,
        cohort_feature_preprocessor: Optional[CohortFeaturePreprocessor] = None,
    ) -> None:
        if voxel_feature_extractor is None:
            raise HABITAPIError(
                "SubjectPipeline requires a voxel feature extractor; there is "
                "no habitat analysis without per-voxel features."
            )
        if supervoxel_feature_extractor is not None and supervoxelizer is None:
            raise HABITAPIError(
                "SubjectPipeline received a supervoxel feature extractor but "
                "no supervoxelizer. Direct voxel clustering has no supervoxel "
                "to describe; either add a supervoxelizer or drop the "
                "extractor."
            )
        if supervoxel_feature_preprocessor is not None and supervoxelizer is None:
            raise HABITAPIError(
                "SubjectPipeline received a supervoxel feature preprocessor "
                "but no supervoxelizer. Without supervoxels there is only one "
                "feature matrix to preprocess; pass it as "
                "voxel_feature_preprocessor instead."
            )
        self.voxel_feature_extractor = voxel_feature_extractor
        self.supervoxelizer = supervoxelizer
        self.habitat_assigner = habitat_assigner
        self.supervoxel_feature_extractor = supervoxel_feature_extractor
        self.voxel_feature_preprocessor = voxel_feature_preprocessor
        self.supervoxel_feature_preprocessor = supervoxel_feature_preprocessor
        self.cohort_feature_preprocessor = cohort_feature_preprocessor

    @property
    def spec(self) -> Spec:
        """Return the composed specification of every stage."""

        def _optional(component: Any) -> Optional[Dict[str, Any]]:
            """Return a component's spec payload, or None when absent."""
            return component.spec.to_dict() if component is not None else None

        stage_specs: Dict[str, Any] = {
            "voxel_feature_extractor": self.voxel_feature_extractor.spec.to_dict(),
            "voxel_feature_preprocessor": _optional(self.voxel_feature_preprocessor),
            "supervoxelizer": _optional(self.supervoxelizer),
            "supervoxel_feature_extractor": _optional(
                self.supervoxel_feature_extractor
            ),
            "supervoxel_feature_preprocessor": _optional(
                self.supervoxel_feature_preprocessor
            ),
            "cohort_feature_preprocessor": _optional(
                self.cohort_feature_preprocessor
            ),
            "habitat_assigner": _optional(self.habitat_assigner),
        }
        return Spec(name="subject_pipeline", params=stage_specs)

    def units(self, subject: Subject) -> Supervoxelization:
        """
        Run every stage up to (but excluding) habitat assignment.

        Exposed separately because cohort-level fitting needs exactly this:
        the clustering units of each training subject, pooled and then used to
        DEFINE the habitats. Sharing one implementation with :meth:`__call__`
        is what guarantees a model is applied to units produced the same way
        they were fitted on.

        Args:
            subject: The subject to process.

        Returns:
            The subject's clustering units. Every ROI voxel is its own unit
            when no supervoxelizer is configured.
        """
        field = self.voxel_feature_extractor(subject)
        # Keep the pre-preprocessing field: statistical supervoxel
        # extractors with ``source="original"`` aggregate exactly this
        # signal (the v0.1 ``-original`` column contract).
        original_field = field
        if self.voxel_feature_preprocessor is not None:
            chain = self.voxel_feature_preprocessor
            field = field.with_feature_frame(
                chain(field.feature_frame()),
                produced_by="feature_preprocessing.subject.voxel",
                spec_fingerprint=chain.spec.fingerprint(),
            )
        if self.supervoxelizer is None:
            return voxel_units(field)
        units = self.supervoxelizer(field)
        if self.supervoxel_feature_extractor is not None:
            # Statistical extractors (``mean`` / ``std`` / ``percentile``,
            # standalone or inside a tree) recompute their statistic from
            # the voxel fields instead of the attached means.
            binder = getattr(self.supervoxel_feature_extractor, "bind_fields", None)
            if callable(binder):
                binder(working=field, original=original_field)
            units = self.supervoxel_feature_extractor(subject, units)
        if self.supervoxel_feature_preprocessor is not None:
            chain = self.supervoxel_feature_preprocessor
            units = units.with_feature_frame(
                chain(units.feature_frame()),
                produced_by="feature_preprocessing.subject.supervoxel",
                spec_fingerprint=chain.spec.fingerprint(),
            )
        return units

    def assign(
        self, units: Supervoxelization
    ) -> Tuple[HabitatMap, Supervoxelization]:
        """
        Assign habitats from clustering units already produced by :meth:`units`.

        This is the train-path reuse hook: cohort-level fit recipes and
        sklearn adapters compute Stage-1 units once, then call this instead
        of :meth:`__call__` (which would re-extract voxel / supervoxel
        features). Predict / apply paths keep calling :meth:`__call__` so
        held-out subjects are still derived from images.

        Args:
            units: Precomputed clustering units for one subject (before
                cohort-level preprocessing).

        Returns:
            ``(habitat_map, units_after_cohort_prep)``. The post-prep units
            feed the v0.1 ``habitats.parquet`` unit table at the writer.

        Raises:
            HABITAPIError: If this is a fit-time pipeline (no assigner).
        """
        if self.habitat_assigner is None:
            raise HABITAPIError(
                "This SubjectPipeline was built without a habitat assigner, so "
                "it can only produce clustering units (pipeline.units(subject)). "
                "Fit a model on those units, then rebuild the pipeline with "
                "model.assigner() to label subjects."
            )
        working = units
        if self.cohort_feature_preprocessor is not None:
            chain = self.cohort_feature_preprocessor
            working = working.with_feature_frame(
                chain.transform(working.feature_frame()),
                produced_by="feature_preprocessing.cohort",
                spec_fingerprint=chain.spec.fingerprint(),
            )
        return self.habitat_assigner(working), working

    def __call__(self, subject: Subject) -> HabitatMap:
        """
        Run voxel features, supervoxelisation and assignment for one subject.

        Args:
            subject: The subject to label.

        Returns:
            The subject's habitat label image.

        Raises:
            HABITAPIError: If this is a fit-time pipeline (no assigner).
        """
        habitat_map, _ = self.assign(self.units(subject))
        return habitat_map

    def label_and_describe(
        self,
        subject: Subject,
        units: Supervoxelization,
        extractors: Sequence[HabitatFeatureExtractor],
    ) -> Tuple[HabitatMap, Optional[FeatureTable], Supervoxelization]:
        """
        Assign habitats from precomputed units, then extract habitat features.

        Args:
            subject: Subject providing images for habitat-level descriptors.
            units: Clustering units from an earlier Stage-1 pass.
            extractors: Habitat feature families; may be empty when only the
                label map is needed.

        Returns:
            ``(habitat_map, feature_table_or_none, units_after_cohort_prep)``.
        """
        habitat_map, prepared = self.assign(units)
        if not extractors:
            return habitat_map, None, prepared
        table = extractors[0](subject, habitat_map)
        for extractor in extractors[1:]:
            table = table.join(extractor(subject, habitat_map))
        return habitat_map, table, prepared

    def extract_features(
        self,
        subject: Subject,
        extractors: Sequence[HabitatFeatureExtractor],
    ) -> FeatureTable:
        """
        Run the pipeline and then the requested habitat feature families.

        Named ``extract_features`` (an action) rather than the bare noun
        ``features``, which would read as an attribute on a callable object.
        Recomputes Stage-1 from ``subject`` (predict-path semantics). When
        units are already in memory, call :meth:`label_and_describe` instead.

        Args:
            subject: The subject to process.
            extractors: Habitat feature families to compute.

        Returns:
            One feature table for that subject, joined across families.

        Raises:
            HABITAPIError: If ``extractors`` is empty.
        """
        if not extractors:
            raise HABITAPIError(
                "SubjectPipeline.extract_features requires at least one "
                "habitat feature extractor."
            )
        _, table, _ = self.label_and_describe(
            subject, self.units(subject), extractors
        )
        assert table is not None
        return table


# ---------------------------------------------------------------------------
# TablePipeline: fitted preprocessing/selection + classifier over FeatureTable
# ---------------------------------------------------------------------------

#: On-disk format identifier and the version this HABIT build can read/write.
#: Bump ``_PIPELINE_FORMAT_VERSION`` (and extend the loader) whenever the
#: layout changes; older files must either load or fail with a clear message.
_PIPELINE_FORMAT_NAME = "habit.tablepipeline"
_PIPELINE_FORMAT_VERSION = 1


class TablePipeline:
    """
    Fitted preprocessing/selection chain plus classifier over feature tables.

    The structural answer to the train/predict leakage class of bugs: the
    preprocessing and feature-selection steps are fitted ONCE on the training
    table and their fitted state is what ``predict``/``transform`` apply to
    any later table -- the prediction data is normalised with the TRAINING
    statistics and reduced with the TRAINING selection, never re-fitted.

    The fitted pipeline is also the artefact a study publishes for external
    validation of its tabular model, which is why :meth:`save` persists the
    steps and the classifier together in one versioned, self-describing
    file (a JSON manifest recording every component's :class:`~habit.spec.specs.Spec`
    alongside the pickled fitted state).

    Args:
        steps: Ordered transformation steps (``TablePreprocessor`` and/or
            ``FeatureSelector`` implementations). May be empty, in which case
            the pipeline is the bare model.
        model: The terminal outcome model -- a :class:`Classifier`,
            :class:`Regressor`, or :class:`SurvivalModel`, matched to the
            endpoint family of the tables it will be fitted on.
        classifier: Deprecated alias for ``model`` (binary/multiclass
            endpoints); kept so existing call sites keep working.
    """

    def __init__(
        self,
        steps: Sequence[Union[TablePreprocessor, FeatureSelector]],
        model: Optional[Union[Classifier, Regressor, SurvivalModel]] = None,
        *,
        classifier: Optional[Classifier] = None,
    ) -> None:
        if model is None and classifier is not None:
            model = classifier
        if model is None:
            raise HABITAPIError("TablePipeline requires a terminal model.")
        self._steps: List[Union[TablePreprocessor, FeatureSelector]] = list(steps)
        self._model = model
        # Which steps learn from repeat-measurement tables (ICC selection).
        self._step_takes_repeats: List[bool] = [
            "repeat_tables" in inspect.signature(step.fit).parameters
            for step in self._steps
        ]
        self._is_fitted = False
        self._fit_output_columns: Tuple[str, ...] = ()

    @property
    def steps(self) -> Tuple[Union[TablePreprocessor, FeatureSelector], ...]:
        """Return the ordered transformation steps."""
        return tuple(self._steps)

    @property
    def model(self) -> Union[Classifier, Regressor, SurvivalModel]:
        """Return the terminal outcome model."""
        return self._model

    @property
    def classifier(self) -> Classifier:
        """Return the terminal model, asserted to be a classifier."""
        return self._model  # type: ignore[return-value]

    @property
    def spec(self) -> Spec:
        """Return the composed specification of every stage."""
        return Spec(
            name="table_pipeline",
            params={
                "steps": [step.spec.to_dict() for step in self._steps],
                "model": self._model.spec.to_dict(),
            },
        )

    def set_random_state(self, seed: int) -> None:
        """
        Seed every stochastic component of the pipeline.

        Propagates to each step and the classifier implementing
        :class:`~habit.domain.protocols.Seedable`; deterministic components
        are untouched (v1.0 naming decisions: one seeding verb, never a
        constructor parameter).
        """
        for component in [*self._steps, self._model]:
            if isinstance(component, Seedable):
                component.set_random_state(seed)

    def fit(
        self,
        table: FeatureTable,
        *,
        repeat_tables: Optional[Sequence[FeatureTable]] = None,
    ) -> "TablePipeline":
        """
        Fit every step in order, then the classifier.

        Each step is fitted on the table produced by the previous step, so
        learned statistics compose exactly as they will at predict time.
        Steps accepting ``repeat_tables`` (test-retest selectors) receive
        them; the rest see only the primary table.

        Args:
            table: Training table with feature columns and an outcome column.
            repeat_tables: Optional aligned repeat-measurement tables, passed
                only to the steps that consume them.

        Returns:
            ``self``, fitted.
        """
        current = table
        for step, takes_repeats in zip(self._steps, self._step_takes_repeats):
            if takes_repeats:
                step.fit(current, repeat_tables=repeat_tables)  # type: ignore[call-arg]
            else:
                step.fit(current)
            current = step.transform(current)
        self._model.fit(current)
        self._fit_output_columns = tuple(current.feature_columns)
        self._is_fitted = True
        return self

    def _check_fitted(self) -> None:
        """Raise when a transformation is requested before fitting."""
        if not self._is_fitted:
            raise HABITAPIError(
                "TablePipeline must be fitted before transform/predict."
            )

    def transform(self, table: FeatureTable) -> FeatureTable:
        """
        Apply the fitted transformation chain to a table.

        Args:
            table: Table carrying the feature columns seen at fit time
                (each fitted step validates its own input schema).

        Returns:
            The table after every fitted step, ready for the classifier.
        """
        self._check_fitted()
        current = table
        for step in self._steps:
            current = step.transform(current)
        return current

    def predict(self, table: FeatureTable) -> pd.Series:
        """
        Predict the terminal model's output for a table's rows.

        Class labels for a classifier, values for a regressor, risk scores
        for a survival model (routed through ``predict_risk``).

        Args:
            table: Table to predict; transformed with the fitted state first.

        Returns:
            Predictions indexed by the table's identifier columns.
        """
        transformed = self.transform(table)
        if isinstance(self._model, SurvivalModel):
            return self._model.predict_risk(transformed)
        return self._model.predict(transformed)

    def predict_proba(self, table: FeatureTable) -> pd.DataFrame:
        """
        Predict class probabilities for a table's rows.

        Only meaningful for a classifier terminal model; regressors and
        survival models have no class-probability output.

        Args:
            table: Table to predict; transformed with the fitted state first.

        Returns:
            Probability frame indexed by the identifier columns, one column
            per class.

        Raises:
            HABITAPIError: If the terminal model is not a classifier.
        """
        if not isinstance(self._model, Classifier):
            raise HABITAPIError(
                "TablePipeline.predict_proba requires a classifier terminal "
                f"model; this pipeline ends in a "
                f"{type(self._model).__name__}. Use predict() (values or "
                "risk) or predict_survival_function() instead."
            )
        return self._model.predict_proba(self.transform(table))

    def predict_survival_function(
        self, table: FeatureTable, times: np.ndarray
    ) -> pd.DataFrame:
        """
        Predict per-subject survival functions at the requested times.

        Args:
            table: Table to predict; transformed with the fitted state first.
            times: Ascending 1-D grid of evaluation times.

        Returns:
            Survival probabilities, one row per subject, one column per time.

        Raises:
            HABITAPIError: If the terminal model is not a survival model.
        """
        if not isinstance(self._model, SurvivalModel):
            raise HABITAPIError(
                "TablePipeline.predict_survival_function requires a survival "
                f"terminal model; this pipeline ends in a "
                f"{type(self._model).__name__}."
            )
        return self._model.predict_survival_function(self.transform(table), times)

    def evaluate(
        self,
        table: FeatureTable,
        metrics: Sequence[Union[Metric, RegressionMetric, SurvivalMetric]],
    ) -> Dict[str, float]:
        """
        Score the pipeline on a labelled table.

        Dispatches by the table's endpoint family:

        - **binary / multiclass** -- classification ``Metric`` objects;
          probability metrics receive the positive-class scores (column
          ``"1"`` for a 0/1 outcome, else the last class column).
        - **continuous** -- ``RegressionMetric`` objects on (true, predicted).
        - **survival** -- ``SurvivalMetric`` objects; risk-based metrics get
          ``predict_risk``, function-based ones get
          ``predict_survival_function`` evaluated on a grid derived from the
          follow-up range.

        Args:
            table: Evaluation table carrying the endpoint column(s).
            metrics: Metrics to compute, keyed in the result by
                ``metric.spec.name``. Must match the endpoint family.

        Returns:
            Mapping of metric name to value.

        Raises:
            HABITAPIError: If ``metrics`` is empty, the table has no
                endpoint, or a metric family does not match the endpoint.
        """
        if not metrics:
            raise HABITAPIError("TablePipeline.evaluate requires metrics.")
        if table.outcome is None:
            raise HABITAPIError(
                "TablePipeline.evaluate requires a table with an outcome; "
                "this table declares none."
            )
        task = table.outcome.task
        if task in ("binary", "multiclass"):
            return self._evaluate_classification(table, metrics)  # type: ignore[arg-type]
        if task == "continuous":
            return self._evaluate_regression(table, metrics)  # type: ignore[arg-type]
        if task == "survival":
            return self._evaluate_survival(table, metrics)  # type: ignore[arg-type]
        raise HABITAPIError(
            f"TablePipeline.evaluate does not know endpoint task {task!r}."
        )

    def _evaluate_classification(
        self, table: FeatureTable, metrics: Sequence[Metric]
    ) -> Dict[str, float]:
        """Classification branch of :meth:`evaluate`."""
        y_true = outcome_series(table, owner="TablePipeline.evaluate").to_numpy()
        y_pred = self.predict(table).to_numpy()
        needs_scores = any(metric.needs_proba for metric in metrics)
        scores: Optional[np.ndarray] = None
        if needs_scores:
            probability_frame = self.predict_proba(table)
            if probability_frame.shape[1] == 2:
                # Binary: the positive-class column ("1" for 0/1 outcomes).
                positive = "1" if "1" in probability_frame.columns else probability_frame.columns[-1]
                scores = probability_frame[positive].to_numpy(dtype=np.float64)
            else:
                scores = probability_frame.to_numpy(dtype=np.float64)
        results: Dict[str, float] = {}
        for metric in metrics:
            results[metric.spec.name] = metric(
                y_true, y_pred, scores if metric.needs_proba else None
            )
        return results

    def _evaluate_regression(
        self, table: FeatureTable, metrics: Sequence[RegressionMetric]
    ) -> Dict[str, float]:
        """Regression branch of :meth:`evaluate`."""
        for metric in metrics:
            if not isinstance(metric, RegressionMetric):
                raise HABITAPIError(
                    f"TablePipeline.evaluate: the table declares a continuous "
                    f"endpoint, but metric {metric.spec.name!r} "
                    f"({type(metric).__name__}) is not a regression metric. "
                    "Use the regression_metric registry (r2, mae, mse, rmse)."
                )
        y_true = outcome_series(table, owner="TablePipeline.evaluate").to_numpy()
        y_pred = self.predict(table).to_numpy()
        return {
            metric.spec.name: metric(y_true, y_pred)
            for metric in metrics
        }

    def _evaluate_survival(
        self, table: FeatureTable, metrics: Sequence[SurvivalMetric]
    ) -> Dict[str, float]:
        """Survival branch of :meth:`evaluate`."""
        for metric in metrics:
            if not isinstance(metric, SurvivalMetric):
                raise HABITAPIError(
                    f"TablePipeline.evaluate: the table declares a survival "
                    f"endpoint, but metric {metric.spec.name!r} "
                    f"({type(metric).__name__}) is not a survival metric. Use "
                    "the survival_metric registry (c_index, "
                    "integrated_brier_score, cumulative_dynamic_auc)."
                )
        time, event = survival_target(table, owner="TablePipeline.evaluate")
        time = time.to_numpy(dtype=np.float64)
        event = event.to_numpy(dtype=bool)
        risk: Optional[np.ndarray] = None
        probability: Optional[np.ndarray] = None
        grid: Optional[np.ndarray] = None
        results: Dict[str, float] = {}
        for metric in metrics:
            if metric.needs_survival_function:
                if probability is None:
                    # One shared grid inside the follow-up range for all
                    # function-based metrics of this evaluation.
                    event_times = time[event]
                    lower = float(event_times.min()) if event_times.size else float(time.min())
                    upper = float(time.max())
                    step = (upper - lower) / 101
                    grid = np.linspace(lower, upper - 0.5 * step, 100)
                    probability = self.predict_survival_function(table, grid).to_numpy()
                results[metric.spec.name] = metric(time, event, probability, times=grid)
            else:
                if risk is None:
                    risk = self.predict(table).to_numpy()
                results[metric.spec.name] = metric(time, event, risk)
        return results

    # -- persistence ----------------------------------------------------

    def save(self, path: Union[str, Path]) -> Path:
        """
        Persist the fitted pipeline in a versioned, self-describing format.

        The ``.habitpipeline`` file is a ZIP archive holding a JSON manifest
        (format name, format version, producing HABIT version, and every
        component's spec and class path) plus the pickled fitted state. The
        manifest keeps the artefact inspectable without deserialising it.

        Args:
            path: Destination file path.

        Returns:
            The path written.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        def _component_record(component: Any) -> Dict[str, Any]:
            cls = type(component)
            return {
                "class": f"{cls.__module__}.{cls.__qualname__}",
                "spec": component.spec.to_dict(),
            }

        manifest = {
            "format": _PIPELINE_FORMAT_NAME,
            "format_version": _PIPELINE_FORMAT_VERSION,
            "habit_version": _habit_version,
            "steps": [_component_record(step) for step in self._steps],
            "model": _component_record(self._model),
            "is_fitted": self._is_fitted,
            "fit_output_columns": list(self._fit_output_columns),
        }
        payload = {
            "steps": self._steps,
            "model": self._model,
            "is_fitted": self._is_fitted,
            "fit_output_columns": self._fit_output_columns,
        }
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(
                "manifest.json",
                json.dumps(manifest, indent=2, sort_keys=True),
            )
            zf.writestr("payload.pkl", pickle.dumps(payload))
        return destination

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TablePipeline":
        """
        Load a pipeline previously written by :meth:`save`.

        Security note: the fitted state is pickle-serialised (the standard
        serialisation for sklearn estimators), so only ever load pipeline
        files from sources you trust.

        Args:
            path: Source file path.

        Returns:
            The loaded pipeline, fitted exactly as when saved.

        Raises:
            CompatibilityError: If the file is not a HABIT table pipeline,
                was written with a newer format version, or its manifest
                does not match its payload.
        """
        source = Path(path)
        with zipfile.ZipFile(source, "r") as archive:
            try:
                manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
            except (KeyError, json.JSONDecodeError) as exc:
                raise CompatibilityError(
                    f"{source} is not a HABIT table pipeline file: {exc}"
                ) from exc
            if manifest.get("format") != _PIPELINE_FORMAT_NAME:
                raise CompatibilityError(
                    f"{source} has format {manifest.get('format')!r}; expected "
                    f"{_PIPELINE_FORMAT_NAME!r}."
                )
            file_version = int(manifest.get("format_version", 0))
            if file_version > _PIPELINE_FORMAT_VERSION:
                raise CompatibilityError(
                    f"{source} was written with format version {file_version}, "
                    f"but this HABIT (v{_habit_version}) reads up to version "
                    f"{_PIPELINE_FORMAT_VERSION}. Upgrade HABIT to load this "
                    "pipeline."
                )
            payload = pickle.loads(archive.read("payload.pkl"))
        pipeline = cls(steps=payload["steps"], model=payload["model"])
        pipeline._is_fitted = bool(payload["is_fitted"])
        pipeline._fit_output_columns = tuple(payload["fit_output_columns"])
        # Cross-check manifest against payload to catch archive corruption.
        manifest_names = [record["spec"]["name"] for record in manifest["steps"]]
        payload_names = [step.spec.name for step in pipeline._steps]
        if manifest_names != payload_names:
            raise CompatibilityError(
                f"{source} is internally inconsistent: manifest steps "
                f"{manifest_names} != payload steps {payload_names}."
            )
        return pipeline
