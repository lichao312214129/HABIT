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

from habit.api.exceptions import CompatibilityError, HABITAPIError
from habit.contracts.habitat import HabitatMap, Supervoxelization, VoxelFeatureField
from habit.contracts.subject import Subject
from habit.contracts.table import FeatureTable
from habit.domain.protocols import (
    HabitatAssigner,
    HabitatFeatureExtractor,
    Seedable,
    Supervoxelizer,
    VoxelFeatureExtractor,
)
from habit.domain.table_protocols import (
    Classifier,
    FeatureSelector,
    Metric,
    TablePreprocessor,
)
from habit._version import __version__ as _habit_version
from habit.spec.specs import Spec

__all__ = ["SubjectPipeline", "TablePipeline"]


def _voxel_units(field: VoxelFeatureField) -> Supervoxelization:
    """
    Wrap a voxel feature field as single-voxel clustering units.

    The one-step and direct-pooling designs cluster voxels directly, with no
    supervoxel step. Representing each voxel as a one-voxel
    ``Supervoxelization`` keeps the assigner contract uniform instead of
    giving assigners a second input type to handle.

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
            fitted model.
    """

    def __init__(
        self,
        voxel_feature_extractor: VoxelFeatureExtractor,
        supervoxelizer: Optional[Supervoxelizer],
        habitat_assigner: HabitatAssigner,
    ) -> None:
        if voxel_feature_extractor is None or habitat_assigner is None:
            raise HABITAPIError(
                "SubjectPipeline requires a voxel feature extractor and a "
                "habitat assigner; only the supervoxelizer may be None."
            )
        self.voxel_feature_extractor = voxel_feature_extractor
        self.supervoxelizer = supervoxelizer
        self.habitat_assigner = habitat_assigner

    @property
    def spec(self) -> Spec:
        """Return the composed specification of every stage."""
        stage_specs: Dict[str, Any] = {
            "voxel_feature_extractor": self.voxel_feature_extractor.spec.to_dict(),
            "supervoxelizer": (
                self.supervoxelizer.spec.to_dict()
                if self.supervoxelizer is not None
                else None
            ),
            "habitat_assigner": self.habitat_assigner.spec.to_dict(),
        }
        return Spec(name="subject_pipeline", params=stage_specs)

    def __call__(self, subject: Subject) -> HabitatMap:
        """
        Run voxel features, supervoxelisation and assignment for one subject.

        Args:
            subject: The subject to label.

        Returns:
            The subject's habitat label image.
        """
        field = self.voxel_feature_extractor(subject)
        if self.supervoxelizer is None:
            units = _voxel_units(field)
        else:
            units = self.supervoxelizer(field)
        return self.habitat_assigner(units)

    def extract_features(
        self,
        subject: Subject,
        extractors: Sequence[HabitatFeatureExtractor],
    ) -> FeatureTable:
        """
        Run the pipeline and then the requested habitat feature families.

        Named ``extract_features`` (an action) rather than the bare noun
        ``features``, which would read as an attribute on a callable object.

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
        habitat_map = self(subject)
        tables = [extractor(subject, habitat_map) for extractor in extractors]
        combined = tables[0]
        for table in tables[1:]:
            combined = combined.join(table)
        return combined


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
            the pipeline is the bare classifier.
        classifier: The terminal outcome model.
    """

    def __init__(
        self,
        steps: Sequence[Union[TablePreprocessor, FeatureSelector]],
        classifier: Classifier,
    ) -> None:
        if classifier is None:
            raise HABITAPIError("TablePipeline requires a classifier.")
        self._steps: List[Union[TablePreprocessor, FeatureSelector]] = list(steps)
        self._classifier = classifier
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
    def classifier(self) -> Classifier:
        """Return the terminal classifier."""
        return self._classifier

    @property
    def spec(self) -> Spec:
        """Return the composed specification of every stage."""
        return Spec(
            name="table_pipeline",
            params={
                "steps": [step.spec.to_dict() for step in self._steps],
                "classifier": self._classifier.spec.to_dict(),
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
        for component in [*self._steps, self._classifier]:
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
        self._classifier.fit(current)
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
        Predict class labels for a table's rows.

        Args:
            table: Table to predict; transformed with the fitted state first.

        Returns:
            Predicted labels indexed by the table's identifier columns.
        """
        return self._classifier.predict(self.transform(table))

    def predict_proba(self, table: FeatureTable) -> pd.DataFrame:
        """
        Predict class probabilities for a table's rows.

        Args:
            table: Table to predict; transformed with the fitted state first.

        Returns:
            Probability frame indexed by the identifier columns, one column
            per class.
        """
        return self._classifier.predict_proba(self.transform(table))

    def evaluate(
        self,
        table: FeatureTable,
        metrics: Sequence[Metric],
    ) -> Dict[str, float]:
        """
        Score the pipeline on a labelled table.

        Probability metrics receive the positive-class scores (column ``"1"``
        for a 0/1 outcome, else the last class column; multi-class problems
        pass the full probability frame, for which the binary calibration
        tests answer ``NaN``). Label metrics receive no scores.

        Args:
            table: Evaluation table carrying the outcome column.
            metrics: Metrics to compute, keyed in the result by
                ``metric.spec.name``.

        Returns:
            Mapping of metric name to value.

        Raises:
            HABITAPIError: If ``metrics`` is empty or the table declares no
                outcome column.
        """
        if not metrics:
            raise HABITAPIError("TablePipeline.evaluate requires metrics.")
        if table.outcome_column is None:
            raise HABITAPIError(
                "TablePipeline.evaluate requires a table with an outcome column."
            )
        y_true = table.frame[table.outcome_column].to_numpy()
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
            "classifier": _component_record(self._classifier),
            "is_fitted": self._is_fitted,
            "fit_output_columns": list(self._fit_output_columns),
        }
        payload = {
            "steps": self._steps,
            "classifier": self._classifier,
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
        pipeline = cls(steps=payload["steps"], classifier=payload["classifier"])
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
