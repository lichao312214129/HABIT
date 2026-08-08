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

import json
import pickle
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline as SkPipeline

from habit.exceptions import CompatibilityError, HABITAPIError
from habit.contracts.habitat import HabitatMap, Supervoxelization, VoxelFeatureField
from habit.contracts.subject import Subject
from habit.contracts.table import FeatureTable
from habit.domain.geometry_align import (
    ON_GEOMETRY_MISMATCH_DEFAULT,
    align_subject_masks,
    coerce_on_geometry_mismatch,
)
from habit.domain.outcome_access import outcome_series, survival_target
from habit.domain.postprocess import ConnectedComponentPostprocess
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
from habit.domain.sklearn_interop import (
    FrameToTable,
    as_outcome_model,
    as_transformer,
    step_consumes_repeat_tables,
    wraps_outcome_model,
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
from habit.utils.log_utils import get_module_logger

__all__ = ["SubjectPipeline", "TablePipeline", "voxel_units"]

_logger = get_module_logger(__name__)


def _is_feature_selector_step(step: object) -> bool:
    """
    Return True when ``step`` is a registered feature selector.

    ``FeatureSelector`` and ``TablePreprocessor`` are structurally similar
    ``@runtime_checkable`` protocols, so ``isinstance(..., FeatureSelector)``
    is not reliable. Registry membership on ``step.spec.name`` is the
    definitive check used by assembly.
    """
    from habit.domain.feature_selection.registry import FeatureSelectorRegistry

    try:
        name = step.spec.name  # type: ignore[attr-defined]
    except AttributeError:
        return False
    return FeatureSelectorRegistry.get(str(name)) is not None


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
        on_geometry_mismatch: How to handle image/mask grid disagreements
            before Stage-1. ``"resample_mask"`` (default) nearest-neighbour
            resamples each ROI onto the first image modality; ``"strict"``
            raises :class:`~habit.exceptions.GeometryError`.
        postprocess_supervoxel: Optional connected-component cleanup applied
            immediately after supervoxelization and before supervoxel feature
            extraction. Ignored when ``supervoxelizer`` is ``None``.
        postprocess_habitat: Optional connected-component cleanup applied
            immediately after habitat assignment and before habitat features.
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
        on_geometry_mismatch: str = ON_GEOMETRY_MISMATCH_DEFAULT,
        postprocess_supervoxel: Optional[ConnectedComponentPostprocess] = None,
        postprocess_habitat: Optional[ConnectedComponentPostprocess] = None,
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
        if postprocess_supervoxel is not None and supervoxelizer is None:
            _logger.warning(
                "SubjectPipeline received postprocess_supervoxel but no "
                "supervoxelizer; supervoxel cleanup is ignored for direct "
                "voxel clustering designs."
            )
            postprocess_supervoxel = None
        self.voxel_feature_extractor = voxel_feature_extractor
        self.supervoxelizer = supervoxelizer
        self.habitat_assigner = habitat_assigner
        self.supervoxel_feature_extractor = supervoxel_feature_extractor
        self.voxel_feature_preprocessor = voxel_feature_preprocessor
        self.supervoxel_feature_preprocessor = supervoxel_feature_preprocessor
        self.cohort_feature_preprocessor = cohort_feature_preprocessor
        self.on_geometry_mismatch = coerce_on_geometry_mismatch(on_geometry_mismatch)
        self.postprocess_supervoxel = postprocess_supervoxel
        self.postprocess_habitat = postprocess_habitat

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
            "on_geometry_mismatch": self.on_geometry_mismatch,
            "postprocess_supervoxel": _optional(self.postprocess_supervoxel),
            "postprocess_habitat": _optional(self.postprocess_habitat),
        }
        return Spec(name="subject_pipeline", params=stage_specs)

    def _prepare_subject(self, subject: Subject) -> Subject:
        """
        Align ROI masks onto the reference image grid when needed.

        Args:
            subject: Incoming subject, possibly with drifted mask geometry.

        Returns:
            A subject whose masks share the reference image voxel grid under
            the configured ``on_geometry_mismatch`` policy.
        """
        return align_subject_masks(
            subject,
            on_geometry_mismatch=self.on_geometry_mismatch,
        )

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
        subject = self._prepare_subject(subject)
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
        if self.postprocess_supervoxel is not None:
            # Clean fragments before describing regions so features and the
            # label map stay aligned for cohort fitting / assignment.
            units = self.postprocess_supervoxel.apply_to_supervoxelization(
                units, field
            )
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
        habitat_map = self.habitat_assigner(working)
        if self.postprocess_habitat is not None:
            habitat_map = self.postprocess_habitat.apply_to_habitat_map(habitat_map)
        return habitat_map, working

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

#: On-disk format identifier and the version this HABIT build WRITES.
#: ``format_version`` 1 files were produced before ``TablePipeline`` became an
#: ``sklearn.pipeline.Pipeline`` subclass; version 2 adds the frame schema of
#: the :class:`~habit.domain.sklearn_interop.FrameToTable` head step. Both are
#: readable -- see :meth:`TablePipeline.load`.
_PIPELINE_FORMAT_NAME = "habit.tablepipeline"
_PIPELINE_FORMAT_VERSION = 2

#: Step name of the ``FrameToTable`` head and of the terminal outcome model.
#: Fixed (rather than derived) so ``pipe.set_params(frame_to_table=...)`` and
#: a ``param_grid`` key like ``"model__component__C"`` are stable, documentable
#: strings rather than something a caller has to discover per pipeline.
_HEAD_STEP_NAME = "frame_to_table"
_MODEL_STEP_NAME = "model"


def _sklearn_pipeline_param_names() -> Tuple[str, ...]:
    """
    Return the constructor parameter names of ``sklearn.pipeline.Pipeline``.

    Read off sklearn itself rather than hard-coded, because the set has grown
    across the supported range (``transform_input`` arrived in 1.6 and HABIT
    supports ``scikit-learn>=1.4,<2``). Anything sklearn declares is a
    parameter ``TablePipeline`` must expose unchanged, and a step name must
    never collide with one.

    Returns:
        Tuple[str, ...]: Parameter names, sorted as sklearn sorts them.
    """
    return tuple(SkPipeline._get_param_names())


def _is_sklearn_step_list(steps: Any) -> bool:
    """
    Report whether ``steps`` is already in scikit-learn ``(name, est)`` form.

    ``TablePipeline`` accepts two shapes for the same argument, and the shape
    decides how it is interpreted:

    * HABIT form -- a flat sequence of HABIT components, which is what call
      sites and YAML-driven assembly pass, and which this class wraps into
      adapters;
    * sklearn form -- ``[(name, estimator), ...]``, which is what
      ``sklearn.base.clone``, ``Pipeline.set_params(steps=...)`` and pipeline
      slicing pass back in, and which must be stored VERBATIM (``clone``
      verifies that ``get_params()`` returns the very object it handed to the
      constructor).

    Args:
        steps: The ``steps`` argument as received.

    Returns:
        bool: ``True`` for sklearn form. An empty sequence is reported as
        HABIT form, so the "no terminal model" error still fires for
        ``TablePipeline(steps=[])``.
    """
    if not isinstance(steps, (list, tuple)) or not steps:
        return False
    return all(
        isinstance(step, tuple) and len(step) == 2 and isinstance(step[0], str)
        for step in steps
    )


def _unique_step_name(component: Any, taken: set) -> str:
    """
    Derive a stable, unique sklearn step name for a HABIT component.

    The component's registered spec name is the name a user already knows
    (``"zscore"``, ``"variance"``, ``"lasso"``), so it is what a
    ``param_grid`` key should read as. Uniqueness and sklearn's own naming
    rules are enforced on top: names must be distinct, must not contain
    ``"__"`` (the parameter separator) and must not collide with a
    constructor parameter of the pipeline.

    Args:
        component: The HABIT component about to be wrapped.
        taken: Names already used; mutated to include the returned name.

    Returns:
        str: The step name.
    """
    try:
        base = str(component.spec.name)
    except AttributeError:
        base = type(component).__name__
    base = base.replace("__", "_") or type(component).__name__
    name = base
    suffix = 2
    while name in taken:
        name = f"{base}_{suffix}"
        suffix += 1
    taken.add(name)
    return name


def _component_spec_payload(component: Any) -> Dict[str, Any]:
    """
    Return one component's ``Spec`` payload for the composed pipeline spec.

    Args:
        component: A HABIT component taken from the pipeline.

    Returns:
        Dict[str, Any]: The component's ``spec.to_dict()``.

    Raises:
        HABITAPIError: When the object carries no ``Spec``. A pipeline holding
            a foreign estimator cannot be described, fingerprinted or saved,
            and failing here is the only way that stays visible instead of
            reappearing as a provenance record nobody can reproduce.
    """
    spec = getattr(component, "spec", None)
    if spec is None or not hasattr(spec, "to_dict"):
        raise HABITAPIError(
            f"TablePipeline.spec needs every step to carry a habit Spec, but "
            f"{type(component).__name__} carries none. Only HABIT components "
            "(and a FrameToTable head) belong in a TablePipeline; a foreign "
            "scikit-learn estimator cannot be fingerprinted or saved."
        )
    return spec.to_dict()


def _build_sklearn_steps(
    components: Sequence[Any],
    model: Any,
) -> List[Tuple[str, Any]]:
    """
    Wrap HABIT components into the ``(name, estimator)`` list sklearn needs.

    Layout, always: a :class:`~habit.domain.sklearn_interop.FrameToTable` head
    (so an sklearn cross-validation driver can hand the pipeline a plain,
    row-sliceable frame), then one
    :class:`~habit.domain.sklearn_interop.TableTransformerEstimator` per
    transformation component, then the terminal outcome-model adapter.

    Every adapter is built with ``copy_on_fit=False``: the pipeline's
    :attr:`TablePipeline.components`, its :meth:`TablePipeline.save` artefact
    and every reporting call site read fitted state off the very component
    objects the caller constructed, so the pipeline must fit them in place.
    ``sklearn.base.clone`` still gives each cross-validation fold its own
    components, so nothing leaks across folds.

    Args:
        components: HABIT transformation components in execution order. A
            ``FrameToTable`` may be given as the FIRST element to declare the
            column schema of the frames an sklearn driver will slice; anywhere
            else it is an error, because rebuilding the table halfway through
            a chain would silently discard the upstream selection.
        model: The terminal outcome model.

    Returns:
        List[Tuple[str, Any]]: The sklearn step list.

    Raises:
        HABITAPIError: On a misplaced ``FrameToTable``.
    """
    items = list(components)
    head: FrameToTable = FrameToTable()
    if items and isinstance(items[0], FrameToTable):
        head = items.pop(0)
    for item in items:
        if isinstance(item, FrameToTable):
            raise HABITAPIError(
                "A FrameToTable step rebuilds the FeatureTable from a plain "
                "frame, so it only makes sense at the HEAD of a "
                "TablePipeline; found one after another step, where it would "
                "silently discard the upstream selection. Move it to the "
                "front of the steps list."
            )
    prepared: List[Tuple[str, Any]] = [(_HEAD_STEP_NAME, head)]
    taken = {_HEAD_STEP_NAME, _MODEL_STEP_NAME, *_sklearn_pipeline_param_names()}
    selector_position = 0
    for component in items:
        name = _unique_step_name(component, taken)
        selector_index: Optional[int] = None
        if _is_feature_selector_step(component):
            selector_position += 1
            selector_index = selector_position
        prepared.append(
            (
                name,
                as_transformer(
                    component,
                    copy_on_fit=False,
                    selector_step_index=selector_index,
                ),
            )
        )
    prepared.append((_MODEL_STEP_NAME, as_outcome_model(model, copy_on_fit=False)))
    return prepared


class TablePipeline(SkPipeline):
    """
    Fitted preprocessing/selection chain plus model over feature tables.

    The structural answer to the train/predict leakage class of bugs: the
    preprocessing and feature-selection steps are fitted ONCE on the training
    table and their fitted state is what ``predict``/``transform`` apply to
    any later table -- the prediction data is normalised with the TRAINING
    statistics and reduced with the TRAINING selection, never re-fitted.

    The fitted pipeline is also the artefact a study publishes for external
    validation of its tabular model, which is why :meth:`save` persists the
    steps and the model together in one versioned, self-describing file (a
    JSON manifest recording every component's
    :class:`~habit.spec.specs.Spec` alongside the pickled fitted state).

    **This IS an ``sklearn.pipeline.Pipeline``.** Subclassing rather than
    re-implementing composition is what gives HABIT's tabular models
    ``clone``, ``get_params``/``set_params``, nested parameter addressing and
    therefore ``GridSearchCV`` / ``RandomizedSearchCV`` / ``cross_val_score``
    for free, instead of a second composition engine that would drift from
    the one the rest of the ecosystem uses.

    Two consequences a caller must know:

    * ``.steps`` has sklearn's meaning -- ``List[Tuple[str, estimator]]``,
      where the estimators are the interop adapters. It is NOT overridden,
      because sklearn's ``_iter`` / ``_validate_steps`` / ``get_params`` /
      ``set_params`` all read and WRITE it directly. The HABIT components are
      reached through :attr:`components` (transformation steps) and
      :attr:`model` (the terminal one).
    * The step list always begins with a
      :class:`~habit.domain.sklearn_interop.FrameToTable` head named
      ``"frame_to_table"`` and ends with the outcome-model adapter named
      ``"model"``. The head is what lets an sklearn cross-validation driver
      pass a plain frame as ``X`` (a ``FeatureTable`` is a frozen dataclass
      and deliberately not row-sliceable); when the pipeline is handed a
      ``FeatureTable`` directly -- HABIT's own entry point -- it passes
      straight through, with no frame round-trip and therefore no dtype
      promotion that could shift a later z-score.

    HABIT's verbs are kept as overrides, because a ``FeatureTable`` in must
    give a HABIT type out: :meth:`fit`, :meth:`transform`, :meth:`predict`
    and :meth:`predict_proba` return tables / labelled Series / labelled
    frames for ``FeatureTable`` input, and plain arrays for frame input (what
    an sklearn scorer expects). :meth:`evaluate`,
    :meth:`predict_survival_function`, :meth:`set_random_state`, :meth:`spec`,
    :meth:`save` and :meth:`load` have no sklearn equivalent and are
    unchanged.

    Args:
        steps: Either the HABIT form -- ordered transformation components
            (``TablePreprocessor`` and/or ``FeatureSelector``
            implementations), optionally preceded by a ``FrameToTable``
            declaring the frame schema; may be empty, in which case the
            pipeline is the bare model -- or the sklearn form
            ``[(name, estimator), ...]``, which is what ``clone``,
            ``set_params(steps=...)`` and slicing pass back in.
        model: The terminal outcome model -- a :class:`Classifier`,
            :class:`Regressor`, or :class:`SurvivalModel`, matched to the
            endpoint family of the tables it will be fitted on. Must be
            omitted when ``steps`` is already in sklearn form (the terminal
            step carries it).
        classifier: Deprecated alias for ``model`` (binary/multiclass
            endpoints); kept so existing call sites keep working.
        **pipeline_options: Forwarded verbatim to
            ``sklearn.pipeline.Pipeline`` (``memory``, ``verbose``, and
            ``transform_input`` on scikit-learn >= 1.6).

    Examples:
        Nested hyperparameter search over a HABIT component's own parameter::

            from sklearn.model_selection import GridSearchCV

            from habit.domain.sklearn_interop import FrameToTable

            pipe = TablePipeline(
                steps=[FrameToTable.from_table(train), ZScorePreprocessor()],
                model=LogisticRegressionClassifier(),
            )
            search = GridSearchCV(pipe, {"model__component__C": [0.1, 1, 10]})
            search.fit(train.frame, outcome_series(train))
    """

    def __init__(
        self,
        steps: Sequence[Any],
        model: Optional[Union[Classifier, Regressor, SurvivalModel]] = None,
        *,
        classifier: Optional[Classifier] = None,
        **pipeline_options: Any,
    ) -> None:
        if _is_sklearn_step_list(steps):
            if model is not None or classifier is not None:
                raise HABITAPIError(
                    "TablePipeline received steps already in scikit-learn "
                    "(name, estimator) form together with a separate model. "
                    "The terminal step already carries the model; passing it "
                    "twice would leave two divergent copies, one of which "
                    "would never be fitted."
                )
            # Stored verbatim: ``sklearn.base.clone`` checks that
            # ``get_params(deep=False)["steps"]`` IS the list it passed in.
            prepared: List[Tuple[str, Any]] = steps  # type: ignore[assignment]
        else:
            if model is None and classifier is not None:
                model = classifier
            if model is None:
                raise HABITAPIError("TablePipeline requires a terminal model.")
            prepared = _build_sklearn_steps(steps, model)
        super().__init__(prepared, **pipeline_options)

    @classmethod
    def _get_param_names(cls) -> List[str]:
        """
        Declare exactly ``sklearn.pipeline.Pipeline``'s constructor parameters.

        ``BaseEstimator._get_param_names`` derives the parameter set from the
        subclass's own ``__init__``, which here would add ``model`` and
        ``classifier``. Both are CONSTRUCTION conveniences, not state: the
        terminal model lives inside ``steps``. Reporting them would make
        ``clone`` rebuild the model twice -- once inside the cloned steps and
        once from the ``model`` parameter -- leaving the pipeline with a
        terminal model that never gets fitted, and ``clone``'s own identity
        check would fail. Delegating to sklearn's own answer also keeps the
        set correct across the supported scikit-learn range.

        Returns:
            List[str]: Parameter names, in sklearn's sorted order.
        """
        return list(_sklearn_pipeline_param_names())

    # -- HABIT views over the sklearn step list ---------------------------

    @property
    def components(self) -> Tuple[Union[TablePreprocessor, FeatureSelector], ...]:
        """
        Return the ordered HABIT transformation components.

        This is the successor of the pre-v1.1 ``.steps`` property. ``.steps``
        now means what sklearn means by it, so the HABIT components -- the
        objects carrying ``spec``, the fitted statistics and the selected
        column names -- are read here instead. The ``FrameToTable`` head and
        the terminal model adapter are excluded: the head is interop
        plumbing, and the model is :attr:`model`.

        Returns:
            Tuple: The unwrapped components, in execution order.
        """
        collected: List[Any] = []
        for _, estimator in self.steps:
            if isinstance(estimator, FrameToTable):
                continue
            if wraps_outcome_model(estimator):
                continue
            collected.append(getattr(estimator, "component", estimator))
        return tuple(collected)

    @property
    def model(self) -> Union[Classifier, Regressor, SurvivalModel]:
        """
        Return the terminal outcome model.

        Returns:
            The HABIT ``Classifier`` / ``Regressor`` / ``SurvivalModel``.

        Raises:
            HABITAPIError: When the pipeline does not end in a HABIT outcome
                model -- which happens for a slice such as ``pipe[:-1]``.
        """
        terminal = self.steps[-1][1] if self.steps else None
        component = getattr(terminal, "component", None)
        if not wraps_outcome_model(terminal) or component is None:
            raise HABITAPIError(
                "This TablePipeline does not end in a HABIT outcome model "
                f"(terminal step: {type(terminal).__name__}), so it has no "
                "model to report. Slices such as pipe[:-1] are transformation "
                "chains only."
            )
        return component

    @property
    def classifier(self) -> Classifier:
        """Return the terminal model, asserted to be a classifier."""
        return self.model  # type: ignore[return-value]

    @property
    def frame_schema(self) -> FrameToTable:
        """
        Return the ``FrameToTable`` head step.

        The one place to read or re-declare the column schema an sklearn
        driver's frames follow. Prefer
        ``pipe.set_params(frame_to_table=FrameToTable.from_table(table))`` to
        replace it, so the change goes through sklearn's own parameter
        machinery and survives ``clone``.

        Raises:
            HABITAPIError: When the pipeline has no ``FrameToTable`` head.
        """
        head = self.steps[0][1] if self.steps else None
        if not isinstance(head, FrameToTable):
            raise HABITAPIError(
                "This TablePipeline has no FrameToTable head step (first "
                f"step: {type(head).__name__}), so it cannot accept a plain "
                "frame as X."
            )
        return head

    @property
    def spec(self) -> Spec:
        """
        Return the composed specification of every stage.

        The payload shape -- ``{"steps": [...], "model": {...}}`` -- is the
        FINGERPRINTED one and is deliberately unchanged by the move to
        ``sklearn.pipeline.Pipeline``: the ``FrameToTable`` head and the
        adapters are interop plumbing, not scientific definition, so they do
        not appear. Renaming or reshaping this payload would move every
        recorded provenance fingerprint in the repository.

        Raises:
            HABITAPIError: When a step carries no ``Spec``, i.e. the pipeline
                holds a foreign estimator that HABIT cannot describe.
        """
        return Spec(
            name="table_pipeline",
            params={
                "steps": [
                    _component_spec_payload(component)
                    for component in self.components
                ],
                "model": _component_spec_payload(self.model),
            },
        )

    def set_random_state(self, seed: int) -> None:
        """
        Seed every stochastic component of the pipeline.

        Propagates to each transformation component and the terminal model
        implementing :class:`~habit.domain.protocols.Seedable`; deterministic
        components are untouched (v1.0 naming decisions: one seeding verb,
        never a constructor parameter).

        Args:
            seed: The seed to install.
        """
        for component in [*self.components, *self._terminal_components()]:
            if isinstance(component, Seedable):
                component.set_random_state(seed)

    def _terminal_components(self) -> Tuple[Any, ...]:
        """Return the terminal model in a tuple, or empty when absent."""
        try:
            return (self.model,)
        except HABITAPIError:
            return ()

    # -- fit / transform / predict ---------------------------------------

    def fit(
        self,
        X: Any,
        y: Any = None,
        *,
        repeat_tables: Optional[Sequence[FeatureTable]] = None,
        **params: Any,
    ) -> "TablePipeline":
        """
        Fit every step in order, then the terminal model.

        Each step is fitted on the table produced by the previous step, so
        learned statistics compose exactly as they will at predict time.

        Args:
            X: Training data. A ``FeatureTable`` (HABIT's entry point: the
                outcome rides inside it and passes through the head step
                untouched) or a plain frame carrying the identifier, feature
                and outcome columns the ``FrameToTable`` head declares.
            y: Training targets. Normally ``None`` for a ``FeatureTable``,
                whose outcome column already supplies them; sklearn's
                cross-validation drivers pass the sliced label array, which
                is cross-checked against the table's own outcome so a
                misaligned ``y`` fails loudly.
            repeat_tables: Optional aligned repeat-measurement tables. Routed
                ONLY to the steps whose ``fit`` declares ``repeat_tables``
                (the test-retest / ICC selectors); the rest never see the
                keyword.
            **params: scikit-learn step-scoped fit parameters in
                ``stepname__param`` form, forwarded unchanged.

        Returns:
            ``self``, fitted.
        """
        routed = dict(params)
        if repeat_tables is not None:
            routed.update(self._repeat_table_params(repeat_tables))
        super().fit(X, y, **routed)
        return self

    def _repeat_table_params(
        self, repeat_tables: Sequence[FeatureTable]
    ) -> Dict[str, Any]:
        """
        Expand ``repeat_tables`` into scikit-learn's step-scoped fit params.

        Test-retest selectors learn from aligned repeat-measurement tables;
        every other step must NOT be handed the keyword, or it would raise on
        an unexpected argument. sklearn's routing key is
        ``"<stepname>__repeat_tables"``, so the pipeline resolves which steps
        consume it and names them explicitly.

        Args:
            repeat_tables: The repeat tables to route.

        Returns:
            Dict[str, Any]: Fit parameters, one entry per consuming step.
            Empty when no step consumes them, which leaves the tables unused
            exactly as before -- declaring repeats for a chain that has no
            ICC selector is a configuration statement, not an error.
        """
        routed: Dict[str, Any] = {}
        for name, estimator in self.steps:
            target = getattr(estimator, "component", estimator)
            if step_consumes_repeat_tables(target):
                routed[f"{name}__repeat_tables"] = repeat_tables
        return routed

    def _check_fitted(self) -> None:
        """
        Raise a HABIT error when a transformation is requested before fitting.

        Deliberately raises :class:`~habit.exceptions.HABITAPIError` rather
        than sklearn's ``NotFittedError``: this is the error HABIT's own
        callers have always caught on this class, and the sklearn-facing
        methods inherited from ``Pipeline`` still raise ``NotFittedError``
        through their own guard.
        """
        if not self.__sklearn_is_fitted__():
            raise HABITAPIError(
                "TablePipeline must be fitted before transform/predict."
            )

    def _transformed(self, X: Any) -> FeatureTable:
        """
        Run the transformation chain (every step but the terminal model).

        Args:
            X: A ``FeatureTable`` or a plain frame.

        Returns:
            FeatureTable: The table the terminal model consumes.
        """
        self._check_fitted()
        current: Any = X
        for _, _, step in self._iter(with_final=False):
            current = step.transform(current)
        return current

    def transform(self, X: Any, **params: Any) -> FeatureTable:  # type: ignore[override]
        """
        Apply the fitted transformation chain to a table.

        Deliberately narrower than ``sklearn.pipeline.Pipeline.transform``,
        which also calls the FINAL step: a ``TablePipeline`` always ends in an
        outcome model, which has no ``transform``, so sklearn's version is
        never available on this class anyway. What callers have always meant
        by ``pipeline.transform(table)`` -- "the classifier-ready table" -- is
        what this returns.

        Args:
            X: Table (or frame) carrying the feature columns seen at fit time;
                each fitted step validates its own input schema.
            **params: Accepted for signature compatibility; must be empty.

        Returns:
            FeatureTable: The table after every fitted transformation step.

        Raises:
            HABITAPIError: If the pipeline is not fitted, or ``params`` is
                non-empty (per-step transform parameters would silently do
                nothing here).
        """
        if params:
            raise HABITAPIError(
                "TablePipeline.transform takes no step parameters; got "
                f"{sorted(params)}."
            )
        return self._transformed(X)

    def predict(self, X: Any, **params: Any) -> Any:  # type: ignore[override]
        """
        Predict the terminal model's output for a table's rows.

        Class labels for a classifier, values for a regressor, risk scores
        for a survival model (routed through ``predict_risk``).

        Args:
            X: Data to predict; transformed with the fitted state first.
            **params: scikit-learn predict parameters, forwarded to the
                terminal adapter on the array path only.

        Returns:
            ``pd.Series`` indexed by the table's identifier columns for
            ``FeatureTable`` input (HABIT semantics), or a plain ``ndarray``
            for frame input, which is what an sklearn scorer expects.
        """
        transformed = self._transformed(X)
        if not isinstance(X, FeatureTable):
            return self.steps[-1][1].predict(transformed, **params)
        model = self.model
        if isinstance(model, SurvivalModel):
            return model.predict_risk(transformed)
        return model.predict(transformed)

    def predict_proba(self, X: Any, **params: Any) -> Any:  # type: ignore[override]
        """
        Predict class probabilities for a table's rows.

        Only meaningful for a classifier terminal model; regressors and
        survival models have no class-probability output.

        Args:
            X: Data to predict; transformed with the fitted state first.
            **params: scikit-learn predict parameters, forwarded to the
                terminal adapter on the array path only.

        Returns:
            A probability frame indexed by the identifier columns, one column
            per class, for ``FeatureTable`` input; a plain ``ndarray`` with
            columns aligned to ``self.classes_`` for frame input.

        Raises:
            HABITAPIError: If the terminal model is not a classifier.
        """
        model = self.model
        if not isinstance(model, Classifier):
            raise HABITAPIError(
                "TablePipeline.predict_proba requires a classifier terminal "
                f"model; this pipeline ends in a "
                f"{type(model).__name__}. Use predict() (values or "
                "risk) or predict_survival_function() instead."
            )
        transformed = self._transformed(X)
        if not isinstance(X, FeatureTable):
            return self.steps[-1][1].predict_proba(transformed, **params)
        return model.predict_proba(transformed)

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
        model = self.model
        if not isinstance(model, SurvivalModel):
            raise HABITAPIError(
                "TablePipeline.predict_survival_function requires a survival "
                f"terminal model; this pipeline ends in a "
                f"{type(model).__name__}."
            )
        return model.predict_survival_function(self._transformed(table), times)

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

    def _fit_output_columns(self) -> Tuple[str, ...]:
        """
        Return the feature columns the terminal model was fitted on.

        Recorded by the terminal adapter at fit time (see
        the terminal outcome-model adapter's ``fit``), which is the only place the
        output of the whole transformation chain is observable without
        re-running it.

        Returns:
            Tuple[str, ...]: The fit-time feature block, empty when unfitted.
        """
        return tuple(getattr(self.steps[-1][1], "feature_columns_", ()))

    def save(self, path: Union[str, Path]) -> Path:
        """
        Persist the fitted pipeline in a versioned, self-describing format.

        The ``.habitpipeline`` file is a ZIP archive holding a JSON manifest
        (format name, format version, producing HABIT version, and every
        component's spec and class path) plus the pickled fitted state. The
        manifest keeps the artefact inspectable without deserialising it.

        What is pickled is the HABIT COMPONENTS, not the sklearn adapters
        wrapping them: the components are the fitted science, the adapters are
        interop plumbing that :meth:`load` rebuilds. Keeping the payload at
        the component level is also what lets one loader read both format
        version 1 (written before this class became an
        ``sklearn.pipeline.Pipeline``) and version 2.

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

        components = list(self.components)
        model = self.model
        is_fitted = self.__sklearn_is_fitted__()
        head = self.steps[0][1] if isinstance(self.steps[0][1], FrameToTable) else None
        manifest = {
            "format": _PIPELINE_FORMAT_NAME,
            "format_version": _PIPELINE_FORMAT_VERSION,
            "habit_version": _habit_version,
            "steps": [_component_record(step) for step in components],
            "model": _component_record(model),
            "is_fitted": is_fitted,
            "fit_output_columns": list(self._fit_output_columns()),
            # Version 2 additions. Recorded in the manifest as well as the
            # payload so the artefact stays inspectable without unpickling.
            "step_names": [name for name, _ in self.steps],
            "declares_frame_schema": bool(head is not None and head.declares_schema),
        }
        payload = {
            "steps": components,
            "model": model,
            "is_fitted": is_fitted,
            "fit_output_columns": self._fit_output_columns(),
            "frame_schema": head,
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

        Both on-disk format versions are readable:

        * **version 1** -- written before this class became an
          ``sklearn.pipeline.Pipeline``. It carries HABIT components and no
          frame schema, so the loader rebuilds the adapters and gives the
          pipeline a schema-less ``FrameToTable`` head. Such a pipeline
          behaves exactly as it did when saved for every ``FeatureTable``
          call; only the "hand it a plain frame" entry point needs a schema
          declared afterwards.
        * **version 2** -- additionally carries the head's frame schema.

        A file whose ``format_version`` this build does not know raises rather
        than being guessed at: silently loading a wrong-but-plausible pipeline
        would produce numbers nobody could trace.

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
        # ``frame_schema`` is absent in version 1 files; a schema-less head
        # still passes FeatureTables through unchanged, which is every path a
        # version 1 pipeline ever had.
        head = payload.get("frame_schema") or FrameToTable()
        pipeline = cls(
            steps=[head, *payload["steps"]],
            model=payload["model"],
        )
        if bool(payload["is_fitted"]):
            pipeline._adopt_fitted_components(
                tuple(payload["fit_output_columns"])
            )
        # Cross-check manifest against payload to catch archive corruption.
        manifest_names = [record["spec"]["name"] for record in manifest["steps"]]
        payload_names = [step.spec.name for step in pipeline.components]
        if manifest_names != payload_names:
            raise CompatibilityError(
                f"{source} is internally inconsistent: manifest steps "
                f"{manifest_names} != payload steps {payload_names}."
            )
        return pipeline

    def _adopt_fitted_components(self, fit_output_columns: Tuple[str, ...]) -> None:
        """
        Mark the freshly built adapters as fitted around already-fitted parts.

        A loaded pipeline's components carry their fitted state (that is what
        was pickled), but the adapters wrapping them were constructed empty.
        scikit-learn decides fitted-ness by looking for trailing-underscore
        attributes on the estimator, so the adapters must adopt their
        component explicitly -- otherwise ``predict`` on a loaded pipeline
        would raise "not fitted" while sitting on a perfectly fitted model.

        Args:
            fit_output_columns: The feature block the terminal model was
                trained on, as recorded in the file.
        """
        for _, estimator in self.steps:
            if isinstance(estimator, FrameToTable):
                continue
            estimator.component_ = estimator.component
        terminal = self.steps[-1][1]
        terminal.feature_columns_ = fit_output_columns
        model = self.model
        if isinstance(model, Classifier):
            # Class labels are re-read from the fitted classifier rather than
            # stored, so a file can never disagree with the model inside it.
            classes = getattr(model, "_classes", None)
            if classes is not None:
                terminal.proba_columns_ = tuple(str(label) for label in classes)
                terminal.classes_ = np.asarray(classes)
