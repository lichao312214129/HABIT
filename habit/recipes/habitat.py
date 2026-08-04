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
"""The three habitat designs, as assembly functions (L4).

Each function here is a WIRING DIAGRAM, not an engine: it reads a
:class:`~habit.spec.specs.HabitatSpec`, builds the declared components, runs
them through :class:`~habit.domain.pipeline.SubjectPipeline` and
:meth:`~habit.contracts.subject.Cohort.map`, and packs the outcome into a
:class:`~habit.recipes.result.StudyResult`. Everything that looks like
orchestration -- parallelism, resume, per-subject failure policy, progress
reporting -- belongs to the execution backend, and everything that looks like
persistence belongs to a :class:`~habit.contracts.ops.ResultWriter`. That
division is deliberate: v0.1's habitat analysis grew into a 2000-line
orchestrator precisely because those concerns were allowed to accumulate in
the same object as the algorithm wiring.

The three designs differ only in WHERE the habitat definition is learned:

* :func:`two_step` -- supervoxels per subject, habitats across the cohort.
* :func:`direct_pooling` -- no supervoxels; habitats across the cohort's
  pooled voxels.
* :func:`one_step` -- no supervoxels; habitats defined INSIDE each subject,
  independently. There is no cohort-level definition, and the habitat ids of
  two subjects are not comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional, Sequence, Tuple

import pandas as pd

from habit.exceptions import HABITAPIError
from habit.contracts.habitat import HabitatMap, HabitatModel
from habit.contracts.manifest import RunManifest
from habit.contracts.ops import ExecutionBackend
from habit.contracts.provenance import Provenance, software_fingerprint
from habit.contracts.subject import Cohort, Subject
from habit.contracts.table import FeatureTable
from habit.domain.assembly import HabitatComponents, build_habitat_components
from habit.domain.protocols import Seedable
from habit.recipes.result import StudyResult
from habit.spec.specs import HabitatSpec

if TYPE_CHECKING:
    # Typing-only reference: the store is an execution-layer concern and is
    # only ever passed through to ``Cohort.map``, never opened here.
    from habit.execution.checkpoint import CheckpointStore

__all__ = ["two_step", "one_step", "direct_pooling", "apply_habitat_model"]

#: Identifier column of the cohort feature table, matching the habitat
#: feature families (``habit.domain.habitat_features._base``).
_SUBJECT_ID_COLUMN = "subject"


def _units_key_prefix(spec: HabitatSpec) -> str:
    """
    Return the checkpoint key prefix for the clustering-units stage.

    Units depend on exactly one subject and the spec, never on the rest of
    the cohort, so a growing cohort reuses earlier subjects' units across
    runs. The spec fingerprint makes any parameter change invalidate the
    entries instead of silently mixing two analyses in one store.
    """
    return f"habitat.units:{spec.fingerprint()}"


def _label_key_prefix(model: HabitatModel) -> str:
    """
    Return the checkpoint key prefix for the assign-and-describe stage.

    Labels are a function of the fitted definition, and ``model_id`` already
    embeds the fitter spec fingerprint, the defining cohort's digest, and any
    cohort-preprocessing state (``with_cohort_preprocessing`` re-derives it),
    so it is the tightest correct scope for both the fit and the apply path.
    """
    return f"habitat.label:{model.model_id}"


def _one_step_key_prefix(spec: HabitatSpec) -> str:
    """
    Return the checkpoint key prefix for the one-step design.

    One-step habitats are defined inside each subject independently, so the
    whole per-subject computation keys on the spec alone -- adding a cohort
    digest here would defeat resume for this design's defining use case,
    reprocessing a stable subject because an unrelated subject joined.
    """
    return f"habitat.one_step:{spec.fingerprint()}"


def _effective_spec(spec: HabitatSpec, seed: Optional[int]) -> HabitatSpec:
    """
    Fold a call-site seed into the spec.

    Args:
        spec: The declared analysis.
        seed: Overriding seed, or ``None`` to keep ``spec.random_seed``.

    Returns:
        The spec that will actually run -- the one recorded in the manifest,
        so the record never disagrees with the execution.
    """
    import dataclasses

    if seed is None:
        return spec
    return dataclasses.replace(spec, random_seed=int(seed))


@dataclass(frozen=True)
class _ComputeUnits:
    """
    Subject operator: run every pipeline stage up to habitat assignment.

    A class rather than ``pipeline.units`` directly because a bound method
    carries no ``cache_key`` and its type (``builtins.method``) would give
    every method-valued operator the same fallback key in one store.

    Attributes:
        pipeline: Subject pipeline without an assigner.
        key_prefix: Checkpoint key prefix from :func:`_units_key_prefix`.
    """

    pipeline: Any
    key_prefix: str

    def __call__(self, subject: Subject) -> Any:
        """Return this subject's clustering units."""
        return self.pipeline.units(subject)

    def cache_key(self, subject: Subject) -> str:
        """Return the spec-scoped checkpoint key for this subject's units."""
        return f"{self.key_prefix}:{subject.subject_id}"


@dataclass(frozen=True)
class _LabelAndDescribe:
    """
    Subject operator: assign habitats, then describe them.

    A class rather than a closure because execution backends may ship the
    operator to another process, and closures do not pickle.

    Attributes:
        pipeline: Fitted subject pipeline (assigner attached).
        extractors: Habitat feature families; may be empty when the study
            only needs the label maps.
        key_prefix: Checkpoint key prefix from :func:`_label_key_prefix`.
    """

    pipeline: Any
    extractors: Tuple[Any, ...]
    key_prefix: str

    def cache_key(self, subject: Subject) -> str:
        """Return the model-scoped checkpoint key for this subject's labels."""
        return f"{self.key_prefix}:{subject.subject_id}"

    def __call__(
        self, subject: Subject
    ) -> Tuple[HabitatMap, Optional[FeatureTable], Any]:
        """
        Return this subject's habitat map, feature row and clustering units.

        The assignment stages are spelled out here instead of calling
        ``pipeline(subject)`` so the post-cohort-preprocessing units are
        available alongside the map with no recomputation: the units feed
        the v0.1 ``habitats.parquet`` unit table at the writer layer. The
        sequence mirrors :meth:`SubjectPipeline.__call__` exactly.
        """
        units = self.pipeline.units(subject)
        chain = self.pipeline.cohort_feature_preprocessor
        if chain is not None:
            units = units.with_feature_frame(
                chain.transform(units.feature_frame()),
                produced_by="feature_preprocessing.cohort",
                spec_fingerprint=chain.spec.fingerprint(),
            )
        habitat_map = self.pipeline.habitat_assigner(units)
        if not self.extractors:
            return habitat_map, None, units
        table = self.extractors[0](subject, habitat_map)
        for extractor in self.extractors[1:]:
            table = table.join(extractor(subject, habitat_map))
        return habitat_map, table, units


@dataclass(frozen=True)
class _DefineAndLabelWithinSubject:
    """
    Subject operator for the one-step design: define habitats, then label.

    Attributes:
        components: Components built from the spec; the fitter is applied to
            ONE subject's units at a time.
        assigner_name: Registered assigner name from the spec.
        assigner_params: Assigner parameters from the spec.
        extractors: Habitat feature families; may be empty.
        seed: Seed forwarded to the assigner when it is stochastic.
        key_prefix: Checkpoint key prefix from :func:`_one_step_key_prefix`.
    """

    components: HabitatComponents
    assigner_name: str
    assigner_params: Tuple[Tuple[str, Any], ...]
    extractors: Tuple[Any, ...]
    seed: Optional[int]
    key_prefix: str

    def cache_key(self, subject: Subject) -> str:
        """Return the spec-scoped checkpoint key for this subject."""
        return f"{self.key_prefix}:{subject.subject_id}"

    def __call__(
        self, subject: Subject
    ) -> Tuple[HabitatModel, HabitatMap, Optional[FeatureTable], Any]:
        """
        Return this subject's own habitat definition, map, feature row and
        clustering units.

        The units ride along because the v0.1 one-step ``habitats.parquet``
        reports one row per defined habitat, aggregated from them.
        """
        units = self.components.pipeline(assigner=None).units(subject)
        model = self.components.fitter.fit([units])
        assigner = model.assigner(self.assigner_name, **dict(self.assigner_params))
        if self.seed is not None and isinstance(assigner, Seedable):
            assigner.set_random_state(self.seed)
        habitat_map = assigner(units)
        if not self.extractors:
            return model, habitat_map, None, units
        table = self.extractors[0](subject, habitat_map)
        for extractor in self.extractors[1:]:
            table = table.join(extractor(subject, habitat_map))
        return model, habitat_map, table, units


def _fit_cohort_model(
    components: HabitatComponents,
    cohort: Cohort,
    units: Sequence[Any],
) -> HabitatModel:
    """
    Learn the cohort-level habitat definition from pooled units.

    Args:
        components: Components declared by the spec.
        cohort: Training cohort, fingerprinted into the model.
        units: One clustering unit set per subject, in cohort order.

    Returns:
        The fitted definition. When a cohort-level preprocessing chain is
        declared, its fitted state travels INSIDE the model: centroids only
        mean something in the space they were computed in.
    """
    chain = components.cohort_chain
    if chain is not None:
        # Cohort statistics come from the pooled TRAINING units and nothing
        # else; this is the one leakage-sensitive step in habitat definition.
        pooled = pd.concat([unit.feature_frame() for unit in units], ignore_index=True)
        chain.fit(pooled)
        units = [
            unit.with_feature_frame(
                chain.transform(unit.feature_frame()),
                produced_by="feature_preprocessing.cohort",
                spec_fingerprint=chain.spec.fingerprint(),
            )
            for unit in units
        ]
    model = components.fitter.fit(units, cohort=cohort)
    if chain is not None:
        model = model.with_cohort_preprocessing(chain.state, chain.spec.to_dict())
    return model


def _build_assigner(
    model: HabitatModel, spec: HabitatSpec, seed: Optional[int]
) -> Any:
    """Bind the declared assigner to a fitted model."""
    assigner = model.assigner(
        spec.habitat_assigner.name, **spec.habitat_assigner.params
    )
    if seed is not None and isinstance(assigner, Seedable):
        assigner.set_random_state(seed)
    return assigner


def _cohort_feature_table(
    tables: Sequence[Optional[FeatureTable]],
    subject_ids: Sequence[str],
    provenance: Provenance,
) -> FeatureTable:
    """
    Stack the per-subject feature rows into one cohort table.

    Args:
        tables: Per-subject tables in cohort order; ``None`` entries mean the
            spec declared no habitat feature family.
        subject_ids: Cohort order, used when there are no features at all.
        provenance: Provenance recorded on the assembled table.

    Returns:
        The cohort table. With no declared feature family it still carries
        one row per subject: an empty table would silently lose the record
        that these subjects were processed.
    """
    present = [table for table in tables if table is not None]
    if not present:
        frame = pd.DataFrame({_SUBJECT_ID_COLUMN: list(subject_ids)})
        return FeatureTable(
            frame=frame,
            id_columns=(_SUBJECT_ID_COLUMN,),
            feature_columns=(),
            provenance=provenance,
        )
    frame = pd.concat(
        [table.frame for table in present], ignore_index=True, sort=False
    )
    return FeatureTable(
        frame=frame,
        id_columns=present[0].id_columns,
        feature_columns=present[0].feature_columns,
        provenance=provenance,
    )


def _manifest(
    design: str,
    spec: HabitatSpec,
    habitat_maps: Sequence[HabitatMap],
    started_at: str,
) -> RunManifest:
    """
    Record what actually ran.

    Args:
        design: Recipe name, e.g. ``"two_step"``.
        spec: The effective spec.
        habitat_maps: Produced maps, whose provenance chains carry every
            executed step.
        started_at: ISO-8601 timestamp taken before the run.

    Returns:
        The manifest. Only successful subjects appear: ``Cohort.map`` raises
        on failure, so a manifest exists only for a complete run.
    """
    provenance = Provenance(
        produced_by=f"recipes.habitat.{design}",
        spec_fingerprint=spec.fingerprint(),
        inputs=tuple(habitat_map.provenance for habitat_map in habitat_maps),
        software=software_fingerprint(),
        random_seed=spec.random_seed,
    )
    return RunManifest(
        spec_payload=spec.to_dict(),
        provenance=provenance,
        subject_outcomes={
            habitat_map.subject_id: "success" for habitat_map in habitat_maps
        },
        started_at=started_at,
        finished_at=datetime.now(timezone.utc).isoformat(),
    )


def _now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _fit_cohort_design(
    design: str,
    cohort: Cohort,
    spec: HabitatSpec,
    backend: Optional[ExecutionBackend],
    seed: Optional[int],
    checkpoint: Optional[CheckpointStore] = None,
) -> StudyResult:
    """
    Run a design whose habitats are defined across the cohort.

    Shared by :func:`two_step` and :func:`direct_pooling`, which differ only
    in whether the spec declares a supervoxelizer. Expressing that difference
    as two functions rather than a ``mode`` string is the point: the caller
    names a design, not a switch value.

    Args:
        design: Recipe name for provenance.
        cohort: Subjects to fit on.
        spec: The effective analysis declaration.
        backend: Optional execution backend for the per-subject stages.
        seed: Optional seed override, already folded into ``spec``.
        checkpoint: Optional store enabling per-subject resume; units key on
            the spec fingerprint, labels on the fitted model's ``model_id``.

    Returns:
        The study result, entirely in memory.
    """
    started_at = _now()
    components = build_habitat_components(spec)
    units = cohort.map(
        _ComputeUnits(components.pipeline(assigner=None), _units_key_prefix(spec)),
        backend=backend,
        checkpoint=checkpoint,
    )
    model = _fit_cohort_model(components, cohort, units)
    assigner = _build_assigner(model, spec, seed)
    labelled = cohort.map(
        _LabelAndDescribe(
            components.pipeline(assigner=assigner),
            components.extractors,
            _label_key_prefix(model),
        ),
        backend=backend,
        checkpoint=checkpoint,
    )
    habitat_maps = tuple(habitat_map for habitat_map, _, _ in labelled)
    manifest = _manifest(design, spec, habitat_maps, started_at)
    return StudyResult(
        habitat_model=model,
        pipeline=components.pipeline(assigner=assigner),
        features=_cohort_feature_table(
            [table for _, table, _ in labelled],
            [subject.subject_id for subject in cohort],
            manifest.provenance,
        ),
        habitat_maps=habitat_maps,
        manifest=manifest,
        units=tuple(subject_units for _, _, subject_units in labelled),
    )


def two_step(
    cohort: Cohort,
    spec: HabitatSpec,
    *,
    backend: Optional[ExecutionBackend] = None,
    seed: Optional[int] = None,
    checkpoint: Optional[CheckpointStore] = None,
) -> StudyResult:
    """
    Supervoxels per subject, then habitats across the cohort.

    The classical design: each subject's ROI is partitioned into
    homogeneous supervoxels, every supervoxel is described by its features,
    and the habitat definition is learned from all subjects' supervoxels
    pooled together.

    Args:
        cohort: Subjects to fit the habitat definition on.
        spec: The analysis to run; ``spec.supervoxelizer`` is required.
        backend: Optional execution backend (parallelism, timeouts, resume).
            Serial when omitted.
        seed: Optional override of ``spec.random_seed``.
        checkpoint: Optional store enabling per-subject resume. Units are
            reused across cohorts sharing a spec; labels are reused only for
            the same fitted definition (``HabitatModel.model_id`` scope).

    Returns:
        The study result, entirely in memory. Call
        :meth:`~habit.recipes.result.StudyResult.save` to persist it.

    Raises:
        HABITAPIError: If the spec declares no supervoxelizer.
    """
    effective = _effective_spec(spec, seed)
    if effective.supervoxelizer is None:
        raise HABITAPIError(
            "two_step requires a supervoxelizer in the spec. A spec without "
            "one clusters voxels directly: use direct_pooling (cohort-level "
            "habitats) or one_step (per-subject habitats)."
        )
    return _fit_cohort_design(
        "two_step", cohort, effective, backend, effective.random_seed, checkpoint
    )


def direct_pooling(
    cohort: Cohort,
    spec: HabitatSpec,
    *,
    backend: Optional[ExecutionBackend] = None,
    seed: Optional[int] = None,
    checkpoint: Optional[CheckpointStore] = None,
) -> StudyResult:
    """
    No supervoxels: habitats learned from the cohort's pooled voxels.

    Every ROI voxel of every subject is a clustering unit, so the habitat
    definition is comparable across subjects but the supervoxel smoothing
    step is skipped entirely.

    Args:
        cohort: Subjects to fit the habitat definition on.
        spec: The analysis to run; ``spec.supervoxelizer`` must be ``None``.
        backend: Optional execution backend. Serial when omitted.
        seed: Optional override of ``spec.random_seed``.
        checkpoint: Optional store enabling per-subject resume (same key
            scoping as :func:`two_step`).

    Returns:
        The study result, entirely in memory.

    Raises:
        HABITAPIError: If the spec declares a supervoxelizer.
    """
    effective = _effective_spec(spec, seed)
    if effective.supervoxelizer is not None:
        raise HABITAPIError(
            "direct_pooling clusters voxels directly, but this spec declares "
            f"the supervoxelizer {effective.supervoxelizer.name!r}. Use "
            "two_step, or drop the supervoxelizer from the spec."
        )
    return _fit_cohort_design(
        "direct_pooling", cohort, effective, backend, effective.random_seed,
        checkpoint,
    )


def one_step(
    cohort: Cohort,
    spec: HabitatSpec,
    *,
    backend: Optional[ExecutionBackend] = None,
    seed: Optional[int] = None,
    checkpoint: Optional[CheckpointStore] = None,
) -> StudyResult:
    """
    Habitats defined inside each subject, independently.

    Each subject's voxels are clustered on their own -- including the
    habitat-count selection, which is re-run per subject -- so two subjects
    may end up with different habitat counts and their habitat ids are NOT
    comparable. This is what v0.1 called ``clustering_mode: one_step``, and
    the incomparability is inherent to the design rather than a limitation
    of this implementation.

    Consequently the returned
    :attr:`~habit.recipes.result.StudyResult.habitat_model` is ``None``:
    there is no cohort-level definition to publish. The per-subject
    definitions are in
    :attr:`~habit.recipes.result.StudyResult.subject_models`.

    Args:
        cohort: Subjects to process.
        spec: The analysis to run; ``spec.supervoxelizer`` must be ``None``
            and ``spec.cohort_feature_preprocessors`` must be empty, since
            nothing crosses subject boundaries in this design.
        backend: Optional execution backend. Serial when omitted.
        seed: Optional override of ``spec.random_seed``.
        checkpoint: Optional store enabling per-subject resume; keys scope
            on the spec fingerprint only, since no subject's computation
            depends on any other subject in this design.

    Returns:
        The study result, entirely in memory.

    Raises:
        HABITAPIError: If the spec declares a supervoxelizer or a
            cohort-level preprocessing chain.
    """
    effective = _effective_spec(spec, seed)
    if effective.supervoxelizer is not None:
        raise HABITAPIError(
            "one_step clusters each subject's voxels directly, but this spec "
            f"declares the supervoxelizer {effective.supervoxelizer.name!r}."
        )
    if effective.cohort_feature_preprocessors:
        raise HABITAPIError(
            "one_step defines habitats within each subject, so a cohort-level "
            "preprocessing chain would fit statistics no step ever uses. "
            "Move those methods to voxel_feature_preprocessors, or use "
            "direct_pooling."
        )
    started_at = _now()
    components = build_habitat_components(effective)
    outcomes = cohort.map(
        _DefineAndLabelWithinSubject(
            components=components,
            assigner_name=effective.habitat_assigner.name,
            assigner_params=tuple(effective.habitat_assigner.params.items()),
            extractors=components.extractors,
            seed=effective.random_seed,
            key_prefix=_one_step_key_prefix(effective),
        ),
        backend=backend,
        checkpoint=checkpoint,
    )
    habitat_maps = tuple(habitat_map for _, habitat_map, _, _ in outcomes)
    manifest = _manifest("one_step", effective, habitat_maps, started_at)
    return StudyResult(
        habitat_model=None,
        pipeline=None,
        features=_cohort_feature_table(
            [table for _, _, table, _ in outcomes],
            [subject.subject_id for subject in cohort],
            manifest.provenance,
        ),
        habitat_maps=habitat_maps,
        manifest=manifest,
        subject_models={
            habitat_map.subject_id: model
            for model, habitat_map, _, _ in outcomes
        },
        units=tuple(subject_units for _, _, _, subject_units in outcomes),
    )


def apply_habitat_model(
    cohort: Cohort,
    spec: HabitatSpec,
    model: HabitatModel,
    *,
    backend: Optional[ExecutionBackend] = None,
    seed: Optional[int] = None,
    checkpoint: Optional[CheckpointStore] = None,
) -> StudyResult:
    """
    Project a published habitat definition onto a new cohort.

    The prediction half of the ladder: no habitat is defined here, the given
    model is applied. The model's own cohort-level preprocessing state is
    restored and re-applied, because centroids only mean something in the
    feature space they were computed in -- skipping it would still produce
    plausible-looking labels, which is precisely why it is not optional.

    Args:
        cohort: Subjects to label.
        spec: The analysis declaration whose upstream stages (voxel features,
            supervoxelisation, per-subject preprocessing) must match those
            the model was fitted with.
        model: A fitted, possibly reloaded, habitat definition.
        backend: Optional execution backend. Serial when omitted.
        seed: Optional override of ``spec.random_seed``.
        checkpoint: Optional store enabling per-subject resume; keys scope
            on ``model.model_id``, so two definitions never share entries.

    Returns:
        The study result for the projected cohort.
    """
    started_at = _now()
    effective = _effective_spec(spec, seed)
    components = build_habitat_components(effective)
    components = _with_model_preprocessing(components, model)
    assigner = _build_assigner(model, effective, effective.random_seed)
    labelled = cohort.map(
        _LabelAndDescribe(
            components.pipeline(assigner=assigner),
            components.extractors,
            _label_key_prefix(model),
        ),
        backend=backend,
        checkpoint=checkpoint,
    )
    habitat_maps = tuple(habitat_map for habitat_map, _, _ in labelled)
    manifest = _manifest("apply_habitat_model", effective, habitat_maps, started_at)
    return StudyResult(
        habitat_model=model,
        pipeline=components.pipeline(assigner=assigner),
        features=_cohort_feature_table(
            [table for _, table, _ in labelled],
            [subject.subject_id for subject in cohort],
            manifest.provenance,
        ),
        habitat_maps=habitat_maps,
        manifest=manifest,
        units=tuple(subject_units for _, _, subject_units in labelled),
    )


def _with_model_preprocessing(
    components: HabitatComponents, model: HabitatModel
) -> HabitatComponents:
    """
    Replace the cohort chain with the FITTED one carried by the model.

    Args:
        components: Components built from the spec, whose cohort chain (if
            any) is unfitted.
        model: The model being applied.

    Returns:
        The components with the model's fitted chain attached, or unchanged
        when the model carries none.
    """
    import dataclasses

    from habit.domain.feature_preprocessing import CohortPreprocessingChain

    state = (model.preprocessing_state or {}).get("cohort_feature_preprocessor")
    if state is None:
        return components
    return dataclasses.replace(
        components, cohort_chain=CohortPreprocessingChain.from_state(state)
    )
