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

import logging
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import pandas as pd

from habit.exceptions import HABITAPIError, ProcessingError
from habit.contracts.habitat import HabitatMap, HabitatModel
from habit.contracts.manifest import RunManifest
from habit.contracts.ops import ExecutionBackend, SubjectResult
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

_LOG = logging.getLogger(__name__)

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
class _SubjectUnits:
    """
    One subject plus clustering units already computed in an earlier stage.

    Used by the cohort-level recipes so the label stage can assign habitats
    without re-running voxel / supervoxel feature extraction (which is the
    dominant cost for ``voxel_radiomics`` and would otherwise run twice).
    """

    subject: Subject
    units: Any

    @property
    def subject_id(self) -> str:
        """Return the subject identity for backends and checkpoints."""
        return self.subject.subject_id


@dataclass(frozen=True)
class _LabelAndDescribe:
    """
    Subject operator: extract units, assign habitats, then describe them.

    Used by :func:`apply_habitat_model`, where units are not already in
    memory. Cohort-level fit recipes use :class:`_AssignPrecomputedUnits`
    instead so voxel radiomics is not paid twice.

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

        Predict-path: Stage-1 runs here because held-out subjects have no
        in-memory units. Post-cohort-preprocessing units ride along for the
        v0.1 ``habitats.parquet`` writer.
        """
        units = self.pipeline.units(subject)
        return self.pipeline.label_and_describe(
            subject, units, self.extractors
        )


@dataclass(frozen=True)
class _AssignPrecomputedUnits:
    """
    Assign habitats from units already produced by the units stage.

    Keeping this as a picklable operator (rather than a closure) preserves
    checkpoint ``cache_key`` behaviour for resume tests while avoiding a
    second ProcessPool wave that would re-extract features and contend for
    the GPU.

    Attributes:
        pipeline: Fitted subject pipeline (assigner attached).
        extractors: Habitat feature families; may be empty.
        key_prefix: Checkpoint key prefix from :func:`_label_key_prefix`.
    """

    pipeline: Any
    extractors: Tuple[Any, ...]
    key_prefix: str

    def cache_key(self, item: _SubjectUnits) -> str:
        """Return the model-scoped checkpoint key for this subject's labels."""
        return f"{self.key_prefix}:{item.subject.subject_id}"

    def __call__(
        self, item: _SubjectUnits
    ) -> Tuple[HabitatMap, Optional[FeatureTable], Any]:
        """Return habitat map, optional feature row, and post-prep units."""
        return self.pipeline.label_and_describe(
            item.subject, item.units, self.extractors
        )


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
        # Stage-1 once: units, then fit/assign/describe without re-extraction.
        units = self.components.pipeline(assigner=None).units(subject)
        model = self.components.habitat_model_fitter.fit([units])
        assigner = model.assigner(self.assigner_name, **dict(self.assigner_params))
        if self.seed is not None and isinstance(assigner, Seedable):
            assigner.set_random_state(self.seed)
        pipeline = self.components.pipeline(assigner=assigner)
        habitat_map, table, prepared = pipeline.label_and_describe(
            subject, units, self.extractors
        )
        return model, habitat_map, table, prepared


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
    chain = components.cohort_feature_preprocessor
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
    model = components.habitat_model_fitter.fit(units, cohort=cohort)
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
    *,
    subject_outcomes: Optional[Mapping[str, str]] = None,
) -> RunManifest:
    """
    Record what actually ran.

    Args:
        design: Recipe name, e.g. ``"two_step"``.
        spec: The effective spec.
        habitat_maps: Produced maps, whose provenance chains carry every
            executed step.
        started_at: ISO-8601 timestamp taken before the run.
        subject_outcomes: Optional per-subject success / failure summaries.
            Defaults to success for every produced habitat map.

    Returns:
        The manifest, including subjects that failed and were excluded when
        ``subject_outcomes`` records them (v0.1 continue parity).
    """
    provenance = Provenance(
        produced_by=f"recipes.habitat.{design}",
        spec_fingerprint=spec.fingerprint(),
        inputs=tuple(habitat_map.provenance for habitat_map in habitat_maps),
        software=software_fingerprint(),
        random_seed=spec.random_seed,
    )
    outcomes: Dict[str, str]
    if subject_outcomes is not None:
        outcomes = dict(subject_outcomes)
    else:
        outcomes = {
            habitat_map.subject_id: "success" for habitat_map in habitat_maps
        }
    return RunManifest(
        spec_payload=spec.to_dict(),
        provenance=provenance,
        subject_outcomes=outcomes,
        started_at=started_at,
        finished_at=datetime.now(timezone.utc).isoformat(),
    )


def _now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _error_summary(error: BaseException) -> str:
    """Format one subject failure for the run manifest."""
    return f"{type(error).__name__}: {error}"


@contextmanager
def _backend_session(backend: Optional[ExecutionBackend]) -> Iterator[None]:
    """
    Reuse persistent workers across recipe stages when the backend supports it.

    Args:
        backend: Optional execution backend from the caller.
    """
    reuse = getattr(backend, "reuse_workers", None)
    if callable(reuse):
        with reuse():
            yield
        return
    with nullcontext():
        yield


def _map_soft(
    cohort: Cohort,
    op: Callable[[Subject], Any],
    *,
    backend: Optional[ExecutionBackend],
    checkpoint: Optional["CheckpointStore"],
    stage: str,
) -> Tuple[Cohort, List[Any], Dict[str, str]]:
    """
    Map ``op`` with soft failure (v0.1 ``on_subject_failure: continue``).

    Args:
        cohort: Subjects to process.
        op: Subject-level operator.
        backend: Optional execution backend.
        checkpoint: Optional resume store.
        stage: Short stage name for log / error messages.

    Returns:
        ``(survivor_cohort, values_in_survivor_order, failures)`` where
        ``failures`` maps subject id to an error summary.

    Raises:
        ProcessingError: If every subject failed.
    """
    slots: Sequence[SubjectResult[Any]] = cohort.map(
        op,
        backend=backend,
        checkpoint=checkpoint,
        raise_on_failure=False,
    )
    failures: Dict[str, str] = {}
    values_by_id: Dict[str, Any] = {}
    for slot in slots:
        if slot.error is not None:
            summary = _error_summary(slot.error)
            failures[slot.subject_id] = summary
            _LOG.warning(
                "[%s] subject %s failed: %s", stage, slot.subject_id, summary
            )
        else:
            values_by_id[slot.subject_id] = slot.value
    if not values_by_id:
        detail = "; ".join(
            f"{sid}: {msg}" for sid, msg in sorted(failures.items())
        )
        raise ProcessingError(
            f"All {len(cohort)} subject(s) failed in recipe stage "
            f"{stage!r}: {detail}"
        )
    survivors = Cohort(
        [subject for subject in cohort if subject.subject_id in values_by_id],
        name=cohort.name,
        metadata=cohort.metadata,
    )
    values = [values_by_id[subject.subject_id] for subject in survivors]
    return survivors, values, failures


def _map_soft_items(
    template_cohort: Cohort,
    items: Sequence[Any],
    op: Callable[[Any], Any],
    *,
    checkpoint: Optional["CheckpointStore"],
    stage: str,
) -> Tuple[Cohort, List[Any], Dict[str, str]]:
    """
    Map a soft-fail operator over arbitrary subject-scoped payloads.

    Used when the payload is richer than a bare :class:`Subject` (for
    example precomputed clustering units). Assignment is always serial in
    the parent process: the heavy feature work already finished in the
    units stage, and re-shipping volumes through a process pool would only
    add pickle/GPU contention.

    Args:
        template_cohort: Cohort that supplies survivor ordering / metadata.
        items: Payloads exposing ``subject_id`` (and optionally ``subject``).
        op: Operator applied to each item.
        checkpoint: Optional resume store.
        stage: Short stage name for log / error messages.

    Returns:
        ``(survivor_cohort, values_in_survivor_order, failures)``.

    Raises:
        ProcessingError: If every item failed.
    """
    from habit.execution.backends import SerialBackend
    from habit.utils.progress_utils import CustomTqdm

    runner = SerialBackend(on_subject_failure="continue")
    total = len(items)
    op_name = type(op).__name__
    bar = CustomTqdm(total=total, desc=f"Cohort.map[{op_name}]")

    def _progress(completed: int, expected: int) -> None:
        bar.total = expected
        bar.n = completed
        bar.refresh()

    try:
        slots = list(
            runner.map(op, items, checkpoint=checkpoint, progress=_progress)
        )
    finally:
        bar.close()

    failures: Dict[str, str] = {}
    values_by_id: Dict[str, Any] = {}
    for slot in slots:
        if slot.error is not None:
            summary = _error_summary(slot.error)
            failures[slot.subject_id] = summary
            _LOG.warning(
                "[%s] subject %s failed: %s", stage, slot.subject_id, summary
            )
        else:
            values_by_id[slot.subject_id] = slot.value
    if not values_by_id:
        detail = "; ".join(
            f"{sid}: {msg}" for sid, msg in sorted(failures.items())
        )
        raise ProcessingError(
            f"All {len(items)} subject(s) failed in recipe stage "
            f"{stage!r}: {detail}"
        )
    survivors = Cohort(
        [
            subject
            for subject in template_cohort
            if subject.subject_id in values_by_id
        ],
        name=template_cohort.name,
        metadata=template_cohort.metadata,
    )
    values = [values_by_id[subject.subject_id] for subject in survivors]
    return survivors, values, failures


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

    Per-subject failures are isolated (v0.1 continue): the recipe proceeds
    with successful subjects and records exclusions on the run manifest.

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
    outcomes: Dict[str, str] = {
        subject.subject_id: "success" for subject in cohort
    }
    with _backend_session(backend):
        units_cohort, units, unit_failures = _map_soft(
            cohort,
            _ComputeUnits(
                components.pipeline(assigner=None), _units_key_prefix(spec)
            ),
            backend=backend,
            checkpoint=checkpoint,
            stage=f"{design}.units",
        )
        for subject_id, summary in unit_failures.items():
            outcomes[subject_id] = summary
        model = _fit_cohort_model(components, units_cohort, units)
        assigner = _build_assigner(model, spec, seed)
        # Reuse units already resident in the parent. Re-mapping subjects
        # through ``_LabelAndDescribe`` would re-run voxel_radiomics (and
        # any GPU workers) a second time -- the dominant cost of texture
        # habitats and a common freeze trigger when ProcessPool × CUDA
        # oversubscribe a laptop GPU.
        labelled_cohort, labelled, label_failures = _map_soft_items(
            units_cohort,
            [
                _SubjectUnits(subject, subject_units)
                for subject, subject_units in zip(units_cohort, units)
            ],
            _AssignPrecomputedUnits(
                components.pipeline(assigner=assigner),
                components.habitat_features,
                _label_key_prefix(model),
            ),
            checkpoint=checkpoint,
            stage=f"{design}.label",
        )
        for subject_id, summary in label_failures.items():
            outcomes[subject_id] = summary
    habitat_maps = tuple(habitat_map for habitat_map, _, _ in labelled)
    for habitat_map in habitat_maps:
        outcomes[habitat_map.subject_id] = "success"
    manifest = _manifest(
        design,
        spec,
        habitat_maps,
        started_at,
        subject_outcomes=outcomes,
    )
    return StudyResult(
        habitat_model=model,
        pipeline=components.pipeline(assigner=assigner),
        features=_cohort_feature_table(
            [table for _, table, _ in labelled],
            [subject.subject_id for subject in labelled_cohort],
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

    Examples:
        >>> from habit import HabitatSpec, Spec, make_synthetic_cohort
        >>> import habit.recipes as recipes
        >>> cohort = make_synthetic_cohort(n_subjects=6, shape=(24, 24, 24), rng=42)
        >>> spec = HabitatSpec(
        ...     name="demo",
        ...     voxel_feature_extractor=Spec("raw", {"modalities": ["T1", "T2"]}),
        ...     supervoxelizer=Spec("kmeans", {"n_supervoxels": 8, "n_init": 5}),
        ...     habitat_model_fitter=Spec(
        ...         "kmeans",
        ...         {"min_habitats": 2, "max_habitats": 3,
        ...          "validation": "silhouette", "n_init": 5},
        ...     ),
        ...     habitat_assigner=Spec("nearest_centroid"),
        ...     habitat_features=(Spec("volume"),),
        ...     random_seed=42,
        ... )
        >>> result = recipes.two_step(cohort, spec)
        >>> result.habitat_model.n_habitats >= 2
        True
        >>> len(result.habitat_maps) == len(cohort)
        True
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
    subject_outcomes: Dict[str, str] = {
        subject.subject_id: "success" for subject in cohort
    }
    with _backend_session(backend):
        survivors, outcomes, failures = _map_soft(
            cohort,
            _DefineAndLabelWithinSubject(
                components=components,
                assigner_name=effective.habitat_assigner.name,
                assigner_params=tuple(effective.habitat_assigner.params.items()),
                extractors=components.habitat_features,
                seed=effective.random_seed,
                key_prefix=_one_step_key_prefix(effective),
            ),
            backend=backend,
            checkpoint=checkpoint,
            stage="one_step",
        )
    for subject_id, summary in failures.items():
        subject_outcomes[subject_id] = summary
    habitat_maps = tuple(habitat_map for _, habitat_map, _, _ in outcomes)
    for habitat_map in habitat_maps:
        subject_outcomes[habitat_map.subject_id] = "success"
    manifest = _manifest(
        "one_step",
        effective,
        habitat_maps,
        started_at,
        subject_outcomes=subject_outcomes,
    )
    return StudyResult(
        habitat_model=None,
        pipeline=None,
        features=_cohort_feature_table(
            [table for _, _, table, _ in outcomes],
            [subject.subject_id for subject in survivors],
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

    Examples:
        Train on one cohort, persist the model, and project it onto new
        subjects (here the same synthetic cohort doubles as the held-out
        data, which also verifies the save/load/apply round-trip):

        >>> from habit import HabitatModel, HabitatSpec, Spec, make_synthetic_cohort
        >>> import habit.recipes as recipes
        >>> cohort = make_synthetic_cohort(n_subjects=5, shape=(20, 20, 20), rng=7)
        >>> spec = HabitatSpec(
        ...     name="demo",
        ...     voxel_feature_extractor=Spec("raw", {"modalities": ["T1", "T2"]}),
        ...     supervoxelizer=Spec("kmeans", {"n_supervoxels": 6, "n_init": 5}),
        ...     habitat_model_fitter=Spec(
        ...         "kmeans",
        ...         {"min_habitats": 2, "max_habitats": 3,
        ...          "validation": "silhouette", "n_init": 5},
        ...     ),
        ...     habitat_assigner=Spec("nearest_centroid"),
        ...     habitat_features=(Spec("volume"),),
        ...     random_seed=7,
        ... )
        >>> train = recipes.two_step(cohort, spec)
        >>> train.habitat_model.save("/tmp/demo.habitatmodel")  # doctest: +SKIP
        >>> reloaded = HabitatModel.load("/tmp/demo.habitatmodel")  # doctest: +SKIP
        >>> projected = recipes.apply_habitat_model(cohort, spec, reloaded)  # doctest: +SKIP
        >>> len(projected.habitat_maps) == len(cohort)  # doctest: +SKIP
        True
    """
    started_at = _now()
    effective = _effective_spec(spec, seed)
    components = build_habitat_components(effective)
    components = _with_model_preprocessing(components, model)
    assigner = _build_assigner(model, effective, effective.random_seed)
    subject_outcomes: Dict[str, str] = {
        subject.subject_id: "success" for subject in cohort
    }
    with _backend_session(backend):
        survivors, labelled, failures = _map_soft(
            cohort,
            _LabelAndDescribe(
                components.pipeline(assigner=assigner),
                components.habitat_features,
                _label_key_prefix(model),
            ),
            backend=backend,
            checkpoint=checkpoint,
            stage="apply_habitat_model",
        )
    for subject_id, summary in failures.items():
        subject_outcomes[subject_id] = summary
    habitat_maps = tuple(habitat_map for habitat_map, _, _ in labelled)
    for habitat_map in habitat_maps:
        subject_outcomes[habitat_map.subject_id] = "success"
    manifest = _manifest(
        "apply_habitat_model",
        effective,
        habitat_maps,
        started_at,
        subject_outcomes=subject_outcomes,
    )
    return StudyResult(
        habitat_model=model,
        pipeline=components.pipeline(assigner=assigner),
        features=_cohort_feature_table(
            [table for _, table, _ in labelled],
            [subject.subject_id for subject in survivors],
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
        components,
        cohort_feature_preprocessor=CohortPreprocessingChain.from_state(state),
    )
