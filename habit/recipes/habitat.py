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
"""Internal engines behind :class:`~habit.recipes.study.Study` (L4).

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

* ``two_step`` -- supervoxels per subject, habitats across the cohort.
* ``direct_pooling`` -- no supervoxels; habitats across the cohort's
  pooled voxels.
* ``one_step`` -- no supervoxels; habitats defined INSIDE each subject,
  independently. There is no cohort-level definition, and the habitat ids of
  two subjects are not comparable.

:func:`_fit_habitat` is the unified implementation: it resolves the ordered
:class:`~habit.spec.specs.Stage` list on the spec and runs the shared
stage dataflow executor. The three named validators remain as thin guards
-- they check that the spec is consistent with the design their name
promises, then delegate to :func:`_fit_habitat`.

This module is private on purpose: the single public entry point is
:class:`~habit.recipes.study.Study` (``study.fit(cohort)`` /
``study.predict(cohort)`` / ``study.fit_predict(cohort)``). Keeping one
public object with one ``fit`` verb mirrors the sklearn estimator lifecycle
instead of exposing a second, function-style vocabulary for the same work.
"""

from __future__ import annotations

import dataclasses
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
    cast,
)

import pandas as pd

from habit.exceptions import HABITAPIError, ProcessingError
from habit.contracts.habitat import (
    HabitatMap,
    HabitatModel,
    Supervoxelization,
)
from habit.contracts.inspection import StepObserver
from habit.contracts.manifest import RunManifest
from habit.contracts.ops import ExecutionBackend, ResultWriter, SubjectResult
from habit.contracts.provenance import Provenance, software_fingerprint
from habit.contracts.subject import Cohort, Subject
from habit.contracts.table import FeatureTable
from habit.domain.assembly import HabitatComponents, build_habitat_components
from habit.domain.pooling import fan_in
from habit.domain.protocols import Seedable
from habit.domain.stages import (
    design_from_stages,
    execute_habitat_dataflow,
    normalize_spec_for_execution,
    resolve_habitat_stages,
)
from habit.recipes.result import StudyResult
from habit.spec.specs import HabitatSpec

if TYPE_CHECKING:
    # Typing-only reference: the store is an execution-layer concern and is
    # only ever passed through to ``Cohort.map``, never opened here.
    from habit.execution.checkpoint import CheckpointStore

_LOG = logging.getLogger(__name__)

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


def _reject_process_inspect(
    inspect: Optional[StepObserver],
    backend: Optional[ExecutionBackend],
) -> None:
    """
    Refuse in-memory inspection under the process backend.

    Step records are produced inside worker processes and cannot reach a
    parent-process recorder. Debugging must use a serial backend.

    Args:
        inspect: Optional step observer from the recipe call site.
        backend: Optional execution backend.

    Raises:
        HABITAPIError: If ``inspect`` is set and ``backend.policy.backend``
            is ``"process"``.
    """
    if inspect is None:
        return
    policy = getattr(backend, "policy", None)
    if getattr(policy, "backend", None) == "process":
        raise HABITAPIError(
            "inspect= is not supported with the process backend: step records "
            "are produced inside worker processes and cannot reach this "
            "recorder. Use RunPolicy(backend='serial') / workers=1 while "
            "debugging."
        )


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

    Used by :func:`_apply_habitat_model`, where units are not already in
    memory. Cohort-level fit recipes use :class:`_AssignPrecomputedUnits`
    instead so voxel radiomics is not paid twice.

    Attributes:
        pipeline: Fitted subject pipeline (assigner attached); may carry an
            optional step observer for inspection.
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
class _OneStepSubjectProduct:
    """
    Per-subject payload of the one-step operator.

    Voxel-level clustering units are the memory-dominant artefact of the
    one-step design: one feature-matrix row plus one unit-id volume per ROI
    voxel, per subject. With ``slim_units`` the operator aggregates the v0.1
    units-table rows inside the worker and returns them instead of the
    units, so voxel arrays never cross the process boundary and the parent
    holds O(n_habitats) rows per subject instead of O(n_voxels).

    Attributes:
        subject_id: Owning subject. Carried explicitly because the habitat
            map may be stripped after streaming persistence, so identity
            must not depend on it.
        model: This subject's own habitat definition.
        habitat_map: Habitat label image, or ``None`` after the streaming
            consumer persisted and stripped it.
        map_provenance: Provenance chain of the habitat map. Kept even when
            the map itself is stripped, so the run manifest stays complete.
        table: Habitat feature row, or ``None`` when the spec declares no
            habitat feature family.
        units: Voxel-level clustering units; present only in
            ``retain="all"`` runs.
        units_rows: This subject's rows of the v0.1 ``habitats`` units table
            at habitat granularity; present when ``units`` were slimmed away.
    """

    subject_id: str
    model: HabitatModel
    habitat_map: Optional[HabitatMap]
    map_provenance: Provenance
    table: Optional[FeatureTable]
    units: Optional[Supervoxelization]
    units_rows: Optional[pd.DataFrame]

    def without_map(self) -> "_OneStepSubjectProduct":
        """Return a copy with the label image dropped (already persisted)."""
        return dataclasses.replace(self, habitat_map=None)


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
        observer: Optional step observer for debugging / QA.
        slim_units: When ``True``, aggregate the v0.1 units-table rows
            in-worker and return them instead of the voxel-level units.
    """

    components: HabitatComponents
    assigner_name: str
    assigner_params: Tuple[Tuple[str, Any], ...]
    extractors: Tuple[Any, ...]
    seed: Optional[int]
    key_prefix: str
    observer: Optional[StepObserver] = None
    slim_units: bool = False

    def cache_key(self, subject: Subject) -> str:
        """Return the spec-scoped checkpoint key for this subject."""
        # The payload shape differs between slim and full returns; the mode
        # segment keeps a store written by one mode from being misread by
        # the other (older entries simply miss and are recomputed).
        mode = "slim" if self.slim_units else "full"
        return f"{self.key_prefix}:{mode}:{subject.subject_id}"

    def __call__(
        self, subject: Subject
    ) -> _OneStepSubjectProduct:
        """
        Return this subject's own habitat definition, map, feature row and
        clustering units (or their aggregated units-table rows when slim).

        The units ride along in full mode because the v0.1 one-step
        ``habitats.parquet`` reports one row per defined habitat, aggregated
        from them. In slim mode that aggregation happens here, in-worker.
        """
        # Stage-1 once: units, then fit/assign/describe without re-extraction.
        units = self.components.pipeline(
            assigner=None, observer=self.observer
        ).units(subject)
        model = self.components.habitat_model_fitter.fit([units])
        assigner = model.assigner(self.assigner_name, **dict(self.assigner_params))
        if self.seed is not None and isinstance(assigner, Seedable):
            assigner.set_random_state(self.seed)
        pipeline = self.components.pipeline(
            assigner=assigner, observer=self.observer
        )
        habitat_map, table, prepared = pipeline.label_and_describe(
            subject, units, self.extractors
        )
        units_rows: Optional[pd.DataFrame] = None
        kept_units: Optional[Supervoxelization] = prepared
        if self.slim_units:
            # Deferred import: the units-table layout lives with the
            # directory writer (L1); aggregating here keeps voxel arrays
            # inside the worker process.
            from habit.adapters.writers import subject_units_frame

            units_rows = subject_units_frame(prepared, habitat_map, "habitat")
            kept_units = None
        return _OneStepSubjectProduct(
            subject_id=str(subject.subject_id),
            model=model,
            habitat_map=habitat_map,
            map_provenance=habitat_map.provenance,
            table=table,
            units=kept_units,
            units_rows=units_rows,
        )


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
        # The fan-in atom pools exactly like the fitter's internal pooling
        # does, and records the subject index alongside the matrix.
        pooled = fan_in(units)
        chain.fit(pooled.frame)
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


def _with_full_spec_payload(model: HabitatModel, spec: HabitatSpec) -> HabitatModel:
    """
    Embed the full analysis specification in the model card.

    The fitter alone records only its own spec, but a circulated model must
    state the whole upstream procedure (voxel features, partition,
    preprocessing chains) so :meth:`~habit.recipes.study.Study.from_model`
    can rebuild it and external validation replays the identical pipeline.
    ``model_id`` derives from the fitter fingerprint and cohort digest, never
    from ``spec_payload``, so enriching the card cannot change the model's
    identity. Keys the fit path already recorded (the resolved fitter spec,
    the fitted cohort preprocessing chain) win over the declared spec's
    entries.
    """
    merged = {**spec.to_dict(), **dict(model.spec_payload)}
    return dataclasses.replace(model, spec_payload=merged)


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
    *,
    inputs: Sequence[Provenance],
    started_at: str,
    subject_outcomes: Optional[Mapping[str, str]] = None,
) -> RunManifest:
    """
    Record what actually ran.

    Args:
        design: Recipe name, e.g. ``"two_step"``.
        spec: The effective spec.
        inputs: Per-subject provenance chains of the produced habitat maps.
            Taken as bare provenances (rather than the maps themselves) so
            streaming runs -- which drop maps from memory after persisting
            them -- still record full lineage.
        started_at: ISO-8601 timestamp taken before the run.
        subject_outcomes: Optional per-subject success / failure summaries.

    Returns:
        The manifest, including subjects that failed and were excluded when
        ``subject_outcomes`` records them (v0.1 continue parity).
    """
    provenance = Provenance(
        produced_by=f"recipes.habitat.{design}",
        spec_fingerprint=spec.fingerprint(),
        inputs=tuple(inputs),
        software=software_fingerprint(),
        random_seed=spec.random_seed,
    )
    outcomes: Dict[str, str] = (
        dict(subject_outcomes) if subject_outcomes is not None else {}
    )
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
    on_item: Optional[Callable[[str, Any], Any]] = None,
) -> Tuple[Cohort, List[Any], Dict[str, str]]:
    """
    Map ``op`` with soft failure (v0.1 ``on_subject_failure: continue``).

    Results are consumed as a STREAM: the backend yields each subject as it
    completes (completion order under the process pool) and ``on_item``
    runs immediately in the parent process, so per-subject artefacts can be
    persisted -- and memory-dominant payloads dropped -- while the rest of
    the cohort is still computing. ``Cohort.map`` is deliberately bypassed
    here because it materialises every result into a list first, which is
    exactly the accumulation long cohorts cannot afford.

    Args:
        cohort: Subjects to process.
        op: Subject-level operator.
        backend: Optional execution backend.
        checkpoint: Optional resume store.
        stage: Short stage name for log / error messages.
        on_item: Optional hook called in the parent as
            ``on_item(subject_id, value)`` for every successful slot --
            including checkpoint-resumed ones, so a crash between the
            backend's checkpoint write and the hook is repaired on resume.
            Its return value is what gets retained; returning a stripped
            copy is how heavy payloads are dropped.

    Returns:
        ``(survivor_cohort, values_in_survivor_order, failures)`` where
        ``failures`` maps subject id to an error summary.

    Raises:
        ProcessingError: If every subject failed, or the backend omitted a
            subject (its contract is one result per item, exactly once).
    """
    from habit.execution.backends import SerialBackend
    from habit.utils.progress_utils import CustomTqdm

    runner = backend if backend is not None else SerialBackend(
        on_subject_failure="continue"
    )
    total = len(cohort)
    op_name = type(op).__name__
    bar = CustomTqdm(total=total, desc=f"Cohort.map[{op_name}]")

    def _progress(completed: int, expected: int) -> None:
        bar.total = expected
        bar.n = completed
        bar.refresh()

    failures: Dict[str, str] = {}
    values_by_id: Dict[str, Any] = {}
    try:
        # ``op`` is a plain callable at this boundary; the backends duck-type
        # it through ``_cache_key_of``, so cast across that intended gap.
        slots = runner.map(
            cast(Any, op),
            list(cohort),
            checkpoint=checkpoint,
            progress=_progress,
        )
        for slot in slots:
            if slot.error is not None:
                summary = _error_summary(slot.error)
                failures[slot.subject_id] = summary
                _LOG.warning(
                    "[%s] subject %s failed: %s", stage, slot.subject_id, summary
                )
                continue
            value = slot.value
            if on_item is not None:
                value = on_item(slot.subject_id, value)
            values_by_id[slot.subject_id] = value
    finally:
        bar.close()
    if len(values_by_id) + len(failures) != total:
        missing = sorted(
            set(str(subject.subject_id) for subject in cohort)
            - set(values_by_id)
            - set(failures)
        )
        raise ProcessingError(
            f"Backend omitted {len(missing)} subject(s) in recipe stage "
            f"{stage!r}: {missing}"
        )
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


#: Retention modes for the in-memory :class:`StudyResult` of a fit.
#: ``all`` keeps every artefact (the historical default); ``maps`` drops the
#: voxel-level clustering units; ``tables`` additionally drops habitat maps
#: and therefore requires a streaming ``writer`` so maps are persisted as
#: their subjects complete.
_RETAIN_MODES = ("all", "maps", "tables")


def _validate_streaming_options(design: str, report: Optional[Any]) -> None:
    """
    Fail fast on unsupported or lossy streaming configurations.

    Args:
        design: Resolved recipe design (``one_step`` / ``two_step`` /
            ``direct_pooling``).
        report: Optional :class:`~habit.report.Report` from the call site.

    Raises:
        HABITAPIError: If figures have no destination, or if a non-one_step
            design is asked to stream (not wired yet).
    """
    if report is None:
        return
    if report.figures and report.resolve_figure_dir() is None:
        raise HABITAPIError(
            "Report.figures requires figure_dir= or a writer that exposes "
            "a filesystem root (DirectoryResultWriter.root)."
        )
    if design != "one_step" and report.streams:
        raise HABITAPIError(
            "Streaming persistence (report= / writer= / retain=) is "
            f"currently supported for the one_step design only; this spec "
            f"runs as {design!r}. Omit report= for cohort-level designs."
        )


def _build_one_step_item_hook(
    report: Optional[Any],
    subjects_by_id: Mapping[str, Subject],
) -> Optional[Callable[[str, _OneStepSubjectProduct], _OneStepSubjectProduct]]:
    """
    Build the per-subject streaming consumer for the one-step design.

    The returned hook runs in the PARENT process the moment the backend
    yields a subject (including checkpoint-resumed ones): it asks the
    Report to persist and draw, then applies the retention policy so
    heavy payloads never accumulate across the cohort.

    Args:
        report: Optional report declaring persist / figures / retain.
        subjects_by_id: Cohort lookup for the Report's ``Subject``.

    Returns:
        The hook, or ``None`` when nothing streams (historical default).
    """
    if report is None or not report.streams:
        return None

    def _on_item(
        subject_id: str, product: _OneStepSubjectProduct
    ) -> _OneStepSubjectProduct:
        from habit.report import SubjectContext

        habitat_map = product.habitat_map
        if habitat_map is None:
            # The backend yields operator products before any stripping, so
            # a missing map here means the operator contract broke.
            raise HABITAPIError(
                f"one_step product for subject {subject_id!r} carries no "
                "habitat map; cannot stream-persist it."
            )
        report.consume_subject(
            SubjectContext(
                subject=subjects_by_id[subject_id],
                habitat_map=habitat_map,
                model=product.model,
            )
        )
        if report.retain == "tables":
            return product.without_map()
        return product

    return _on_item


def _fit_habitat(
    cohort: Cohort,
    spec: HabitatSpec,
    *,
    backend: Optional[ExecutionBackend] = None,
    seed: Optional[int] = None,
    checkpoint: Optional[CheckpointStore] = None,
    inspect: Optional[StepObserver] = None,
    writer: Optional[ResultWriter] = None,
    retain: str = "all",
    on_subject_complete: Optional[
        Callable[[Subject, HabitatMap, HabitatModel], None]
    ] = None,
    persist_subject_models: bool = True,
    report: Optional[Any] = None,
) -> StudyResult:
    """
    Fit the habitat analysis the spec declares, whatever its dataflow.

    This is the unified, stage-driven implementation behind
    :meth:`~habit.recipes.study.Study.fit`: the ordered
    :attr:`~habit.spec.specs.HabitatSpec.stages` list (explicit or expanded
    from the named-field sugar) is resolved and executed by one shared
    dataflow executor. The named designs (:func:`_two_step`, :func:`_one_step`,
    :func:`_direct_pooling`) are thin validators that check the spec against
    the design their name promises, then delegate here.

    Args:
        cohort: Subjects to fit on (required when the stage list contains
            ``pool`` / cohort-level ``fit``).
        spec: The analysis to run. Strategy is inferred from the stage
            sequence (partition+pool → two_step; pool only → direct_pooling;
            neither → one_step).
        backend: Optional execution backend (parallelism, timeouts, resume).
            Serial when omitted.
        seed: Optional override of ``spec.random_seed``.
        checkpoint: Optional store enabling per-subject resume.
        inspect: Optional step observer for in-memory debugging / QA.
            Unsupported with the process backend.
        writer: Optional streaming writer. When given (``one_step`` design
            only), each subject's habitat map is persisted the moment the
            backend yields it -- a crashed run keeps every completed
            subject on disk.
        retain: ``"all"`` (default) keeps every artefact in memory;
            ``"maps"`` drops voxel-level clustering units (the
            memory-dominant payload); ``"tables"`` additionally drops
            habitat maps and therefore requires ``writer``.
        on_subject_complete: Optional callback fired in the parent process
            as ``(subject, habitat_map, model)`` once per completed subject
            (including checkpoint-resumed ones), before retention stripping.
            Unlike ``inspect`` it is parent-side, so it works with the
            process backend.
        persist_subject_models: With a streaming ``writer``, also write
            ``<subject_id>.habitatmodel`` per subject (one-step fits one
            definition per subject).

    Returns:
        The study result. With default ``retain="all"`` it is entirely in
        memory -- call :meth:`~habit.recipes.result.StudyResult.save` to
        persist it. Under streaming, maps (and optionally per-subject
        models) are already on disk and ``save`` only writes the cohort
        tables and manifest.

    Raises:
        HABITAPIError: If the declared dataflow is inconsistent
            (see :meth:`~habit.spec.specs.HabitatSpec.validate_dataflow`).

    Examples:
        >>> from habit import HabitatSpec, Spec, make_synthetic_cohort
        >>> from habit.recipes import Study
        >>> cohort = make_synthetic_cohort(n_subjects=4, shape=(20, 20, 20), rng=0)
        >>> spec = HabitatSpec(
        ...     name="demo",
        ...     voxel_feature_extractor=Spec("raw", {"modalities": ["T1", "T2"]}),
        ...     supervoxelizer=Spec("kmeans", {"n_supervoxels": 6, "n_init": 5}),
        ...     habitat_model_fitter=Spec(
        ...         "kmeans",
        ...         {"min_habitats": 2, "max_habitats": 3,
        ...          "validation": "elbow", "n_init": 5},
        ...     ),
        ...     habitat_assigner=Spec("nearest_centroid"),
        ...     habitat_features=(Spec("volume"),),
        ...     pooling="cohort",
        ...     random_seed=42,
        ... )
        >>> result = Study(spec=spec).fit_predict(cohort)
        >>> result.habitat_model.n_habitats >= 2
        True
    """
    from habit.report import coerce_report

    effective = _effective_spec(spec, seed)
    effective.validate_dataflow()
    # Force role resolution early so illegal sequences fail before compute.
    resolved = resolve_habitat_stages(effective)
    report = coerce_report(
        report,
        writer=writer,
        retain=retain,
        on_subject_complete=on_subject_complete,
        persist_subject_models=persist_subject_models,
    )
    _validate_streaming_options(design_from_stages(resolved), report)
    _reject_process_inspect(inspect, backend)
    started_at = _now()
    subjects_by_id: Dict[str, Subject] = {
        str(subject.subject_id): subject for subject in cohort
    }

    def _map_units(units_cohort, components, eff_spec, observer):
        return _map_soft(
            units_cohort,
            _ComputeUnits(
                components.pipeline(assigner=None, observer=observer),
                _units_key_prefix(eff_spec),
            ),
            backend=backend,
            checkpoint=checkpoint,
            stage="habitat.units",
        )

    def _map_one_step(one_cohort, components, eff_spec, observer):
        return _map_soft(
            one_cohort,
            _DefineAndLabelWithinSubject(
                components=components,
                assigner_name=eff_spec.habitat_assigner.name,
                assigner_params=tuple(eff_spec.habitat_assigner.params.items()),
                extractors=components.habitat_features,
                seed=eff_spec.random_seed,
                key_prefix=_one_step_key_prefix(eff_spec),
                observer=observer,
                slim_units=(report is not None and report.retain != "all"),
            ),
            backend=backend,
            checkpoint=checkpoint,
            stage="one_step",
            on_item=_build_one_step_item_hook(report, subjects_by_id),
        )

    def _map_labels_compat(
        units_cohort, units, components, eff_spec, model, observer
    ):
        assigner = _build_assigner(model, eff_spec, eff_spec.random_seed)
        return _map_soft_items(
            units_cohort,
            [
                _SubjectUnits(subject, subject_units)
                for subject, subject_units in zip(units_cohort, units)
            ],
            _AssignPrecomputedUnits(
                components.pipeline(assigner=assigner, observer=observer),
                components.habitat_features,
                _label_key_prefix(model),
            ),
            checkpoint=checkpoint,
            stage="habitat.label",
        )

    with _backend_session(backend):
        flow = execute_habitat_dataflow(
            cohort,
            effective,
            map_soft_units=_map_units,
            map_soft_labels=_map_labels_compat,
            map_soft_one_step=_map_one_step,
            seed=effective.random_seed,
            inspect=inspect,
        )

    model = flow.model
    if model is not None:
        model = _with_full_spec_payload(model, flow.spec)
    manifest = _manifest(
        flow.design,
        flow.spec,
        inputs=flow.map_provenances,
        started_at=started_at,
        subject_outcomes=flow.outcomes,
    )
    assigner = (
        None
        if model is None
        else _build_assigner(model, flow.spec, flow.spec.random_seed)
    )
    return StudyResult(
        habitat_model=model,
        pipeline=(
            None
            if flow.model is None
            else flow.components.pipeline(assigner=assigner)
        ),
        features=_cohort_feature_table(
            list(flow.tables),
            list(flow.subject_ids),
            manifest.provenance,
        ),
        habitat_maps=flow.habitat_maps,
        manifest=manifest,
        subject_models=flow.subject_models,
        units=flow.units,
        units_rows=flow.units_rows,
        maps_persisted=(
            report is not None
            and report.writer is not None
            and "habitat_map" in report.persist
        ),
        inspection=inspect,
    )


def _two_step(
    cohort: Cohort,
    spec: HabitatSpec,
    *,
    backend: Optional[ExecutionBackend] = None,
    seed: Optional[int] = None,
    checkpoint: Optional[CheckpointStore] = None,
    inspect: Optional[StepObserver] = None,
    writer: Optional[ResultWriter] = None,
    retain: str = "all",
    on_subject_complete: Optional[
        Callable[[Subject, HabitatMap, HabitatModel], None]
    ] = None,
    persist_subject_models: bool = True,
    report: Optional[Any] = None,
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
        inspect: Optional step observer for in-memory debugging / QA.
            Unsupported with the process backend.

    Returns:
        The study result, entirely in memory. Call
        :meth:`~habit.recipes.result.StudyResult.save` to persist it.

    Raises:
        HABITAPIError: If the spec declares no supervoxelizer, or declares
            ``pooling="none"`` (a subject-level dataflow contradicts the
            cohort-level definition this design fits).

    Examples:
        >>> from habit import HabitatSpec, Spec, make_synthetic_cohort
        >>> from habit.recipes import Study
        >>> cohort = make_synthetic_cohort(n_subjects=6, shape=(24, 24, 24), rng=42)
        >>> spec = HabitatSpec(
        ...     name="demo",
        ...     voxel_feature_extractor=Spec("raw", {"modalities": ["T1", "T2"]}),
        ...     supervoxelizer=Spec("kmeans", {"n_supervoxels": 8, "n_init": 5}),
        ...     habitat_model_fitter=Spec(
        ...         "kmeans",
        ...         {"min_habitats": 2, "max_habitats": 3,
        ...          "validation": "elbow", "n_init": 5},
        ...     ),
        ...     habitat_assigner=Spec("nearest_centroid"),
        ...     habitat_features=(Spec("volume"),),
        ...     random_seed=42,
        ... )
        >>> result = Study(spec=spec, design="two_step").fit_predict(cohort)
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
    if effective.pooling == "none":
        raise HABITAPIError(
            "two_step fits one cohort-level habitat definition, but this "
            "spec declares pooling='none' (subject-level definition). Use "
            "one_step, or drop the pooling declaration from the spec."
        )
    return _fit_habitat(
        cohort,
        effective,
        backend=backend,
        checkpoint=checkpoint,
        inspect=inspect,
        writer=writer,
        retain=retain,
        on_subject_complete=on_subject_complete,
        persist_subject_models=persist_subject_models,
        report=report,
    )


def _direct_pooling(
    cohort: Cohort,
    spec: HabitatSpec,
    *,
    backend: Optional[ExecutionBackend] = None,
    seed: Optional[int] = None,
    checkpoint: Optional[CheckpointStore] = None,
    inspect: Optional[StepObserver] = None,
    writer: Optional[ResultWriter] = None,
    retain: str = "all",
    on_subject_complete: Optional[
        Callable[[Subject, HabitatMap, HabitatModel], None]
    ] = None,
    persist_subject_models: bool = True,
    report: Optional[Any] = None,
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
            scoping as :func:`_two_step`).
        inspect: Optional step observer for in-memory debugging / QA.
            Unsupported with the process backend.

    Returns:
        The study result, entirely in memory.

    Raises:
        HABITAPIError: If the spec declares a supervoxelizer, or declares
            ``pooling="none"`` (a subject-level dataflow contradicts the
            cohort-level definition this design fits).
    """
    effective = _effective_spec(spec, seed)
    if effective.supervoxelizer is not None:
        raise HABITAPIError(
            "direct_pooling clusters voxels directly, but this spec declares "
            f"the supervoxelizer {effective.supervoxelizer.name!r}. Use "
            "two_step, or drop the supervoxelizer from the spec."
        )
    if effective.pooling == "none":
        raise HABITAPIError(
            "direct_pooling fits one cohort-level habitat definition, but "
            "this spec declares pooling='none' (subject-level definition). "
            "Use one_step, or drop the pooling declaration from the spec."
        )
    return _fit_habitat(
        cohort,
        effective,
        backend=backend,
        checkpoint=checkpoint,
        inspect=inspect,
        writer=writer,
        retain=retain,
        on_subject_complete=on_subject_complete,
        persist_subject_models=persist_subject_models,
        report=report,
    )


def _one_step(
    cohort: Cohort,
    spec: HabitatSpec,
    *,
    backend: Optional[ExecutionBackend] = None,
    seed: Optional[int] = None,
    checkpoint: Optional[CheckpointStore] = None,
    inspect: Optional[StepObserver] = None,
    writer: Optional[ResultWriter] = None,
    retain: str = "all",
    on_subject_complete: Optional[
        Callable[[Subject, HabitatMap, HabitatModel], None]
    ] = None,
    persist_subject_models: bool = True,
    report: Optional[Any] = None,
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
        inspect: Optional step observer for in-memory debugging / QA.
            Unsupported with the process backend.
        writer: Optional streaming writer: each subject's habitat map (and
            optionally its ``.habitatmodel``) is persisted as soon as the
            backend yields it, so a crashed run keeps completed subjects.
        retain: ``"all"`` keeps every artefact in memory; ``"maps"`` drops
            voxel-level units; ``"tables"`` additionally drops maps and
            requires ``writer``.
        on_subject_complete: Optional parent-process callback
            ``(subject, habitat_map, model)`` per completed subject.
        persist_subject_models: With ``writer``, also persist each
            subject's own habitat definition.

    Returns:
        The study result, entirely in memory.

    Raises:
        HABITAPIError: If the spec declares a supervoxelizer, a
            cohort-level preprocessing chain, or ``pooling="cohort"`` (a
            cohort-level dataflow contradicts this design).
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
    if effective.pooling == "cohort":
        raise HABITAPIError(
            "one_step defines habitats within each subject, but this spec "
            "declares pooling='cohort'. Use two_step / direct_pooling, or "
            "declare pooling='none'."
        )
    if effective.pooling is None:
        # Canonicalise the undeclared dataflow to the design this alias
        # actually runs, so the recorded spec (manifest, checkpoint keys)
        # states the subject-level dataflow explicitly.
        import dataclasses

        effective = dataclasses.replace(effective, pooling="none")
    return _fit_habitat(
        cohort,
        effective,
        backend=backend,
        checkpoint=checkpoint,
        inspect=inspect,
        writer=writer,
        retain=retain,
        on_subject_complete=on_subject_complete,
        persist_subject_models=persist_subject_models,
        report=report,
    )


def _apply_habitat_model(
    cohort: Cohort,
    spec: HabitatSpec,
    model: HabitatModel,
    *,
    backend: Optional[ExecutionBackend] = None,
    seed: Optional[int] = None,
    checkpoint: Optional[CheckpointStore] = None,
    inspect: Optional[StepObserver] = None,
) -> StudyResult:
    """
    Project a published habitat definition onto a new cohort.

    The prediction half of the ladder, implementing
    :meth:`~habit.recipes.study.Study.predict`: no habitat is defined here,
    the given model is applied. The model's own cohort-level preprocessing
    state is restored and re-applied, because centroids only mean something
    in the feature space they were computed in -- skipping it would still
    produce plausible-looking labels, which is precisely why it is not
    optional.

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
        inspect: Optional step observer for in-memory debugging / QA.
            Unsupported with the process backend.

    Returns:
        The study result for the projected cohort.

    Examples:
        Train on one cohort, persist the model, and project it onto new
        subjects (here the same synthetic cohort doubles as the held-out
        data, which also verifies the save/load/apply round-trip):

        >>> from habit import HabitatModel, HabitatSpec, Spec, make_synthetic_cohort
        >>> from habit.recipes import Study
        >>> cohort = make_synthetic_cohort(n_subjects=5, shape=(20, 20, 20), rng=7)
        >>> spec = HabitatSpec(
        ...     name="demo",
        ...     voxel_feature_extractor=Spec("raw", {"modalities": ["T1", "T2"]}),
        ...     supervoxelizer=Spec("kmeans", {"n_supervoxels": 6, "n_init": 5}),
        ...     habitat_model_fitter=Spec(
        ...         "kmeans",
        ...         {"min_habitats": 2, "max_habitats": 3,
        ...          "validation": "elbow", "n_init": 5},
        ...     ),
        ...     habitat_assigner=Spec("nearest_centroid"),
        ...     habitat_features=(Spec("volume"),),
        ...     random_seed=7,
        ... )
        >>> study = Study(spec=spec, design="two_step").fit(cohort)
        >>> study.model_.save("/tmp/demo.habitatmodel")  # doctest: +SKIP
        >>> projected = Study.from_model("/tmp/demo.habitatmodel").predict(cohort)  # doctest: +SKIP
        >>> len(projected.habitat_maps) == len(cohort)  # doctest: +SKIP
        True
    """
    _reject_process_inspect(inspect, backend)
    started_at = _now()
    effective = _effective_spec(spec, seed)
    effective.validate_dataflow()
    # Name-only stages leave named fields empty until roles are inferred;
    # normalize so assembly (and subject pipelines) see the same components
    # that _fit_habitat used. Fitted cohort preprocess still comes from the
    # model via _with_model_preprocessing below.
    resolved = resolve_habitat_stages(effective)
    effective = normalize_spec_for_execution(effective, resolved)
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
                components.pipeline(assigner=assigner, observer=inspect),
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
        inputs=tuple(habitat_map.provenance for habitat_map in habitat_maps),
        started_at=started_at,
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
        inspection=inspect,
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
    from habit.domain.feature_preprocessing import CohortPreprocessingChain

    state = (model.preprocessing_state or {}).get("cohort_feature_preprocessor")
    if state is None:
        return components
    return dataclasses.replace(
        components,
        cohort_feature_preprocessor=CohortPreprocessingChain.from_state(state),
    )
