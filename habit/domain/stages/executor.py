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
"""Unified stage dataflow executor for habitat analysis (L3).

Three legacy designs (two_step / one_step / direct_pooling) share this one
path. The only level jumps are ``pool`` (fan_in subject→cohort) and
``assign`` (cohort model → per-subject maps). Numeric work reuses
:class:`~habit.domain.pipeline.SubjectPipeline` and
:func:`~habit.domain.pooling.fan_in` so golden baselines stay bit-identical.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from habit.contracts.habitat import HabitatMap, HabitatModel, Supervoxelization
from habit.contracts.inspection import (
    STEP_HABITAT_FEATURES,
    STEP_HABITAT_MAP,
    STEP_SUPERVOXELS_DESCRIBED,
    STEP_SUPERVOXELS_PARTITION,
    STEP_SUPERVOXELS_POSTPROCESSED,
    STEP_SUPERVOXELS_PREPROCESSED,
    STEP_UNITS_COHORT_PREPROCESSED,
    STEP_VOXEL_FEATURES_PREPROCESSED,
    STEP_VOXEL_FEATURES_RAW,
    StepObserver,
    StepRecord,
)
from habit.contracts.subject import Cohort, Subject
from habit.contracts.table import FeatureTable
from habit.domain.assembly import HabitatComponents, build_habitat_components
from habit.domain.pooling import fan_in
from habit.domain.protocols import Seedable
from habit.domain.stages.resolve import (
    ResolvedStage,
    design_from_stages,
    resolve_habitat_stages,
)
from habit.exceptions import HABITAPIError
from habit.spec.specs import (
    ROLE_ASSIGN,
    ROLE_EXTRACT_SUPERVOXEL_FEATURES,
    ROLE_EXTRACT_VOXEL_FEATURES,
    ROLE_FIT,
    ROLE_PARTITION,
    ROLE_POOL,
    ROLE_POSTPROCESS_HABITAT,
    ROLE_POSTPROCESS_SUPERVOXEL,
    ROLE_PREPROCESS,
    ROLE_QUANTIFY,
    HabitatSpec,
    Stage,
    _named_fields_from_stages,
)

__all__ = [
    "StageInspectionAdapter",
    "COHORT_SUBJECT_ID",
    "run_subject_stage_prefix",
    "execute_habitat_dataflow",
    "normalize_spec_for_execution",
]

#: Sentinel subject id for cohort-level inspection records after pool/fit.
COHORT_SUBJECT_ID = "__cohort__"

#: Legacy pipeline step name → role that owns that boundary.
_LEGACY_STEP_ROLE: Dict[str, str] = {
    STEP_VOXEL_FEATURES_RAW: ROLE_EXTRACT_VOXEL_FEATURES,
    STEP_VOXEL_FEATURES_PREPROCESSED: ROLE_PREPROCESS,
    STEP_SUPERVOXELS_PARTITION: ROLE_PARTITION,
    STEP_SUPERVOXELS_POSTPROCESSED: ROLE_POSTPROCESS_SUPERVOXEL,
    STEP_SUPERVOXELS_DESCRIBED: ROLE_EXTRACT_SUPERVOXEL_FEATURES,
    STEP_SUPERVOXELS_PREPROCESSED: ROLE_PREPROCESS,
    STEP_UNITS_COHORT_PREPROCESSED: ROLE_PREPROCESS,
    STEP_HABITAT_MAP: ROLE_ASSIGN,
    STEP_HABITAT_FEATURES: ROLE_QUANTIFY,
}


@dataclass
class StageInspectionAdapter:
    """
    Remap legacy SubjectPipeline step names onto ``{stage}.output`` names.

    Also accepts direct cohort-level records. Outer ``inspect`` may be any
    :class:`~habit.contracts.inspection.StepObserver`.
    """

    inner: StepObserver
    role_to_stage: Dict[str, str]
    #: Remaining preprocess stage names in encounter order (voxel → svx → cohort).
    preprocess_names: List[str]

    def wants(self, step: str) -> bool:
        """Return whether the inner observer wants ``step`` or its remap."""
        if self.inner.wants(step):
            return True
        role = _LEGACY_STEP_ROLE.get(step)
        if role is None:
            return False
        if role == ROLE_PREPROCESS:
            return any(
                self.inner.wants(f"{name}.output") for name in self.preprocess_names
            ) or self.inner.wants(step)
        stage_name = self.role_to_stage.get(role)
        if stage_name is None:
            return self.inner.wants(step)
        return self.inner.wants(f"{stage_name}.output") or self.inner.wants(step)

    def __call__(self, record: StepRecord) -> None:
        """Forward a record, rewriting ``step`` to ``{stage}.output`` when known."""
        role = _LEGACY_STEP_ROLE.get(record.step)
        stage_name: Optional[str] = None
        if role == ROLE_PREPROCESS and self.preprocess_names:
            stage_name = self.preprocess_names.pop(0)
        elif role is not None:
            stage_name = self.role_to_stage.get(role)
        if stage_name is not None:
            mapped = replace(record, step=f"{stage_name}.output")
            if self.inner.wants(mapped.step) or self.inner.wants(record.step):
                self.inner(mapped)
            return
        if self.inner.wants(record.step):
            self.inner(record)

    def emit_cohort(
        self,
        stage_name: str,
        payload: Any,
        produced_by: str,
        fingerprint: Optional[str] = None,
    ) -> None:
        """Emit one cohort-level record after pool / fit."""
        step = f"{stage_name}.output"
        if not self.inner.wants(step):
            return
        self.inner(
            StepRecord(
                step=step,
                subject_id=COHORT_SUBJECT_ID,
                payload=payload,
                produced_by=produced_by,
                spec_fingerprint=fingerprint,
            )
        )


def _role_to_stage_map(
    resolved: Sequence[ResolvedStage],
) -> Tuple[Dict[str, str], List[str]]:
    """Build role→first stage name and ordered preprocess stage names."""
    role_to_stage: Dict[str, str] = {}
    preprocess_names: List[str] = []
    for stage in resolved:
        if stage.role == ROLE_PREPROCESS:
            preprocess_names.append(stage.name)
            continue
        role_to_stage.setdefault(stage.role, stage.name)
    return role_to_stage, preprocess_names


def normalize_spec_for_execution(
    spec: HabitatSpec,
    resolved: Sequence[ResolvedStage],
) -> HabitatSpec:
    """
    Ensure named fields match resolved stages so assembly can build components.

    Args:
        spec: Incoming specification.
        resolved: Role-resolved stages.

    Returns:
        Spec with named fields derived from resolved roles (stages kept).
    """
    staged = tuple(
        Stage(name=item.name, component=item.component.component, role=item.role)
        for item in resolved
    )
    fields = _named_fields_from_stages(staged)
    if not spec._stages_explicit:
        # Re-enter via sugar path so historical fingerprints stay stable.
        return replace(spec, stages=None, **fields)
    return replace(spec, stages=staged, **fields)


def run_subject_stage_prefix(
    subject: Subject,
    spec: HabitatSpec,
    *,
    inspect: Optional[StepObserver] = None,
    resolved: Optional[Sequence[ResolvedStage]] = None,
) -> Supervoxelization:
    """
    Run the subject-level stage prefix on one subject (no Cohort required).

    This is the atomic subject call for everything before ``pool`` / cohort
    ``fit``. Stages after the watershed are not executed here.

    Args:
        subject: One imaging subject.
        spec: Habitat specification.
        inspect: Optional step observer.
        resolved: Optional precomputed role resolution.

    Returns:
        Clustering units for the subject (supervoxels or voxel units).
    """
    resolved_stages = tuple(resolved or resolve_habitat_stages(spec))
    effective = normalize_spec_for_execution(spec, resolved_stages)
    components = build_habitat_components(effective)
    observer = inspect
    if inspect is not None:
        role_map, prep_names = _role_to_stage_map(resolved_stages)
        observer = StageInspectionAdapter(inspect, role_map, list(prep_names))
    return components.pipeline(assigner=None, observer=observer).units(subject)


def _seed_components(components: HabitatComponents, seed: Optional[int]) -> None:
    """Propagate ``seed`` to every Seedable slot on the component bag."""
    if seed is None:
        return
    for attr in (
        "voxel_feature_extractor",
        "supervoxelizer",
        "supervoxel_feature_extractor",
        "habitat_model_fitter",
    ):
        component = getattr(components, attr, None)
        if isinstance(component, Seedable):
            component.set_random_state(int(seed))
    for extractor in components.habitat_features:
        if isinstance(extractor, Seedable):
            extractor.set_random_state(int(seed))


def _fit_cohort_from_units(
    components: HabitatComponents,
    cohort: Cohort,
    units: Sequence[Supervoxelization],
    *,
    adapter: Optional[StageInspectionAdapter],
    pool_stage_name: str,
    fit_stage_name: str,
) -> HabitatModel:
    """Fan-in, optional cohort preprocess, fit -- shared cohort path."""
    pooled = fan_in(units)
    if adapter is not None:
        adapter.emit_cohort(
            pool_stage_name,
            pooled.frame,
            produced_by="stages.pool",
            fingerprint=None,
        )
    chain = components.cohort_feature_preprocessor
    working_units: Sequence[Supervoxelization] = units
    if chain is not None:
        chain.fit(pooled.frame)
        working_units = [
            unit.with_feature_frame(
                chain.transform(unit.feature_frame()),
                produced_by="feature_preprocessing.cohort",
                spec_fingerprint=chain.spec.fingerprint(),
            )
            for unit in units
        ]
        if adapter is not None:
            # Prefer the first post-pool preprocess stage name when present.
            prep_name = next(
                (
                    name
                    for name in adapter.preprocess_names
                    # names already popped for subject path; use role map
                ),
                None,
            )
            # Emit under units.cohort_preprocessed legacy + stage remap via adapter
            adapter(
                StepRecord(
                    step=STEP_UNITS_COHORT_PREPROCESSED,
                    subject_id=COHORT_SUBJECT_ID,
                    payload=fan_in(working_units).frame,
                    produced_by="feature_preprocessing.cohort",
                    spec_fingerprint=chain.spec.fingerprint(),
                )
            )
            del prep_name
    model = components.habitat_model_fitter.fit(working_units, cohort=cohort)
    if chain is not None:
        model = model.with_cohort_preprocessing(chain.state, chain.spec.to_dict())
    if adapter is not None:
        adapter.emit_cohort(
            fit_stage_name,
            {"model_id": model.model_id, "n_habitats": int(model.n_habitats)},
            produced_by="stages.fit",
            fingerprint=None,
        )
    return model


@dataclass(frozen=True)
class HabitatDataflowResult:
    """In-memory products of :func:`execute_habitat_dataflow` (pre-StudyResult)."""

    design: str
    spec: HabitatSpec
    components: HabitatComponents
    model: Optional[HabitatModel]
    subject_models: Dict[str, HabitatModel]
    habitat_maps: Tuple[HabitatMap, ...]
    tables: Tuple[Optional[FeatureTable], ...]
    units: Tuple[Supervoxelization, ...]
    subject_ids: Tuple[str, ...]
    outcomes: Dict[str, str]
    inspection: Optional[StepObserver]


def execute_habitat_dataflow(
    cohort: Cohort,
    spec: HabitatSpec,
    *,
    map_soft_units: Any,
    map_soft_labels: Any,
    map_soft_one_step: Any,
    seed: Optional[int] = None,
    inspect: Optional[StepObserver] = None,
) -> HabitatDataflowResult:
    """
    Run the unified stage dataflow for a cohort.

    The recipe layer supplies ``map_soft_*`` callables so parallelism /
    checkpoint / soft-failure policy stay in L4 while this function owns the
    scientific stage order.

    Args:
        cohort: Subjects to process.
        spec: Effective habitat specification.
        map_soft_units: ``(cohort, op) -> (survivor_cohort, units, failures)``.
        map_soft_labels: label-stage mapper over precomputed units.
        map_soft_one_step: one-step per-subject mapper.
        seed: Random seed already folded into ``spec`` when applicable.
        inspect: Optional step observer.

    Returns:
        Packed dataflow products for the recipe to wrap as StudyResult.
    """
    resolved = resolve_habitat_stages(spec)
    effective = normalize_spec_for_execution(spec, resolved)
    design = design_from_stages(resolved)
    components = build_habitat_components(effective)
    _seed_components(components, seed if seed is not None else effective.random_seed)

    role_map, prep_names = _role_to_stage_map(resolved)
    adapter: Optional[StageInspectionAdapter] = None
    pipeline_observer: Optional[StepObserver] = inspect
    if inspect is not None:
        adapter = StageInspectionAdapter(inspect, role_map, list(prep_names))
        pipeline_observer = adapter

    outcomes: Dict[str, str] = {
        subject.subject_id: "success" for subject in cohort
    }

    if design == "one_step":
        survivors, rows, failures = map_soft_one_step(
            cohort, components, effective, pipeline_observer
        )
        for subject_id, summary in failures.items():
            outcomes[subject_id] = summary
        habitat_maps = tuple(habitat_map for _, habitat_map, _, _ in rows)
        for habitat_map in habitat_maps:
            outcomes[habitat_map.subject_id] = "success"
        return HabitatDataflowResult(
            design=design,
            spec=effective,
            components=components,
            model=None,
            subject_models={
                habitat_map.subject_id: model
                for model, habitat_map, _, _ in rows
            },
            habitat_maps=habitat_maps,
            tables=tuple(table for _, _, table, _ in rows),
            units=tuple(subject_units for _, _, _, subject_units in rows),
            subject_ids=tuple(subject.subject_id for subject in survivors),
            outcomes=outcomes,
            inspection=inspect,
        )

    # Cohort-level designs (two_step / direct_pooling): subject prefix → pool
    # → (cohort preprocess) → fit → assign → quantify.
    units_cohort, units, unit_failures = map_soft_units(
        cohort, components, effective, pipeline_observer
    )
    for subject_id, summary in unit_failures.items():
        outcomes[subject_id] = summary

    pool_name = role_map.get(ROLE_POOL, ROLE_POOL)
    fit_name = role_map.get(ROLE_FIT, ROLE_FIT)
    model = _fit_cohort_from_units(
        components,
        units_cohort,
        units,
        adapter=adapter,
        pool_stage_name=pool_name,
        fit_stage_name=fit_name,
    )
    labelled_cohort, labelled, label_failures = map_soft_labels(
        units_cohort, units, components, effective, model, pipeline_observer
    )
    for subject_id, summary in label_failures.items():
        outcomes[subject_id] = summary
    habitat_maps = tuple(habitat_map for habitat_map, _, _ in labelled)
    for habitat_map in habitat_maps:
        outcomes[habitat_map.subject_id] = "success"
    return HabitatDataflowResult(
        design=design,
        spec=effective,
        components=components,
        model=model,
        subject_models={},
        habitat_maps=habitat_maps,
        tables=tuple(table for _, table, _ in labelled),
        units=tuple(subject_units for _, _, subject_units in labelled),
        subject_ids=tuple(subject.subject_id for subject in labelled_cohort),
        outcomes=outcomes,
        inspection=inspect,
    )
