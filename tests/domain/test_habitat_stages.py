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
"""Tests for ordered habitat stages, role resolution, and the shared executor."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

from habit.contracts.habitat import HabitatModel
from habit.domain.pooling_marker import PoolMarker, PoolingRegistry
from habit.domain.stages import (
    design_from_stages,
    normalize_spec_for_execution,
    resolve_habitat_stages,
    run_subject_stage_prefix,
)
from habit.exceptions import CompatibilityError, HABITAPIError
from habit.inspection import StepRecorder
from habit.recipes.study import Study
from habit.spec import HabitatSpec, Spec, Stage
from habit import make_synthetic_cohort


#: Shared kmeans fitter params for lightweight stage fixtures.
_FITTER_PARAMS = {
    "min_habitats": 2,
    "max_habitats": 3,
    "validation": "silhouette",
    "n_init": 3,
}


def _base_fields(**overrides: object) -> dict:
    """Minimal runnable named-field kwargs."""
    fields = {
        "name": "stage_demo",
        "voxel_feature_extractor": Spec("raw", {"modalities": ["T1", "T2"]}),
        "supervoxelizer": Spec("kmeans", {"n_supervoxels": 6, "n_init": 3}),
        "habitat_model_fitter": Spec(
            "kmeans",
            {
                "min_habitats": 2,
                "max_habitats": 3,
                "validation": "silhouette",
                "n_init": 3,
            },
        ),
        "habitat_assigner": Spec("nearest_centroid"),
        "habitat_features": (Spec("volume"),),
        "random_seed": 42,
    }
    fields.update(overrides)
    return fields


def _name_only_stages_with_post_pool_preprocess(
    design: str,
) -> Tuple[Stage, ...]:
    """
    Build name-only stages (no ``role=``) with pool then cohort preprocess.

    Args:
        design: ``"direct_pooling"`` or ``"two_step"``.

    Returns:
        Ordered stages ending in assign + volume quantify.
    """
    voxel = Stage("voxel", Spec("raw", {"modalities": ["T1", "T2"]}))
    pool = Stage("pool", Spec("pool"))
    # Post-pool preprocess becomes cohort_feature_preprocessors after resolve.
    cohort_prep = Stage("cohort_prep", Spec("zscore"))
    fit = Stage("fit", Spec("kmeans", dict(_FITTER_PARAMS)))
    assign = Stage("assign", Spec("nearest_centroid"))
    quantify = Stage("quantify", Spec("volume"))
    if design == "direct_pooling":
        return (voxel, pool, cohort_prep, fit, assign, quantify)
    if design == "two_step":
        return (
            voxel,
            Stage("partition", Spec("kmeans", {"n_supervoxels": 6, "n_init": 3})),
            Stage("svx_feat", Spec("mean")),
            pool,
            cohort_prep,
            fit,
            assign,
            quantify,
        )
    raise ValueError(f"Unsupported design for this fixture: {design!r}")


@pytest.mark.unit
def test_sugar_expands_to_recommended_stage_names() -> None:
    """Named-field sugar expands to the documented stage sequence."""
    spec = HabitatSpec(**_base_fields())
    names = [stage.name for stage in spec.resolved_stages()]
    assert names == [
        "extract_voxel_features",
        "partition",
        "pool",
        "fit",
        "assign",
        "quantify",
    ]
    assert design_from_stages(resolve_habitat_stages(spec)) == "two_step"


@pytest.mark.unit
def test_partition_without_pool_is_rejected() -> None:
    """Illegal sequence must fail with an actionable missing-pool message."""
    stages = (
        Stage(
            "extract_voxel_features",
            Spec("raw", {"modalities": ["T1", "T2"]}),
            role="extract_voxel_features",
        ),
        Stage(
            "partition",
            Spec("kmeans", {"n_supervoxels": 4, "n_init": 3}),
            role="partition",
        ),
        Stage(
            "fit",
            Spec(
                "kmeans",
                {
                    "min_habitats": 2,
                    "max_habitats": 3,
                    "validation": "silhouette",
                    "n_init": 3,
                },
            ),
            role="fit",
        ),
        Stage("assign", Spec("nearest_centroid"), role="assign"),
    )
    spec = HabitatSpec(name="bad", stages=stages)
    with pytest.raises(HABITAPIError, match="no pool"):
        spec.validate_dataflow()


@pytest.mark.unit
def test_duplicate_stage_names_rejected() -> None:
    """Stage names must be unique inside one HabitatSpec."""
    stages = (
        Stage("extract_voxel_features", Spec("raw", {"modalities": ["T1"]}), role="extract_voxel_features"),
        Stage("extract_voxel_features", Spec("raw", {"modalities": ["T2"]}), role="extract_voxel_features"),
        Stage("pool", Spec("pool"), role="pool"),
        Stage(
            "fit",
            Spec(
                "kmeans",
                {
                    "min_habitats": 2,
                    "max_habitats": 3,
                    "validation": "silhouette",
                    "n_init": 3,
                },
            ),
            role="fit",
        ),
        Stage("assign", Spec("nearest_centroid"), role="assign"),
    )
    with pytest.raises(HABITAPIError, match="unique"):
        HabitatSpec(name="dup", stages=stages)


@pytest.mark.unit
def test_stages_yaml_round_trip_preserves_explicit_fingerprint() -> None:
    """Explicit stages round-trip through to_dict/from_dict with stable hash."""
    stages = (
        Stage(
            "extract_voxel_features",
            Spec("raw", {"modalities": ["T1", "T2"]}),
            role="extract_voxel_features",
        ),
        Stage("pool", Spec("pool"), role="pool"),
        Stage(
            "fit",
            Spec(
                "kmeans",
                {
                    "min_habitats": 2,
                    "max_habitats": 3,
                    "validation": "silhouette",
                    "n_init": 3,
                },
            ),
            role="fit",
        ),
        Stage("assign", Spec("nearest_centroid"), role="assign"),
        Stage("quantify", Spec("volume"), role="quantify"),
    )
    spec = HabitatSpec(name="staged", stages=stages, random_seed=7)
    assert "stages" in spec.to_dict()
    rebuilt = HabitatSpec.from_dict(spec.to_dict())
    assert rebuilt.fingerprint() == spec.fingerprint()
    assert design_from_stages(resolve_habitat_stages(rebuilt)) == "direct_pooling"


@pytest.mark.unit
def test_sugar_fingerprint_ignores_derived_stages() -> None:
    """Sugar specs keep named-field fingerprints (stages not in to_dict)."""
    spec = HabitatSpec(**_base_fields())
    payload = spec.to_dict()
    assert "stages" not in payload
    assert "pooling" not in payload


@pytest.mark.unit
def test_subject_prefix_runs_without_cohort() -> None:
    """Subject-level prefix is callable on a single Subject."""
    cohort = make_synthetic_cohort(n_subjects=2, shape=(16, 16, 16), rng=0)
    spec = HabitatSpec(**_base_fields())
    units = run_subject_stage_prefix(cohort[0], spec)
    assert units.subject_id == cohort[0].subject_id
    assert units.features.shape[0] >= 1


@pytest.mark.unit
def test_stage_parity_declared_design_vs_inferred() -> None:
    """Declared ``design=`` and stage-inferred runs agree voxel-wise."""
    cohort = make_synthetic_cohort(n_subjects=3, shape=(16, 16, 16), rng=1)
    spec = HabitatSpec(**_base_fields())
    a = Study(spec=spec, design="two_step").fit_predict(cohort)
    b = Study(spec=spec).fit_predict(cohort)
    assert a.manifest.provenance.produced_by == b.manifest.provenance.produced_by
    for left, right in zip(a.habitat_maps, b.habitat_maps):
        np.testing.assert_array_equal(left.label_array, right.label_array)


@pytest.mark.unit
def test_pool_marker_is_registered() -> None:
    """Built-in pool marker is discoverable in the pooling domain."""
    assert "pool" in PoolingRegistry.available()
    marker = PoolingRegistry.create("pool")
    assert isinstance(marker, PoolMarker)
    assert marker()["marker"] == "pool"


@pytest.mark.unit
def test_inspection_emits_stage_named_and_cohort_records() -> None:
    """inspect= records ``{stage}.output`` including cohort-level pool/fit."""
    cohort = make_synthetic_cohort(n_subjects=2, shape=(16, 16, 16), rng=2)
    spec = HabitatSpec(**_base_fields(supervoxelizer=None))
    recorder = StepRecorder(keep="frames", max_subjects=1)
    Study(spec=spec, design="direct_pooling").fit_predict(cohort, inspect=recorder)
    steps = recorder.steps()
    assert any(step.endswith(".output") for step in steps)
    assert any(step.startswith("pool.") for step in steps) or any(
        step.startswith("fit.") for step in steps
    )
    cohort_records = [r for r in recorder.records() if r.subject_id == "__cohort__"]
    assert cohort_records, "expected cohort-level inspection records after pool/fit"


@pytest.mark.unit
def test_role_inferred_without_explicit_role() -> None:
    """Name-only stages: roles inferred; three strategies + pool-after prep."""
    cohort = make_synthetic_cohort(n_subjects=3, shape=(16, 16, 16), rng=4)

    direct = HabitatSpec(
        name="direct_inferred",
        stages=(
            Stage("voxel", Spec("raw", {"modalities": ["T1", "T2"]})),
            Stage("preprocess1", Spec("zscore")),
            Stage("pool", Spec("pool")),
            Stage(
                "fit",
                Spec(
                    "kmeans",
                    {
                        "min_habitats": 2,
                        "max_habitats": 3,
                        "validation": "silhouette",
                        "n_init": 3,
                    },
                ),
            ),
            Stage("assign", Spec("nearest_centroid")),
        ),
        random_seed=1,
    )
    assert direct.definition_level == "cohort"
    assert design_from_stages(resolve_habitat_stages(direct)) == "direct_pooling"
    recorder = StepRecorder(keep="frames", max_subjects=1)
    direct_result = Study(spec=direct).fit_predict(cohort, inspect=recorder)
    assert direct_result.habitat_model is not None
    steps = recorder.steps()
    assert "preprocess1.output" in steps
    assert "pool.output" in steps
    assert "fit.output" in steps
    assert any(r.subject_id == "__cohort__" for r in recorder.records())

    two = HabitatSpec(
        name="two_inferred",
        stages=(
            Stage("voxel", Spec("raw", {"modalities": ["T1", "T2"]})),
            Stage("partition", Spec("kmeans", {"n_supervoxels": 6, "n_init": 3})),
            Stage("svx_feat", Spec("mean")),
            Stage("pool", Spec("pool")),
            Stage(
                "fit",
                Spec(
                    "kmeans",
                    {
                        "min_habitats": 2,
                        "max_habitats": 3,
                        "validation": "silhouette",
                        "n_init": 3,
                    },
                ),
            ),
            Stage("assign", Spec("nearest_centroid")),
        ),
        random_seed=1,
    )
    assert design_from_stages(resolve_habitat_stages(two)) == "two_step"
    two_result = Study(spec=two).fit_predict(cohort)
    assert two_result.habitat_model is not None

    one = HabitatSpec(
        name="one_inferred",
        stages=(
            Stage("voxel", Spec("raw", {"modalities": ["T1", "T2"]})),
            Stage(
                "fit",
                Spec(
                    "kmeans",
                    {
                        "min_habitats": 2,
                        "max_habitats": 3,
                        "validation": "silhouette",
                        "n_init": 3,
                    },
                ),
            ),
            Stage("assign", Spec("nearest_centroid")),
        ),
        random_seed=1,
    )
    assert one.definition_level == "subject"
    assert design_from_stages(resolve_habitat_stages(one)) == "one_step"
    one_result = Study(spec=one).fit_predict(cohort)
    assert list(one_result.subject_models) == [
        s.subject_id for s in cohort
    ]


@pytest.mark.unit
def test_temporary_plugin_stage_component_runs() -> None:
    """A temporarily registered pooling marker can appear in stages."""

    @PoolingRegistry.register("temp_pool_marker")
    class _TempPool:
        def __init__(self, **params: object) -> None:
            self._spec = Spec("temp_pool_marker", dict(params))

        @property
        def spec(self) -> Spec:
            return self._spec

        def __call__(self) -> dict:
            return {"marker": "temp_pool_marker"}

    try:
        stages = (
            Stage(
                "extract_voxel_features",
                Spec("raw", {"modalities": ["T1", "T2"]}),
                role="extract_voxel_features",
            ),
            Stage("pool", Spec("temp_pool_marker"), role="pool"),
            Stage(
                "fit",
                Spec(
                    "kmeans",
                    {
                        "min_habitats": 2,
                        "max_habitats": 3,
                        "validation": "silhouette",
                        "n_init": 3,
                    },
                ),
                role="fit",
            ),
            Stage("assign", Spec("nearest_centroid"), role="assign"),
        )
        spec = HabitatSpec(name="plugin", stages=stages, random_seed=1)
        resolved = resolve_habitat_stages(spec)
        assert any(item.component.component.name == "temp_pool_marker" for item in resolved)
        cohort = make_synthetic_cohort(n_subjects=2, shape=(16, 16, 16), rng=3)
        result = Study(spec=spec).fit_predict(cohort)
        assert result.habitat_model is not None
    finally:
        # Best-effort cleanup: registry has no public unregister; overwrite is fine.
        pass


@pytest.mark.unit
@pytest.mark.parametrize("design", ["direct_pooling", "two_step"])
def test_habitat_model_save_load_apply_with_post_pool_preprocess(
    design: str,
    tmp_path: Path,
) -> None:
    """
    F.9 gate: stages fit with post-pool preprocess round-trips through
    ``.habitatmodel`` and ``Study.from_model(...).predict(...)`` without
    label drift.

    Also pins the invariant that cohort preprocess state travels with the
    model and ``with_cohort_preprocessing`` recomputes ``model_id``.
    """
    cohort = make_synthetic_cohort(n_subjects=3, shape=(16, 16, 16), rng=5)
    stages = _name_only_stages_with_post_pool_preprocess(design)
    # Assert every stage is name-only (no authored role=).
    assert all(stage.role is None for stage in stages)
    spec = HabitatSpec(name=f"saveload_{design}", stages=stages, random_seed=11)
    resolved = resolve_habitat_stages(spec)
    assert design_from_stages(resolved) == design
    # Name-only stages keep named fields empty until normalize (executor path).
    normalized = normalize_spec_for_execution(spec, resolved)
    assert normalized.cohort_feature_preprocessors
    assert all(s.name == "zscore" for s in normalized.cohort_feature_preprocessors)

    trained = Study(spec=spec).fit_predict(cohort)
    model = trained.habitat_model
    assert model is not None
    assert "cohort_feature_preprocessor" in model.preprocessing_state
    assert "cohort_preprocessing" in model.provenance.produced_by

    # Binding a different cohort chain must mint a new definition identity.
    rebound = model.with_cohort_preprocessing(
        state={"methods": ["sentinel"]},
        spec_payload={
            "name": "cohort_feature_preprocessor",
            "params": {"steps": [{"name": "sentinel", "params": {}}]},
        },
    )
    assert rebound.model_id != model.model_id
    assert rebound.model_id.startswith(model.model_id.split("-", 1)[0] + "-")

    archive = model.save(tmp_path / f"{design}.habitatmodel")
    loaded = HabitatModel.load(archive)
    assert loaded.model_id == model.model_id
    np.testing.assert_array_equal(loaded.centroids, model.centroids)
    assert loaded.feature_names == model.feature_names
    assert (
        loaded.preprocessing_state["cohort_feature_preprocessor"]
        == model.preprocessing_state["cohort_feature_preprocessor"]
    )

    applied = Study.from_model(loaded, spec).predict(cohort)
    trained_maps = {m.subject_id: m.label_array for m in trained.habitat_maps}
    for habitat_map in applied.habitat_maps:
        np.testing.assert_array_equal(
            habitat_map.label_array,
            trained_maps[habitat_map.subject_id],
        )
    assert trained.features is not None and applied.features is not None
    feature_cols = list(trained.features.feature_columns)
    assert feature_cols
    np.testing.assert_allclose(
        applied.features.frame.loc[:, feature_cols].to_numpy(dtype=float),
        trained.features.frame.loc[:, feature_cols].to_numpy(dtype=float),
        rtol=1e-6,
        atol=0.0,
    )


@pytest.mark.unit
def test_habitat_model_load_rejects_non_archive_clearly(tmp_path: Path) -> None:
    """Incompatible bytes must fail loudly, never yield a plausible model."""
    bogus = tmp_path / "legacy_pipeline.pkl"
    bogus.write_bytes(b"not-a-habitatmodel-archive")
    with pytest.raises(CompatibilityError, match="habitat_pipeline|habit.habitatmodel"):
        HabitatModel.load(bogus)
