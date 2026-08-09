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

import numpy as np
import pytest

from habit.domain.pooling_marker import PoolMarker, PoolingRegistry
from habit.domain.stages import (
    design_from_stages,
    resolve_habitat_stages,
    run_subject_stage_prefix,
)
from habit.exceptions import HABITAPIError
from habit.inspection import StepRecorder
from habit.spec import HabitatSpec, Spec, Stage
from habit import make_synthetic_cohort
import habit.recipes as recipes


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
def test_stage_parity_two_step_alias_vs_fit_habitat() -> None:
    """two_step alias and fit_habitat agree voxel-wise on labels."""
    cohort = make_synthetic_cohort(n_subjects=3, shape=(16, 16, 16), rng=1)
    spec = HabitatSpec(**_base_fields())
    a = recipes.two_step(cohort, spec)
    b = recipes.fit_habitat(cohort, spec)
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
    recipes.direct_pooling(cohort, spec, inspect=recorder)
    steps = recorder.steps()
    assert any(step.endswith(".output") for step in steps)
    assert any(step.startswith("pool.") for step in steps) or any(
        step.startswith("fit.") for step in steps
    )
    cohort_records = [r for r in recorder.records() if r.subject_id == "__cohort__"]
    assert cohort_records, "expected cohort-level inspection records after pool/fit"


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
        result = recipes.fit_habitat(cohort, spec)
        assert result.habitat_model is not None
    finally:
        # Best-effort cleanup: registry has no public unregister; overwrite is fine.
        pass
