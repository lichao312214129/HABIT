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
"""Step inspection: optional in-memory observation of habitat pipelines."""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

from habit import (
    HABITAPIError,
    HabitatSpec,
    Spec,
    StepRecorder,
    make_synthetic_cohort,
)
from habit.contracts.inspection import (
    STEP_HABITAT_FEATURES,
    STEP_HABITAT_MAP,
    STEP_SUPERVOXELS_DESCRIBED,
    STEP_SUPERVOXELS_PARTITION,
    STEP_UNITS_COHORT_PREPROCESSED,
    STEP_VOXEL_FEATURES_PREPROCESSED,
    STEP_VOXEL_FEATURES_RAW,
    StepRecord,
)
from habit.execution.process_pool import ProcessPoolBackend
import habit.recipes as recipes


def _two_step_spec(*, with_sv_fx: bool = False, with_voxel_prep: bool = False) -> HabitatSpec:
    """Return a small synthetic two-step HabitatSpec."""
    sv_fx = None
    if with_sv_fx:
        sv_fx = Spec(
            "concat",
            {
                "children": [
                    {"name": "mean", "params": {"modality": "T1"}},
                    {"name": "std", "params": {"modality": "T1", "as_": "t1_spread"}},
                ],
            },
        )
    voxel_prep = ()
    if with_voxel_prep:
        voxel_prep = (
            Spec("winsorize", {"winsor_limits": (0.05, 0.05), "across_features": False}),
        )
    return HabitatSpec(
        name="inspect_demo",
        voxel_feature_extractor=Spec("raw", {"modalities": ["T1", "T2"]}),
        voxel_feature_preprocessors=voxel_prep,
        supervoxelizer=Spec("kmeans", {"n_supervoxels": 6, "n_init": 3}),
        supervoxel_feature_extractor=sv_fx,
        habitat_model_fitter=Spec(
            "kmeans",
            {
                "min_habitats": 2,
                "max_habitats": 3,
                "validation": "silhouette",
                "n_init": 3,
            },
        ),
        habitat_assigner=Spec("nearest_centroid"),
        habitat_features=(Spec("volume"), Spec("msi")),
        random_seed=11,
    )


class _CountingObserver:
    """Spy observer that records which steps were delivered."""

    def __init__(self, steps: List[str] | None = None) -> None:
        self._steps = None if steps is None else frozenset(steps)
        self.calls: List[str] = []

    def wants(self, step: str) -> bool:
        return self._steps is None or step in self._steps

    def __call__(self, record: StepRecord) -> None:
        self.calls.append(record.step)


def test_inspect_none_matches_baseline_labels_and_fingerprint() -> None:
    """Default inspect=None must not change scientific outputs."""
    cohort = make_synthetic_cohort(n_subjects=3, shape=(12, 12, 12), rng=3)
    spec = _two_step_spec()
    baseline = recipes.two_step(cohort, spec)
    again = recipes.two_step(cohort, spec, inspect=None)
    with_rec = recipes.two_step(cohort, spec, inspect=StepRecorder(max_subjects=1))
    assert again.inspection is None
    assert with_rec.inspection is not None
    # Observer must never enter the analysis declaration / fingerprint.
    assert baseline.manifest.spec_payload == again.manifest.spec_payload
    assert baseline.manifest.spec_payload == with_rec.manifest.spec_payload
    assert len(baseline.habitat_maps) == len(with_rec.habitat_maps)
    for left, right in zip(baseline.habitat_maps, with_rec.habitat_maps):
        np.testing.assert_array_equal(left.label_array, right.label_array)
    assert list(baseline.features.feature_columns) == list(
        with_rec.features.feature_columns
    )
    np.testing.assert_allclose(
        baseline.features.frame[list(baseline.features.feature_columns)].to_numpy(),
        with_rec.features.frame[list(with_rec.features.feature_columns)].to_numpy(),
        rtol=1e-6,
        atol=1e-6,
        equal_nan=True,
    )


def test_two_step_captures_core_steps_with_recorder() -> None:
    """two_step with mean/std extractor emits the expected inspection steps."""
    cohort = make_synthetic_cohort(n_subjects=2, shape=(12, 12, 12), rng=5)
    spec = _two_step_spec(with_sv_fx=True, with_voxel_prep=True)
    rec = StepRecorder(max_subjects=2)
    result = recipes.two_step(cohort, spec, inspect=rec)
    assert result.inspection is rec
    steps = set(rec.steps())
    assert STEP_VOXEL_FEATURES_RAW in steps
    assert STEP_VOXEL_FEATURES_PREPROCESSED in steps
    assert STEP_SUPERVOXELS_PARTITION in steps
    assert STEP_SUPERVOXELS_DESCRIBED in steps
    assert STEP_UNITS_COHORT_PREPROCESSED in steps
    assert STEP_HABITAT_MAP in steps
    assert STEP_HABITAT_FEATURES in steps
    summary = rec.summary()
    assert not summary.empty
    sid = cohort[0].subject_id
    raw = rec.frame(STEP_VOXEL_FEATURES_RAW, sid)
    described = rec.frame(STEP_SUPERVOXELS_DESCRIBED, sid)
    cohort_units = rec.frame(STEP_UNITS_COHORT_PREPROCESSED, sid)
    assert raw.shape[0] > 0
    assert described.shape[0] == 6
    assert result.habitat_model is not None
    assert list(cohort_units.columns) == list(result.habitat_model.feature_names)


def test_steps_filter_skips_unwanted_observer_calls() -> None:
    """Filtered steps must not invoke the observer."""
    cohort = make_synthetic_cohort(n_subjects=2, shape=(10, 10, 10), rng=6)
    spec = _two_step_spec(with_sv_fx=True)
    spy = _CountingObserver(steps=[STEP_SUPERVOXELS_DESCRIBED])
    recipes.two_step(cohort, spec, inspect=spy)
    assert spy.calls
    assert set(spy.calls) == {STEP_SUPERVOXELS_DESCRIBED}


def test_max_subjects_and_subjects_filters() -> None:
    """Subject filters limit retained records."""
    cohort = make_synthetic_cohort(n_subjects=3, shape=(10, 10, 10), rng=7)
    spec = _two_step_spec()
    only = cohort[0].subject_id
    rec = StepRecorder(steps=[STEP_VOXEL_FEATURES_RAW], subjects=[only])
    recipes.two_step(cohort, spec, inspect=rec)
    assert rec.subjects() == (only,)
    capped = StepRecorder(steps=[STEP_VOXEL_FEATURES_RAW], max_subjects=1)
    recipes.two_step(cohort, spec, inspect=capped)
    assert len(capped.subjects()) == 1


def test_process_backend_rejects_inspect() -> None:
    """Process backend + inspect must raise a clear HABITAPIError."""
    cohort = make_synthetic_cohort(n_subjects=2, shape=(10, 10, 10), rng=8)
    spec = _two_step_spec()
    rec = StepRecorder(steps=[STEP_VOXEL_FEATURES_RAW], max_subjects=1)
    with pytest.raises(HABITAPIError, match="serial|workers=1"):
        recipes.two_step(cohort, spec, backend=ProcessPoolBackend(workers=1), inspect=rec)


def test_one_step_and_direct_pooling_and_apply_smoke() -> None:
    """Other habitat recipes accept inspect= without error."""
    cohort = make_synthetic_cohort(n_subjects=2, shape=(10, 10, 10), rng=9)
    dp_spec = HabitatSpec(
        name="dp_inspect",
        voxel_feature_extractor=Spec("raw", {"modalities": ["T1", "T2"]}),
        supervoxelizer=None,
        habitat_model_fitter=Spec(
            "kmeans",
            {"min_habitats": 2, "max_habitats": 3, "validation": "inertia", "n_init": 3},
        ),
        habitat_assigner=Spec("nearest_centroid"),
        habitat_features=(Spec("volume"),),
        random_seed=9,
    )
    rec_dp = StepRecorder(steps=[STEP_VOXEL_FEATURES_RAW, STEP_HABITAT_MAP], max_subjects=1)
    dp = recipes.direct_pooling(cohort, dp_spec, inspect=rec_dp)
    assert STEP_VOXEL_FEATURES_RAW in rec_dp.steps()
    assert STEP_HABITAT_MAP in rec_dp.steps()

    os_spec = HabitatSpec(
        name="os_inspect",
        voxel_feature_extractor=Spec("raw", {"modalities": ["T1", "T2"]}),
        supervoxelizer=None,
        habitat_model_fitter=Spec(
            "kmeans",
            {"min_habitats": 2, "max_habitats": 3, "validation": "inertia", "n_init": 3},
        ),
        habitat_assigner=Spec("nearest_centroid"),
        habitat_features=(Spec("volume"),),
        random_seed=9,
    )
    rec_os = StepRecorder(steps=[STEP_VOXEL_FEATURES_RAW, STEP_HABITAT_MAP], max_subjects=1)
    os_result = recipes.one_step(cohort, os_spec, inspect=rec_os)
    assert os_result.habitat_model is None
    assert STEP_HABITAT_MAP in rec_os.steps()

    train = recipes.two_step(cohort, _two_step_spec())
    assert train.habitat_model is not None
    rec_apply = StepRecorder(steps=[STEP_HABITAT_MAP], max_subjects=1)
    applied = recipes.apply_habitat_model(
        cohort, _two_step_spec(), train.habitat_model, inspect=rec_apply
    )
    assert len(applied.habitat_maps) == len(cohort)
    assert STEP_HABITAT_MAP in rec_apply.steps()
