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
# See the License for the specific license governing permissions and
# limitations under the License.
#
"""Windows-spawn picklability and serial/process two-step habitat parity.

After the v2 physical-package rename, ``SubjectPipeline`` and the recipe
stage operators must still cross a ``spawn`` process-pool boundary. This
module checks that they pickle, then runs a minimal two-step habitat study
under ``SerialBackend`` and ``ProcessPoolBackend(workers=2)`` and requires
identical habitat labels (0 mismatch).
"""

from __future__ import annotations

import pickle
from typing import Any, List

import numpy as np
import pytest

from habit.datasets import make_synthetic_cohort
from habit.execution import ProcessPoolBackend, SerialBackend
from habit.pipeline.assembly import build_habitat_components
from habit.recipes.habitat import _ComputeUnits
from habit.recipes.study import Study
from habit.spec import HabitatSpec, Spec


def _two_step_spec() -> HabitatSpec:
    """Return a small seeded two-step spec (no heavy radiomics families)."""
    return HabitatSpec(
        name="process_pool_two_step",
        voxel_feature_extractor=Spec("raw", {"modalities": ["T1", "T2"]}),
        supervoxelizer=Spec("kmeans", {"n_supervoxels": 6, "n_init": 3}),
        habitat_model_fitter=Spec(
            "kmeans",
            {
                "min_habitats": 2,
                "max_habitats": 3,
                "validation": "elbow",
                "n_init": 3,
            },
        ),
        habitat_assigner=Spec("nearest_centroid"),
        habitat_features=(Spec("volume"),),
        random_seed=21,
    )


def _roundtrip(value: Any) -> Any:
    """
    Pickle and unpickle ``value`` under the default protocol.

    Args:
        value: Object that must survive a ``spawn`` worker import.

    Returns:
        The reconstructed object (identity is not required).
    """
    return pickle.loads(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def _stage_payloads(pipeline: Any) -> List[Any]:
    """
    Collect the live stage objects a two-step units pipeline carries.

    Args:
        pipeline: Fit-time :class:`~habit.pipeline.SubjectPipeline`.

    Returns:
        Non-``None`` stage attributes plus the pipeline itself.
    """
    names = (
        "voxel_feature_extractor",
        "supervoxelizer",
        "habitat_assigner",
        "supervoxel_feature_extractor",
        "voxel_feature_preprocessor",
        "supervoxel_feature_preprocessor",
        "cohort_feature_preprocessor",
        "postprocess_supervoxel",
        "postprocess_habitat",
    )
    payloads: List[Any] = [pipeline]
    for name in names:
        stage = getattr(pipeline, name)
        if stage is not None:
            payloads.append(stage)
    return payloads


@pytest.mark.integration
def test_two_step_pipeline_and_operators_are_picklable() -> None:
    """SubjectPipeline and _ComputeUnits survive pickle after the v2 rename."""
    spec = _two_step_spec()
    components = build_habitat_components(spec)
    pipeline = components.pipeline(assigner=None)
    operator = _ComputeUnits(pipeline, key_prefix="units:v2")

    for payload in (*_stage_payloads(pipeline), components, operator):
        restored = _roundtrip(payload)
        assert type(restored) is type(payload)


@pytest.mark.integration
def test_process_pool_two_step_matches_serial_labels() -> None:
    """ProcessPoolBackend(workers=2) labels match SerialBackend exactly."""
    cohort = make_synthetic_cohort(n_subjects=4, shape=(12, 12, 12), rng=21)
    spec = _two_step_spec()

    serial_result = Study(spec=spec).fit_predict(
        cohort, backend=SerialBackend()
    )
    parallel_result = Study(spec=spec).fit_predict(
        cohort,
        backend=ProcessPoolBackend(workers=2, auto_retry_rounds=0),
    )

    assert serial_result.habitat_model is not None
    assert parallel_result.habitat_model is not None
    assert (
        serial_result.habitat_model.n_habitats
        == parallel_result.habitat_model.n_habitats
    )
    assert len(serial_result.habitat_maps) == len(parallel_result.habitat_maps)

    serial_by_id = {item.subject_id: item for item in serial_result.habitat_maps}
    parallel_by_id = {
        item.subject_id: item for item in parallel_result.habitat_maps
    }
    assert set(serial_by_id) == set(parallel_by_id)

    mismatches = 0
    for subject_id, serial_map in serial_by_id.items():
        parallel_map = parallel_by_id[subject_id]
        if not np.array_equal(serial_map.label_array, parallel_map.label_array):
            mismatches += 1
    assert mismatches == 0
