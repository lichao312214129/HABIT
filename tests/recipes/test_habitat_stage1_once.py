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
"""Regression: train-path Stage-1 extractors run once per subject.

The historical bug was cohort fit recipes (and the sklearn adapter) computing
clustering units, then calling ``pipeline(subject)`` / ``extract_features``
in the label stage -- which re-ran ``voxel_radiomics`` (~2×). These tests
count extractor calls so the reuse path cannot regress silently.
"""

from __future__ import annotations

from typing import Any, List

import pytest

from habit.compat.sklearn import as_estimator
from habit.contracts.habitat import VoxelFeatureField
from habit.contracts.subject import Subject
from habit.datasets import make_synthetic_cohort
from habit.domain.voxel_features.raw import RawVoxelFeatures
from habit.recipes.study import Study
from habit.spec.specs import HabitatSpec, Spec


class _CountingRaw(RawVoxelFeatures):
    """Raw voxel extractor that records how often Stage-1 runs."""

    def __init__(self, *args: Any, call_log: List[str], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._call_log = call_log

    def __call__(self, subject: Subject) -> VoxelFeatureField:
        self._call_log.append(subject.subject_id)
        return super().__call__(subject)


def _install_counting_raw(monkeypatch: pytest.MonkeyPatch, call_log: List[str]) -> None:
    """
    Force assembly to use :class:`_CountingRaw` for ``raw`` voxel features.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        call_log: Mutable list receiving subject ids on each Stage-1 call.
    """

    def _factory(**params: Any) -> _CountingRaw:
        return _CountingRaw(call_log=call_log, **params)

    # Patch the assembly import site (not trees.py): build_habitat_components
    # binds ``build_voxel_extractor`` from ``habit.domain.assembly``.
    monkeypatch.setattr(
        "habit.domain.assembly.build_voxel_extractor",
        lambda spec: (
            _factory(**spec.params)
            if spec.name == "raw" and "children" not in spec.params
            else pytest.fail(f"unexpected voxel extractor spec: {spec.name}")
        ),
    )


def _direct_pooling_spec() -> HabitatSpec:
    """Small direct-pooling spec (no supervoxels; Stage-1 is raw voxels)."""
    return HabitatSpec(
        name="stage1_once_direct",
        voxel_feature_extractor=Spec(
            name="raw", params={"modalities": ["T1", "T2"]}
        ),
        supervoxelizer=None,
        habitat_model_fitter=Spec(
            name="kmeans",
            params={
                "n_habitats": 2,
                "n_init": 3,
                "min_habitats": 2,
                "max_habitats": 3,
            },
        ),
        habitat_assigner=Spec(name="nearest_centroid", params={}),
        habitat_features=(Spec(name="volume", params={}),),
        random_seed=0,
    )


def _two_step_spec() -> HabitatSpec:
    """Small two-step spec (Stage-1 still goes through the voxel extractor)."""
    return HabitatSpec(
        name="stage1_once_two_step",
        voxel_feature_extractor=Spec(
            name="raw", params={"modalities": ["T1", "T2"]}
        ),
        supervoxelizer=Spec(name="kmeans", params={"n_supervoxels": 6, "n_init": 3}),
        habitat_model_fitter=Spec(
            name="kmeans",
            params={
                "n_habitats": 2,
                "n_init": 3,
                "min_habitats": 2,
                "max_habitats": 3,
            },
        ),
        habitat_assigner=Spec(name="nearest_centroid", params={}),
        habitat_features=(Spec(name="volume", params={}),),
        random_seed=0,
    )


def _one_step_spec() -> HabitatSpec:
    """One-step spec: habitats defined inside each subject."""
    return HabitatSpec(
        name="stage1_once_one_step",
        voxel_feature_extractor=Spec(
            name="raw", params={"modalities": ["T1", "T2"]}
        ),
        supervoxelizer=None,
        habitat_model_fitter=Spec(
            name="kmeans",
            params={
                "n_habitats": 2,
                "n_init": 3,
                "min_habitats": 2,
                "max_habitats": 3,
            },
        ),
        habitat_assigner=Spec(name="nearest_centroid", params={}),
        habitat_features=(Spec(name="volume", params={}),),
        random_seed=0,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "design,spec_factory",
    [
        ("direct_pooling", _direct_pooling_spec),
        ("two_step", _two_step_spec),
        ("one_step", _one_step_spec),
    ],
)
def test_train_recipe_runs_stage1_once_per_subject(
    monkeypatch: pytest.MonkeyPatch,
    design: str,
    spec_factory: Any,
) -> None:
    """Train studies must not re-extract voxel features in the label stage."""
    call_log: List[str] = []
    _install_counting_raw(monkeypatch, call_log)
    cohort = make_synthetic_cohort(
        n_subjects=3, modalities=("T1", "T2"), shape=(12, 12, 12), rng=1
    )
    recipe = Study(spec=spec_factory(), design=design)
    result = recipe.fit_predict(cohort)
    assert len(result.habitat_maps) == 3
    assert call_log == [subject.subject_id for subject in cohort]


@pytest.mark.unit
def test_apply_habitat_model_recomputes_stage1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Predict/apply must recompute Stage-1 from images (no train units)."""
    call_log: List[str] = []
    _install_counting_raw(monkeypatch, call_log)
    cohort = make_synthetic_cohort(
        n_subjects=2, modalities=("T1", "T2"), shape=(12, 12, 12), rng=2
    )
    spec = _direct_pooling_spec()
    trained = Study(spec=spec, design="direct_pooling").fit_predict(cohort)
    assert trained.habitat_model is not None
    train_calls = list(call_log)
    assert train_calls == [subject.subject_id for subject in cohort]

    call_log.clear()
    projected = Study.from_model(trained.habitat_model, spec).predict(cohort)
    assert len(projected.habitat_maps) == 2
    assert call_log == [subject.subject_id for subject in cohort]


@pytest.mark.unit
def test_sklearn_fit_transform_runs_stage1_once_per_subject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HabitatFeaturesEstimator.fit_transform must reuse fit-time units."""
    call_log: List[str] = []
    _install_counting_raw(monkeypatch, call_log)
    cohort = make_synthetic_cohort(
        n_subjects=3, modalities=("T1", "T2"), shape=(12, 12, 12), rng=3
    )
    matrix = as_estimator(_direct_pooling_spec()).fit_transform(list(cohort))
    assert matrix.shape[0] == 3
    assert call_log == [subject.subject_id for subject in cohort]
