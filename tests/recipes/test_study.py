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
"""Object-style :class:`~habit.recipes.study.Study` entry points."""

from __future__ import annotations

import pytest

from habit.datasets import make_synthetic_cohort
from habit.recipes.study import Study, direct_pooling_habitat, one_step_habitat, two_step_habitat
from habit.recipes.result import StudyResult
from habit.spec.specs import HabitatSpec, Spec


@pytest.mark.unit
def test_study_fit_delegates_to_two_step_recipe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Study.fit forwards backend and checkpoint kwargs to the recipe."""
    captured: dict[str, object] = {}
    cohort = make_synthetic_cohort(n_subjects=2, modalities=("T1",), shape=(8, 8, 8), rng=0)
    spec = HabitatSpec(
        name="unit",
        voxel_feature_extractor=Spec(name="raw", params={"modalities": ["T1"]}),
        supervoxelizer=Spec(name="kmeans", params={"n_supervoxels": 4}),
        habitat_model_fitter=Spec(name="kmeans", params={"n_habitats": 2}),
        habitat_assigner=Spec(name="nearest_centroid", params={}),
        random_seed=0,
    )

    def _fake_two_step(cohort_arg, spec_arg, *, backend=None, seed=None, checkpoint=None):
        captured.update(
            {
                "cohort": cohort_arg,
                "spec": spec_arg,
                "backend": backend,
                "seed": seed,
                "checkpoint": checkpoint,
            }
        )
        return StudyResult(
            habitat_model=None,
            pipeline=None,
            features=object(),  # type: ignore[arg-type]
            habitat_maps=(),
            manifest=object(),  # type: ignore[arg-type]
        )

    monkeypatch.setitem(
        __import__("habit.recipes.study", fromlist=["_RECIPE_BY_DESIGN"])._RECIPE_BY_DESIGN,
        "two_step",
        _fake_two_step,
    )
    study = Study(spec=spec, design="two_step")
    result = study.fit(cohort, seed=7)

    assert isinstance(result, StudyResult)
    assert captured["cohort"] is cohort
    assert captured["spec"] is spec
    assert captured["seed"] == 7


@pytest.mark.unit
def test_two_step_habitat_factory_builds_spec() -> None:
    """Convenience factory wires modalities and habitat count into the spec."""
    study = two_step_habitat(
        modalities=["T1", "T2"],
        n_supervoxels=32,
        n_habitats=4,
        habitat_features=["volume"],
        random_seed=11,
    )
    assert study.design == "two_step"
    assert study.spec.supervoxelizer is not None
    assert study.spec.habitat_model_fitter.params["n_habitats"] == 4
    assert study.spec.voxel_feature_extractor.params["modalities"] == ["T1", "T2"]


@pytest.mark.unit
def test_one_step_and_direct_pooling_factories_omit_supervoxelizer() -> None:
    """Design-specific factories must not declare a supervoxel stage."""
    one = one_step_habitat(modalities=["T1"], n_habitats=3)
    pooled = direct_pooling_habitat(modalities=["T1"], n_habitats=3)
    assert one.spec.supervoxelizer is None
    assert pooled.spec.supervoxelizer is None
