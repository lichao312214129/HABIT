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
"""
What each habitat recipe refuses to do.

The three designs are distinguished by which optional pipeline stages they
use, so a spec meant for one of them is structurally accepted by the others
and would run to completion producing a *different analysis under the caller's
chosen name*. Nothing downstream could detect that, which is why the mismatch
is rejected at the entry point rather than tolerated.
"""

from __future__ import annotations

import pytest

from habit.exceptions import HABITAPIError
from habit.spec.specs import HabitatSpec, Spec


def _spec(**overrides: object) -> HabitatSpec:
    """
    Build a structurally valid spec; overrides select the design.

    Args:
        **overrides: Fields to replace, e.g. ``supervoxelizer=None``.

    Returns:
        The specification.
    """
    fields = {
        "name": "test",
        "voxel_feature_extractor": Spec(name="raw", params={"modalities": ["t1"]}),
        "supervoxelizer": Spec(name="kmeans", params={}),
        "habitat_model_fitter": Spec(name="kmeans", params={}),
        "habitat_assigner": Spec(name="nearest_centroid", params={}),
    }
    fields.update(overrides)
    return HabitatSpec(**fields)  # type: ignore[arg-type]


@pytest.mark.unit
def test_two_step_requires_a_supervoxelizer() -> None:
    """A spec with no supervoxel stage is not a two-step analysis."""
    import habit.recipes as recipes

    with pytest.raises(HABITAPIError, match="two_step requires a supervoxelizer"):
        recipes.two_step(object(), _spec(supervoxelizer=None))  # type: ignore[arg-type]


@pytest.mark.unit
def test_direct_pooling_rejects_a_supervoxelizer() -> None:
    """Direct pooling clusters voxels; a supervoxel stage contradicts it."""
    import habit.recipes as recipes

    with pytest.raises(HABITAPIError, match="direct_pooling clusters voxels"):
        recipes.direct_pooling(object(), _spec())  # type: ignore[arg-type]


@pytest.mark.unit
def test_one_step_rejects_a_supervoxelizer() -> None:
    """One-step clusters each subject's voxels; a supervoxel stage contradicts it."""
    import habit.recipes as recipes

    with pytest.raises(HABITAPIError, match="one_step clusters each subject"):
        recipes.one_step(object(), _spec())  # type: ignore[arg-type]


@pytest.mark.unit
def test_one_step_rejects_a_cohort_preprocessing_chain() -> None:
    """
    Cohort-level preprocessing has no meaning when nothing crosses subjects.

    v0.1 accepted the combination and silently ignored the chain; refusing it
    is the difference between a caller learning their configuration is
    meaningless and their believing it took effect.
    """
    import habit.recipes as recipes

    spec = _spec(
        supervoxelizer=None,
        cohort_feature_preprocessors=(Spec(name="zscore", params={}),),
    )
    with pytest.raises(HABITAPIError, match="cohort-level"):
        recipes.one_step(object(), spec)  # type: ignore[arg-type]


@pytest.mark.unit
def test_seed_argument_overrides_the_spec() -> None:
    """
    ``seed=`` reaches the components, since it changes the science.

    Args:
        None.
    """
    from habit.recipes.habitat import _effective_spec

    assert _effective_spec(_spec(random_seed=1), 7).random_seed == 7
    assert _effective_spec(_spec(random_seed=1), None).random_seed == 1


@pytest.mark.unit
def test_direct_pooling_summary_keeps_spec_random_seed_with_cohort_chain() -> None:
    """
    Cohort preprocessing must not erase ``HabitatSpec.random_seed`` from the model card.

    Regression for the path ``fitter.fit`` -> ``with_cohort_preprocessing`` that
    previously left ``summary()`` reporting ``random seed: None`` even when the
    spec set ``random_seed=42``.
    """
    from habit.datasets import make_synthetic_cohort
    import habit.recipes as recipes

    cohort = make_synthetic_cohort(
        n_subjects=3, modalities=("T1", "T2"), shape=(12, 12, 12), rng=0
    )
    spec = HabitatSpec(
        name="direct_pooling_seeded",
        voxel_feature_extractor=Spec(name="raw", params={"modalities": ["T1", "T2"]}),
        supervoxelizer=None,
        habitat_model_fitter=Spec(
            name="kmeans",
            params={
                "min_habitats": 2,
                "max_habitats": 3,
                "validation": "silhouette",
                "n_init": 3,
            },
        ),
        habitat_assigner=Spec(name="nearest_centroid", params={}),
        voxel_feature_preprocessors=(
            Spec(name="minmax", params={"across_features": False}),
        ),
        cohort_feature_preprocessors=(
            Spec(
                name="binning",
                params={"n_bins": 4, "bin_strategy": "uniform", "across_features": False},
            ),
        ),
        habitat_features=(Spec(name="volume", params={}),),
        random_seed=42,
    )

    result = recipes.direct_pooling(cohort, spec)
    model = result.habitat_model
    assert model is not None
    assert model.provenance.random_seed == 42
    assert "random seed        : 42" in model.summary()
    assert "cohort_feature_preprocessor" in model.preprocessing_state
    assert "cohort_preprocessing" in model.provenance.produced_by


@pytest.mark.unit
def test_public_recipe_surface() -> None:
    """The recipe layer exposes exactly the assembly functions plus the result."""
    import habit.recipes as recipes

    assert set(recipes.__all__) == {
        "two_step",
        "one_step",
        "direct_pooling",
        "apply_habitat_model",
        "extract_habitat_features",
        "traditional_radiomics",
        "compare_models",
        "pairwise_delong_test",
        "preprocess_image",
        "preprocess_images",
        "preprocess_subject",
        "icc_analysis",
        "test_retest_analysis",
        "sort_dicom",
        "run_from_yaml",
        "Study",
        "StudyResult",
        "train_model",
        "cross_validate",
        "predict_model",
        "ModelResult",
        "CVResult",
        "PredictionResult",
        "two_step_habitat",
        "one_step_habitat",
        "direct_pooling_habitat",
        "dice",
        "dicom_info",
        "merge_tables",
    }
