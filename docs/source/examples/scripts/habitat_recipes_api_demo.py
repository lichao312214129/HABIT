#!/usr/bin/env python
"""
Habitat recipe API: stages-first fit_habitat + apply + StudyResult / model I/O.

* **Primary** — ``HabitatSpec.stages`` + ``recipes.fit_habitat`` (strategy
  inferred: partition+pool → two_step; pool only → direct_pooling;
  neither → one_step).
* **Compat** — thin aliases ``two_step`` / ``one_step`` / ``direct_pooling``
  and named-field sugar still call the same executor.
* **Atomic** — ``result.pipeline(subject)`` and ``apply_habitat_model``.
* **Persistence** — ``StudyResult.save`` and ``HabitatModel.save/load``.

Accompanies ``docs/source/examples/habitat_recipes_api.rst``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from habit import HabitatModel, HabitatSpec, Spec, Stage, make_synthetic_cohort
import habit.recipes as recipes


def _quantify_stages() -> tuple[Stage, ...]:
    """Shared habitat-feature stages for the demos."""
    return (
        Stage("quantify", Spec("volume")),
        Stage("quantify2", Spec("msi")),
        Stage("quantify3", Spec("ith_score")),
        Stage("quantify4", Spec("non_radiomics")),
        # Heavy PyRadiomics families (opt-in; require pyradiomics):
        # Stage("quantify5", Spec("traditional")),
        # Stage("quantify6", Spec("whole_habitat")),
        # Stage("quantify7", Spec("each_habitat")),
    )


def _fit_stage() -> Stage:
    """Shared k-means habitat fitter stage."""
    return Stage(
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
    )


cohort = make_synthetic_cohort(n_subjects=3, shape=(16, 16, 16), rng=11)

print("=== fit_habitat two_step shape (partition + pool) ===")
two_step_spec = HabitatSpec(
    name="two_step",
    stages=(
        Stage("extract_voxel_features", Spec("raw", {"modalities": ["T1", "T2"]})),
        Stage(
            "preprocess1",
            Spec(
                "winsorize",
                {"winsor_limits": (0.05, 0.05), "across_features": False},
            ),
        ),
        Stage("preprocess2", Spec("minmax", {"across_features": False})),
        Stage("partition", Spec("kmeans", {"n_supervoxels": 8, "n_init": 3})),
        Stage("pool", Spec("pool")),
        Stage(
            "preprocess3",
            Spec(
                "binning",
                {
                    "n_bins": 8,
                    "bin_strategy": "uniform",
                    "across_features": False,
                },
            ),
        ),
        _fit_stage(),
        Stage("assign", Spec("nearest_centroid")),
        *_quantify_stages(),
    ),
    random_seed=11,
)
two = recipes.fit_habitat(cohort, two_step_spec)
assert two.habitat_model is not None
print(f"  habitats={two.habitat_model.n_habitats}, "
      f"features={len(two.features.feature_columns)}")

print("=== apply_habitat_model + .habitatmodel round-trip ===")
with tempfile.TemporaryDirectory(prefix="habit_api_habitat_") as tmp:
    archive = Path(tmp) / "demo.habitatmodel"
    two.habitat_model.save(archive)
    reloaded: HabitatModel = HabitatModel.load(archive)
    predicted = recipes.apply_habitat_model(cohort, two_step_spec, reloaded)
    print(f"  reloaded model_id={reloaded.model_id}")
    print(f"  predict maps={len(predicted.habitat_maps)}")

    saved = two.save(Path(tmp) / "study_out", write_cluster_plots=False)
    print(f"  StudyResult.save -> {saved.name}")

print("=== Atomic: SubjectPipeline on one subject ===")
pipeline = two.pipeline
assert pipeline is not None
one_map = pipeline(cohort[0])
print(f"  pipeline({cohort[0].subject_id}) labels="
      f"{sorted(set(int(v) for v in one_map.label_array.ravel() if v > 0))}")

print("=== fit_habitat one_step shape (neither partition nor pool) ===")
one_step_spec = HabitatSpec(
    name="one_step",
    stages=(
        Stage("extract_voxel_features", Spec("raw", {"modalities": ["T1", "T2"]})),
        Stage(
            "preprocess1",
            Spec(
                "winsorize",
                {"winsor_limits": (0.05, 0.05), "across_features": False},
            ),
        ),
        Stage("preprocess2", Spec("minmax", {"across_features": False})),
        _fit_stage(),
        Stage("assign", Spec("nearest_centroid")),
        *_quantify_stages(),
    ),
    random_seed=11,
)
one = recipes.fit_habitat(cohort, one_step_spec)
print(f"  subject_models={list(one.subject_models)}")

print("=== fit_habitat direct_pooling shape (pool only) ===")
direct_spec = HabitatSpec(
    name="direct_pooling",
    stages=(
        Stage("extract_voxel_features", Spec("raw", {"modalities": ["T1", "T2"]})),
        Stage(
            "preprocess1",
            Spec(
                "winsorize",
                {"winsor_limits": (0.05, 0.05), "across_features": False},
            ),
        ),
        Stage("preprocess2", Spec("minmax", {"across_features": False})),
        Stage("pool", Spec("pool")),
        Stage(
            "preprocess3",
            Spec(
                "binning",
                {
                    "n_bins": 8,
                    "bin_strategy": "uniform",
                    "across_features": False,
                },
            ),
        ),
        _fit_stage(),
        Stage("assign", Spec("nearest_centroid")),
        *_quantify_stages(),
    ),
    random_seed=11,
)
pool = recipes.fit_habitat(cohort, direct_spec)
assert pool.habitat_model is not None
print(f"  habitats={pool.habitat_model.n_habitats}")

print("=== Compat aliases (thin validators → same fit_habitat) ===")
# Named-field sugar + mode-named aliases remain supported.
sugar = HabitatSpec(
    name="two_step_sugar",
    voxel_feature_extractor=Spec("raw", {"modalities": ["T1", "T2"]}),
    supervoxelizer=Spec("kmeans", {"n_supervoxels": 8, "n_init": 3}),
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
    habitat_features=(Spec("volume"),),
    random_seed=11,
)
alias = recipes.two_step(cohort, sugar)
print(f"  recipes.two_step sugar: habitats={alias.habitat_model.n_habitats}")

# Eye-check: open habitats on anatomy (napari). Set HABIT_NO_VIEW=1 to skip.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _habitat_eye_check import eye_check_study
eye_check_study(cohort, two)
