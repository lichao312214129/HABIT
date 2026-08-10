#!/usr/bin/env python
"""
One-step habitat analysis on a synthetic cohort.

Each subject is clustered independently at the voxel level (no partition
stage, no pool). The fitted per-subject state is frozen into
:class:`~habit.contracts.HabitatModel` entries inside
``StudyResult.subject_models``; there is no single cohort-level
``habitat_model``.

Primary API: HabitatSpec.stages + recipes.fit_habitat (neither partition
nor pool ⇒ one_step).

This script accompanies ``docs/source/examples/one_step_habitat.rst``.

Run from the repository root::

    python docs/source/examples/scripts/one_step_habitat_demo.py
"""

from __future__ import annotations

from habit import HabitatSpec, Spec, Stage, make_synthetic_cohort
import habit.recipes as recipes

cohort = make_synthetic_cohort(n_subjects=4, shape=(20, 20, 20), rng=42)
print(f"Cohort: {len(cohort)} subjects")

# Neither partition nor pool ⇒ one_step (inferred from the stage sequence).
spec = HabitatSpec(
    name="habitat_one_step",
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
        Stage(
            "fit",
            Spec(
                "kmeans",
                {
                    "min_habitats": 2,
                    "max_habitats": 4,
                    "validation": "silhouette",
                    "n_init": 5,
                },
            ),
        ),
        Stage("assign", Spec("nearest_centroid")),
        Stage("quantify", Spec("volume")),
        Stage("quantify2", Spec("msi")),
        Stage("quantify3", Spec("ith_score")),
        Stage("quantify4", Spec("non_radiomics")),
        # Heavy PyRadiomics families (opt-in; require pyradiomics):
        # Stage("quantify5", Spec("traditional")),
        # Stage("quantify6", Spec("whole_habitat")),
        # Stage("quantify7", Spec("each_habitat")),
    ),
    random_seed=42,
)

result = recipes.fit_habitat(cohort, spec)

print(f"\nCohort-level habitat_model: {result.habitat_model}")
print(f"Per-subject models: {len(result.subject_models)} subjects")
for subject_id, model in sorted(result.subject_models.items()):
    print(f"  {subject_id}: {model.n_habitats} habitats, id={model.model_id}")

print(f"\nHabitat maps: {len(result.habitat_maps)}")
print(f"Feature table: {result.features.frame.shape[0]} rows x "
      f"{len(result.features.feature_columns)} columns")

# Eye-check: open habitats on anatomy (napari). Set HABIT_NO_VIEW=1 to skip.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _habitat_eye_check import eye_check_study
eye_check_study(cohort, result)
