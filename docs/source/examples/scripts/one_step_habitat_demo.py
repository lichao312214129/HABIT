#!/usr/bin/env python
"""
One-step habitat analysis on a synthetic cohort.

Each subject is clustered independently at the voxel level (no supervoxel
stage, no cohort-level preprocessing chain). The fitted per-subject state is
frozen into :class:`~habit.contracts.HabitatModel` entries inside
``StudyResult.subject_models``; there is no single cohort-level
``habitat_model``.

This script accompanies ``docs/source/examples/one_step_habitat.rst``.

Run from the repository root::

    python docs/source/examples/scripts/one_step_habitat_demo.py
"""

from __future__ import annotations

from habit import HabitatSpec, Spec, make_synthetic_cohort
import habit.recipes as recipes

cohort = make_synthetic_cohort(n_subjects=4, shape=(20, 20, 20), rng=42)
print(f"Cohort: {len(cohort)} subjects")

# Keyword arguments follow the per-subject runtime order (not HabitatSpec field
# definition order). One-step has no cohort_feature_preprocessors stage:
#   voxel features -> voxel prep -> fit habitats inside this subject
#   -> assign -> habitat features.
spec = HabitatSpec(
    name="habitat_one_step",
    # 1. Per-voxel features inside each ROI.
    voxel_feature_extractor=Spec("raw", {"modalities": ["T1", "T2"]}),
    # 2. Stateless per-subject prep before clustering.
    voxel_feature_preprocessors=(
        Spec("winsorize", {"winsor_limits": (0.05, 0.05), "across_features": False}),
        Spec("minmax", {"across_features": False}),
    ),
    # 3. No supervoxel stage: voxels are clustered inside each subject.
    supervoxelizer=None,
    # 4. Fit a habitat definition on THIS subject's voxels only.
    habitat_model_fitter=Spec(
        "kmeans",
        {"min_habitats": 2, "max_habitats": 4, "validation": "silhouette", "n_init": 5},
    ),
    # 5. Assign labels from that subject-local definition.
    habitat_assigner=Spec("nearest_centroid"),
    # 6. Describe habitats after assignment.
    habitat_features=(
        Spec("volume"),
        Spec("msi"),
        Spec("ith_score"),
        Spec("non_radiomics"),
        # Heavy PyRadiomics families (opt-in; require pyradiomics):
        # Spec("traditional"),
        # Spec("whole_habitat"),
        # Spec("each_habitat"),
    ),
    random_seed=42,
)

result = recipes.one_step(cohort, spec)

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
