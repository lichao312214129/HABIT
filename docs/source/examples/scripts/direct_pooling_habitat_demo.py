#!/usr/bin/env python
"""
Direct-pooling habitat analysis on a synthetic cohort.

All ROI voxels from all subjects are pooled before cohort-level clustering.
Both ``voxel_feature_preprocessors`` (per subject) and
``cohort_feature_preprocessors`` (across the pooled table) apply during
training.

This script accompanies ``docs/source/examples/direct_pooling_habitat.rst``.

Run from the repository root::

    python docs/source/examples/scripts/direct_pooling_habitat_demo.py
"""

from __future__ import annotations

from habit import HabitatSpec, Spec, make_synthetic_cohort
import habit.recipes as recipes

cohort = make_synthetic_cohort(n_subjects=5, shape=(18, 18, 18), rng=42)
print(f"Cohort: {len(cohort)} subjects")

# Keyword arguments are ordered by the runtime pipeline (not HabitatSpec field
# definition order). Dataclass defaults put preprocessors after the fitter in
# the class body; keyword calls may follow execution order instead:
#   voxel features -> voxel prep -> (no supervoxels) -> cohort prep
#   -> fit habitats -> assign -> habitat features.
spec = HabitatSpec(
    name="habitat_direct_pooling",
    # 1. Per-voxel features inside each ROI.
    voxel_feature_extractor=Spec("raw", {"modalities": ["T1", "T2"]}),
    # 2. Stateless per-subject prep on voxel feature rows (before pooling).
    voxel_feature_preprocessors=(
        Spec("winsorize", {"winsor_limits": (0.05, 0.05), "across_features": False}),
        Spec("minmax", {"across_features": False}),
    ),
    # 3. No supervoxel stage: every ROI voxel is a clustering unit.
    supervoxelizer=None,
    # 4. Stateful prep fitted on pooled training units, then applied.
    cohort_feature_preprocessors=(
        Spec("binning", {"n_bins": 8, "bin_strategy": "uniform", "across_features": False}),
    ),
    # 5. Learn the cohort-level habitat definition on pooled units.
    habitat_model_fitter=Spec(
        "kmeans",
        {"min_habitats": 2, "max_habitats": 4, "validation": "silhouette", "n_init": 5},
    ),
    # 6. Paint each unit with the nearest habitat centroid.
    habitat_assigner=Spec("nearest_centroid"),
    # 7. Describe habitats after assignment.
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

result = recipes.direct_pooling(cohort, spec)

print("\n--- Cohort-level habitat model ---")
print(result.habitat_model.summary())
print(f"\nHabitat maps: {len(result.habitat_maps)}")
print(f"Clustering units (voxel rows): {sum(u.features.shape[0] for u in result.units)}")
print(f"Feature table: {result.features.frame.shape}")

# Persist via the public StudyResult.save API.
# Default map_format is "nrrd" (v0.1 layout). Use "nii.gz" / "nii" / "mha" / "mhd"
# when a different label-map container is needed.
out_dir = result.save("out/direct_pooling_demo", map_format="nii.gz")
print(f"\nSaved study to {out_dir}")
for path in sorted(out_dir.glob("*_habitats.nii.gz")):
    print(f"  {path.name}")

# Eye-check: open habitats on anatomy (napari). Set HABIT_NO_VIEW=1 to skip.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _habitat_eye_check import eye_check_study
eye_check_study(cohort, result)
