#!/usr/bin/env python
"""
Direct-pooling habitat analysis on a synthetic cohort.

All ROI voxels from all subjects are pooled before cohort-level clustering.
Preprocess stages may run before and after the ``pool`` marker (post-pool
feature preprocess is first-class).

Primary API: HabitatSpec.stages + recipes.Study(...).fit_predict (pool only ⇒
direct_pooling).

This script accompanies ``docs/source/examples/direct_pooling_habitat.rst``.

Run from the repository root::

    python docs/source/examples/scripts/direct_pooling_habitat_demo.py
"""

from __future__ import annotations

from habit import HabitatSpec, Spec, Stage, make_synthetic_cohort
import habit.recipes as recipes

cohort = make_synthetic_cohort(n_subjects=5, shape=(18, 18, 18), rng=42)
print(f"Cohort: {len(cohort)} subjects")

# pool without partition ⇒ direct_pooling. preprocess1/2 run per subject
# before pool; preprocess3 runs on pooled units (cohort-level).
spec = HabitatSpec(
    name="habitat_direct_pooling",
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

result = recipes.Study(spec=spec).fit_predict(cohort)

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
