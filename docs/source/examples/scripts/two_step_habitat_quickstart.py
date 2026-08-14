#!/usr/bin/env python
"""
Two-step habitat analysis, end to end, on demo_data (one subject is enough).

Accompanies ``docs/source/examples/two_step_habitat.rst``.
Run from the repository root::

    python docs/source/examples/scripts/two_step_habitat_quickstart.py
"""

from __future__ import annotations

# BEGIN example
from habit import HabitatSpec, Spec, Stage, cohort_from_directory
import habit.recipes as recipes

# Change DATA / MODALITIES / ROI to your preprocessed layout
DATA = "demo_data/preprocessed"
MODALITIES = ("LAP",)
ROI = "LAP"

cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:2]
print(f"Cohort: {len(cohort)} subjects -> {list(cohort.subject_ids)}")

# What can go in Spec("...")? See docs/source/how_to/habitat_components.rst
# or: list_plugins("voxel_feature_extractor") / get_param_schema("kmeans", "habitat_model_fitter")
spec = HabitatSpec(
    name="habitat_two_step",
    stages=(
        Stage("extract_voxel_features", Spec("raw", {"modalities": list(MODALITIES)})),
        Stage("partition", Spec("kmeans", {"n_supervoxels": 8, "n_init": 5})),
        Stage("pool", Spec("pool")),
        Stage(
            "fit",
            Spec(
                "kmeans",
                {
                    "min_habitats": 2,
                    "max_habitats": 10,
                    "validation": "elbow",
                    "n_init": 5,
                },
            ),
        ),
        Stage("assign", Spec("nearest_centroid")),
        Stage("quantify", Spec("volume")),
        Stage("quantify2", Spec("msi")),
        Stage("quantify3", Spec("ith_score")),
        Stage("quantify4", Spec("non_radiomics")),
    ),
    random_seed=42,
)
print(f"Spec fingerprint: {spec.fingerprint()}")

result = recipes.Study(spec=spec).fit_predict(cohort)
print(result.habitat_model.summary())
print(f"Habitat maps: {len(result.habitat_maps)}")
print(f"Feature columns (first 6): {list(result.features.feature_columns)[:6]}")
print(result.manifest.describe_methods())

# Persist wherever you like — swap this path for your project
out_dir = result.save("out/two_step_demo")
print(f"Saved study to {out_dir}")
# END example

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _example_roi import save_habitat_study_figures
    from _habitat_eye_check import eye_check_study

    save_habitat_study_figures(cohort, result, prefix="two_step")
    eye_check_study(cohort, result)
