#!/usr/bin/env python
"""
Direct-pooling habitat analysis on demo_data.

Accompanies ``docs/source/examples/direct_pooling_habitat.rst``.
Run from the repository root::

    python docs/source/examples/scripts/direct_pooling_habitat_demo.py
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
print(f"Cohort: {len(cohort)} subjects")

spec = HabitatSpec(
    name="habitat_direct_pooling",
    stages=(
        Stage("extract_voxel_features", Spec("raw", {"modalities": list(MODALITIES)})),
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

result = recipes.Study(spec=spec).fit_predict(cohort)
print(result.habitat_model.summary())
print(f"Habitat maps: {len(result.habitat_maps)}")
out_dir = result.save("out/direct_pooling_demo")
print(f"Saved study to {out_dir}")
# END example

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _example_roi import save_habitat_study_figures
    from _habitat_eye_check import eye_check_study

    save_habitat_study_figures(cohort, result, prefix="direct_pooling")
    eye_check_study(cohort, result)
