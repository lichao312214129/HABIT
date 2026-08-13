#!/usr/bin/env python
"""
Train habitats on demo_data, then extract light habitat-map features.

Accompanies ``docs/source/examples/feature_extraction.rst``.
Run from the repository root::

    python docs/source/examples/scripts/feature_extraction_demo.py
"""

from __future__ import annotations

# BEGIN example
from pathlib import Path

from habit import HabitatSpec, Spec, Stage, cohort_from_directory
import habit.recipes as recipes

# Change DATA / MODALITIES / ROI to your preprocessed layout
DATA = "demo_data/preprocessed"
MODALITIES = ("LAP",)
ROI = "LAP"

cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:2]
spec = HabitatSpec(
    name="extract_demo",
    stages=(
        Stage("extract_voxel_features", Spec("raw", {"modalities": list(MODALITIES)})),
        Stage("partition", Spec("kmeans", {"n_supervoxels": 6, "n_init": 3})),
        Stage("pool", Spec("pool")),
        Stage(
            "fit",
            Spec(
                "kmeans",
                {
                    "min_habitats": 2,
                    "max_habitats": 4,
                    "validation": "elbow",
                    "n_init": 3,
                },
            ),
        ),
        Stage("assign", Spec("nearest_centroid")),
        Stage("quantify", Spec("volume")),
        Stage("quantify2", Spec("msi")),
        Stage("quantify3", Spec("ith_score")),
        Stage("quantify4", Spec("non_radiomics")),
    ),
    random_seed=11,
)

train_result = recipes.Study(spec=spec).fit_predict(cohort)
maps_dir = Path("out/habitat_maps")
train_result.save(maps_dir, write_maps=True, write_units_table=True)
print(f"Trained: {train_result.habitat_model.n_habitats} habitats -> {maps_dir}")

# Batch extract from the maps you just wrote (paths are plain strings)
feature_types = ["volume", "msi", "ith_score", "non_radiomics"]
extract_result = recipes.extract_habitat_features(
    {
        "raw_img_folder": DATA,
        "habitats_map_folder": str(maps_dir),
        "out_dir": "out/features",
        "n_processes": 1,
        "habitat_pattern": "*_habitats.nrrd",
        "feature_types": feature_types,
        "n_habitats": train_result.habitat_model.n_habitats,
    }
)
print(f"Extracted {feature_types} -> {extract_result.output_dir}")
# END example

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _example_roi import save_habitat_study_figures
    from _habitat_eye_check import eye_check_study

    save_habitat_study_figures(cohort, train_result, prefix="feature_extract")
    eye_check_study(cohort, train_result)
