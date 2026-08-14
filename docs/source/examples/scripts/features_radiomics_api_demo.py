#!/usr/bin/env python
"""
Feature extraction + traditional radiomics API (lightweight demo).

In-memory habitat studies already return feature tables when
``habitat_features`` is set on the spec. Directory-driven
``extract_habitat_features`` / ``traditional_radiomics`` are the CLI twins;
they are exercised end-to-end in ``demo_data/results/api/05_*`` and ``06_*``.

Accompanies ``docs/source/examples/features_radiomics_api.rst``.
"""

from __future__ import annotations

from pathlib import Path

from habit import HabitatSpec, Spec, Stage, make_synthetic_cohort
import habit.recipes as recipes

REPO_ROOT = Path(__file__).resolve().parents[4]
IMAGING = REPO_ROOT / "demo_data" / "preprocessed"
HABITAT_MAPS = REPO_ROOT / "demo_data" / "results" / "api" / "02_habitat_two_step"

# BEGIN example
print("=== In-memory: habitat_features on two_step ===")
cohort = make_synthetic_cohort(n_subjects=2, shape=(12, 12, 12), rng=3)
spec = HabitatSpec(
    name="feat_demo",
    stages=(
        Stage("extract_voxel_features", Spec("raw", {"modalities": ["T1", "T2"]})),
        Stage("partition", Spec("kmeans", {"n_supervoxels": 5, "n_init": 2})),
        Stage("pool", Spec("pool")),
        Stage(
            "fit",
            Spec(
                "kmeans",
                {
                    "min_habitats": 2,
                    "max_habitats": 3,
                    "validation": "elbow",
                    "n_init": 2,
                },
            ),
        ),
        Stage("assign", Spec("nearest_centroid")),
        Stage("quantify", Spec("volume")),
        Stage("quantify2", Spec("msi")),
        Stage("quantify3", Spec("ith_score")),
        Stage("quantify4", Spec("non_radiomics")),
    ),
    random_seed=3,
)
result = recipes.Study(spec=spec).fit_predict(cohort)
print(f"  feature columns ({len(result.features.feature_columns)}):")
for name in result.features.feature_columns:
    print(f"    - {name}")
# END example

# BEGIN figures
# Paste after the Script block. Uses cohort and result.
from habit.viz import plot_habitat_overlay

Path("out").mkdir(exist_ok=True)
fig = plot_habitat_overlay(
    cohort[0].image("T1"),
    result.habitat_maps[0],
    title="habitats",
)
fig.savefig("out/features_radiomics_api_overlay.png", dpi=150, bbox_inches="tight")
print("Wrote out/features_radiomics_api_overlay.png")
# END figures

# Eye-check: open habitats on anatomy (napari). Set HABIT_NO_VIEW=1 to skip.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _habitat_eye_check import eye_check_study
eye_check_study(cohort, result)

maps_ok = HABITAT_MAPS.is_dir() and any(HABITAT_MAPS.glob("*_habitats.nrrd"))
print("=== Directory recipes (call pattern; full run in coverage) ===")
print(
    f"  imaging present={IMAGING.is_dir()}, "
    f"habitat maps present={maps_ok}"
)
print(
    "  extract_habitat_features({raw_img_folder, habitats_map_folder, "
    "out_dir, feature_types:[volume, msi, ith_score, non_radiomics, "
    "# traditional, # whole_habitat, # each_habitat]})"
)
print(
    "  traditional_radiomics({paths.images_folder, paths.out_dir, "
    "processing.process_image_types})"
)
