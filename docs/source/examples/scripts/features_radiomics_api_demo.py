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

from habit import HabitatSpec, Spec, make_synthetic_cohort
import habit.recipes as recipes

REPO_ROOT = Path(__file__).resolve().parents[4]
IMAGING = REPO_ROOT / "demo_data" / "preprocessed" / "processed_images"
HABITAT_MAPS = REPO_ROOT / "demo_data" / "results" / "api" / "02_habitat_two_step"

print("=== In-memory: habitat_features on two_step ===")
cohort = make_synthetic_cohort(n_subjects=2, shape=(12, 12, 12), rng=3)
spec = HabitatSpec(
    name="feat_demo",
    voxel_feature_extractor=Spec("raw", {"modalities": ["T1", "T2"]}),
    supervoxelizer=Spec("kmeans", {"n_supervoxels": 5, "n_init": 2}),
    habitat_model_fitter=Spec(
        "kmeans",
        {"min_habitats": 2, "max_habitats": 3, "validation": "silhouette", "n_init": 2},
    ),
    habitat_assigner=Spec("nearest_centroid"),
    habitat_features=(Spec("volume"), Spec("msi"), Spec("ith_score")),
    random_seed=3,
)
result = recipes.two_step(cohort, spec)
print(f"  feature columns ({len(result.features.feature_columns)}):")
for name in result.features.feature_columns:
    print(f"    - {name}")

maps_ok = HABITAT_MAPS.is_dir() and any(HABITAT_MAPS.glob("*_habitats.nrrd"))
print("=== Directory recipes (call pattern; full run in coverage) ===")
print(
    f"  imaging present={IMAGING.is_dir()}, "
    f"habitat maps present={maps_ok}"
)
print(
    "  extract_habitat_features({raw_img_folder, habitats_map_folder, "
    "out_dir, feature_types:[traditional, msi, ith_score, ...]})"
)
print(
    "  traditional_radiomics({paths.images_folder, paths.out_dir, "
    "processing.process_image_types})"
)
