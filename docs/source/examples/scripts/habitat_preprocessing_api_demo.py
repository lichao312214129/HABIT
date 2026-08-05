#!/usr/bin/env python
"""
Clustering feature-preprocessing chains (subject + cohort) via HabitatSpec.

This is NOT image preprocessing. Image intensity resampling lives in
``preprocess_subject`` / ``preprocess_images``. Here the rows are voxels or
supervoxels on the way to a habitat definition.

* **Subject chain** (stateless) — ``voxel_feature_preprocessors`` /
  ``supervoxel_feature_preprocessors``.
* **Cohort chain** (stateful; travels inside HabitatModel) —
  ``cohort_feature_preprocessors``.

Accompanies ``docs/source/examples/habitat_preprocessing_api.rst``.
"""

from __future__ import annotations

from habit import HabitatSpec, Spec, make_synthetic_cohort
from habit.domain.assembly import build_subject_chain
import habit.recipes as recipes

cohort = make_synthetic_cohort(n_subjects=3, shape=(14, 14, 14), rng=5)

# Keyword order follows the runtime pipeline (not HabitatSpec field order).
spec = HabitatSpec(
    name="feature_prep_demo",
    voxel_feature_extractor=Spec("raw", {"modalities": ["T1", "T2"]}),
    voxel_feature_preprocessors=(
        Spec("winsorize", {"winsor_limits": (0.05, 0.05), "across_features": False}),
        Spec("minmax", {"across_features": False}),
    ),
    supervoxelizer=Spec("kmeans", {"n_supervoxels": 6, "n_init": 3}),
    cohort_feature_preprocessors=(
        Spec("binning", {"n_bins": 6, "bin_strategy": "uniform", "across_features": False}),
    ),
    habitat_model_fitter=Spec(
        "kmeans",
        {"min_habitats": 2, "max_habitats": 3, "validation": "silhouette", "n_init": 3},
    ),
    habitat_assigner=Spec("nearest_centroid"),
    habitat_features=(Spec("volume"),),
    random_seed=5,
)

print("=== Spec declares both preprocessor levels ===")
print(f"  voxel steps:  {[s.name for s in spec.voxel_feature_preprocessors]}")
print(f"  cohort steps: {[s.name for s in spec.cohort_feature_preprocessors]}")

result = recipes.two_step(cohort, spec)
assert result.habitat_model is not None
print(f"=== two_step with feature chains: habitats={result.habitat_model.n_habitats} ===")

# Atomic subject-level chain (no cohort / recipe required).
chain = build_subject_chain(list(spec.voxel_feature_preprocessors))
assert chain is not None
units = result.units[0]
transformed = chain(units.features)
print("=== Atomic SubjectPreprocessingChain ===")
print(f"  in={units.features.shape}, out={transformed.shape}")
