#!/usr/bin/env python
"""
Two-step habitat analysis, end to end, on a synthetic cohort.

This script accompanies docs/source/examples/two_step_habitat.rst. It runs
anywhere (no files on disk, fixed seeds) and prints the objects a real study
inspects: the fitted HabitatModel, the per-subject habitat maps, and the
habitat feature table.

Run from the repository root:

    python docs/source/examples/scripts/two_step_habitat_quickstart.py
"""

from habit import HabitatSpec, Spec, make_synthetic_cohort
import habit.recipes as recipes

# 1. Cohort: six synthetic subjects, two modalities, one ROI each. Replace
#    with cohort_from_directory("demo_data/preprocessed", modalities=..., roi=...)
#    for real data; everything downstream is identical.
cohort = make_synthetic_cohort(
    n_subjects=6,
    modalities=("T1", "T2"),
    shape=(24, 24, 24),
    rng=42,
)
print(f"Cohort: {len(cohort)} subjects -> {list(cohort.subject_ids)}")

# 2. Spec: the whole analysis as one frozen, fingerprintable value object.
# Keyword arguments follow the runtime pipeline (not HabitatSpec field order):
#   voxel features -> supervoxels -> fit habitats on pooled units
#   -> assign -> habitat features.
spec = HabitatSpec(
    name="habitat_two_step",
    # 1. Voxel level: concatenate the raw intensities of both modalities.
    voxel_feature_extractor=Spec("raw", {"modalities": ["T1", "T2"]}),
    # 2. Supervoxel level: per-subject k-means over the voxel features.
    supervoxelizer=Spec("kmeans", {"n_supervoxels": 8, "n_init": 5}),
    # 3. Cohort level: k-means over pooled supervoxels; habitat count in
    #    [2, 3] selected by silhouette score.
    habitat_model_fitter=Spec(
        "kmeans",
        {"min_habitats": 2, "max_habitats": 3, "validation": "silhouette", "n_init": 5},
    ),
    # 4. Assignment: each supervoxel takes the habitat of its nearest centroid.
    habitat_assigner=Spec("nearest_centroid"),
    # 5. Habitat feature families, computed after assignment.
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
print(f"Spec fingerprint: {spec.fingerprint()}")

# 3. Train: fit the cohort-level habitat definition and label every subject.
#    Non-batch alternative after training:
#      habitat_map = result.pipeline(cohort[0])
#    Fit-time units (no assigner): build_habitat_components(spec).pipeline(
#        assigner=None).units(cohort[0])  — see habitat_preprocessing_demo.py.
result = recipes.two_step(cohort, spec)

print("\n--- Fitted habitat model ---")
print(result.habitat_model.summary())

print(f"\nHabitat maps: {len(result.habitat_maps)} "
      f"(one per subject, label ids 1..{result.habitat_model.n_habitats})")

print(f"Feature table: {result.features.frame.shape[0]} subjects x "
      f"{len(result.features.feature_columns)} features")
print("First feature columns:", list(result.features.feature_columns)[:6])

# 4. Methods paragraph, generated from the run manifest: only steps that
#    actually executed are stated.
print("\n--- Methods paragraph (from the run manifest) ---")
print(result.manifest.describe_methods())

# 5. Persist: writing to disk is a separate, explicit act in v1.0.
#    result.save("out/two_step_demo") writes <subject>_habitats.nrrd,
#    habitat_model.habitatmodel, habitat_features.csv and run_manifest.json.
print("\nTo persist everything: result.save('out/two_step_demo')")

# 6. Eye-check: open habitats on anatomy (napari). Set HABIT_NO_VIEW=1 to skip.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _habitat_eye_check import eye_check_study
eye_check_study(cohort, result)
