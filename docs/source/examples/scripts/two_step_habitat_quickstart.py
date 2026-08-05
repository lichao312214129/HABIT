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
#    with cohort_from_directory("processed_images", modalities=..., roi=...)
#    for real data; everything downstream is identical.
cohort = make_synthetic_cohort(
    n_subjects=6,
    modalities=("T1", "T2"),
    shape=(24, 24, 24),
    rng=42,
)
print(f"Cohort: {len(cohort)} subjects -> {list(cohort.subject_ids)}")

# 2. Spec: the whole analysis as one frozen, fingerprintable value object.
spec = HabitatSpec(
    name="habitat_two_step",
    # Voxel level: concatenate the raw intensities of both modalities.
    voxel_feature_extractor=Spec("raw", {"modalities": ["T1", "T2"]}),
    # Supervoxel level: per-subject k-means over the voxel features.
    supervoxelizer=Spec("kmeans", {"n_supervoxels": 8, "n_init": 5}),
    # Cohort level: k-means over pooled supervoxels; the habitat count is
    # selected in [2, 3] by the silhouette score.
    habitat_model_fitter=Spec(
        "kmeans",
        {"min_habitats": 2, "max_habitats": 3, "validation": "silhouette", "n_init": 5},
    ),
    # Assignment: each supervoxel takes the habitat of its nearest centroid.
    habitat_assigner=Spec("nearest_centroid"),
    # Habitat feature families, computed after assignment.
    habitat_features=(Spec("volume"), Spec("msi"), Spec("ith_score")),
    random_seed=42,
)
print(f"Spec fingerprint: {spec.fingerprint()}")

# 3. Train: fit the cohort-level habitat definition and label every subject.
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
