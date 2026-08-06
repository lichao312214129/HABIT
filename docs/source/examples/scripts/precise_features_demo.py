"""Precise-feature screen demo: which voxel features deserve to define habitats.

Runs the Prior et al. (2024) precision screen on a synthetic cohort with
fast raw-intensity features (so the demo needs no PyRadiomics), then shows
the two artefacts of the workflow: the PreciseFeatureSet evidence table and
the feature whitelist that restricts a later habitat run to the precise set.
"""

import tempfile
from pathlib import Path

from habit import HabitatSpec, Spec, make_synthetic_cohort
from habit.domain import PreciseFeatureSet, RawVoxelFeatures
from habit.recipes import identify_precise_voxel_features

cohort = make_synthetic_cohort(n_subjects=3, modalities=("T1", "T2"), rng=42)


def raw_factory(kernel_radius: int, bin_width: float) -> RawVoxelFeatures:
    """Map the settings grid onto an extractor.

    Raw intensities have no kernel-radius or bin-width setting, so the grid
    point is ignored; with voxel radiomics this is where ``kernel_radius``
    and ``bin_width`` would be forwarded (the default factory does exactly
    that with the bundled CT preset).
    """
    return RawVoxelFeatures(modalities=["T1", "T2"])


print("=== precision screen (repeatability experiment) ===")
precise = identify_precise_voxel_features(
    cohort,
    extractor_factory=raw_factory,
    kernel_radii=(3,),  # one radius: skip the kernel-radius experiment
    bin_widths=(12,),  # one width: skip the bin-width experiment
    seed=7,
    show_progress=False,
)
print(f"  precise features: {list(precise.feature_names)}")
print(precise.to_frame().round(3).to_string(index=False))

print("\n=== whitelist bridge into a habitat spec ===")
whitelist = precise.preprocessor()
print(f"  whitelist spec: {whitelist.spec.to_dict()}")
spec = HabitatSpec(
    name="precise_demo",
    voxel_feature_extractor=Spec("raw", {"modalities": ["T1", "T2"]}),
    # The whitelist goes FIRST: only precise features reach scaling/clustering.
    voxel_feature_preprocessors=(
        whitelist.spec,
        Spec("minmax", {"across_features": False}),
    ),
    supervoxelizer=Spec("kmeans", {"n_supervoxels": 6, "n_init": 3}),
    habitat_model_fitter=Spec(
        "kmeans",
        {"min_habitats": 2, "max_habitats": 3, "validation": "silhouette", "n_init": 3},
    ),
    habitat_assigner=Spec("nearest_centroid"),
    random_seed=11,
)
print(f"  habitat spec fingerprint: {spec.fingerprint()}")

print("\n=== artefact round trip ===")
with tempfile.TemporaryDirectory(prefix="habit_precise_") as tmp:
    path = precise.save(Path(tmp) / "precise_features.json")
    reloaded = PreciseFeatureSet.load(path)
    print(f"  saved {path.name}; reloaded features: {list(reloaded.feature_names)}")
