"""
Comparing habitat maps with and without preprocessing
=====================================================

Input: the same cohort, the same two-step design, the same ``k`` and
seed. Output: two :class:`~habit.contracts.HabitatMap` overlays. The
only change is ``voxel_feature_preprocessors``.
"""

# %%
# Change ``DATA`` / ``MODALITIES`` / ``ROI`` to your preprocessed layout.
from pathlib import Path

import matplotlib.pyplot as plt

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.recipes import Study
from habit.spec import HabitatSpec, Spec
from habit.viz import plot_habitat_overlay

DATA = fetch_demo()
MODALITIES = ("LAP",)
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:2]
Path("out").mkdir(exist_ok=True)


def two_step(preprocessors: tuple) -> HabitatSpec:
    """Build the same two-step spec with or without voxel preprocessing."""
    return HabitatSpec(
        name="two_step",
        voxel_feature_extractor=Spec("raw", {"modalities": list(MODALITIES), "roi": ROI}),
        voxel_feature_preprocessors=preprocessors,
        supervoxelizer=Spec("kmeans", {"n_supervoxels": 8, "n_init": 3}),
        habitat_model_fitter=Spec("kmeans", {"n_habitats": 3, "n_init": 3}),
        habitat_assigner=Spec("nearest_centroid"),
        random_seed=0,
        pooling="cohort",
    )


plain = Study(spec=two_step(())).fit_predict(cohort)
fig_plain = plot_habitat_overlay(
    cohort[0].image(ROI),
    plain.habitat_maps[0],
    title="habitats without preprocessing",
    axis=0,
    crop_to="labels",
)
fig_plain.savefig("out/habitats_without_preprocessing.png", dpi=150, bbox_inches="tight")
plt.show()

chain = (
    Spec("winsorize", {"winsor_limits": (0.05, 0.05), "across_features": False}),
    Spec("minmax", {"across_features": False}),
)
prepped = Study(spec=two_step(chain)).fit_predict(cohort)
fig_prepped = plot_habitat_overlay(
    cohort[0].image(ROI),
    prepped.habitat_maps[0],
    title="habitats with preprocessing",
    axis=0,
    crop_to="labels",
)
fig_prepped.savefig("out/habitats_with_preprocessing.png", dpi=150, bbox_inches="tight")
plt.show()
