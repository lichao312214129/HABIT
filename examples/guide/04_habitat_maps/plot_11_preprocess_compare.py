"""
Comparing habitat maps with and without preprocessing
=====================================================

Input: the same cohort, the same two-step design, the same ``k`` and
seed. Output: two :class:`~habit.contracts.HabitatMap` overlays. The
only change is ``voxel_feature_preprocessors``.
"""

# %%
# Change ``DATA`` / ``MODALITIES`` / ``ROI`` to your preprocessed layout.
# sphinx_gallery_thumbnail_number = 3
from pathlib import Path

import matplotlib.pyplot as plt

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.recipes import Study
from habit.spec import HabitatSpec, Spec
from habit.viz import plot_habitat_overlay
import numpy as np

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
    crop_to="labels",
)
fig_prepped.savefig("out/habitats_with_preprocessing.png", dpi=150, bbox_inches="tight")
plt.show()

fraction_columns = [
    str(name) for name in plain.features.frame.columns if str(name).endswith("volume_fraction")
]
fractions_plain = plain.features.frame.iloc[0][fraction_columns].to_numpy(dtype=float)
fractions_prepped = prepped.features.frame.iloc[0][fraction_columns].to_numpy(dtype=float)
labels = [name.replace("_volume_fraction", "").replace("_", " ") for name in fraction_columns]
fig_bar, ax = plt.subplots(figsize=(6.2, 3.2))
x = np.arange(len(labels))
ax.bar(x - 0.18, fractions_plain, width=0.36, label="without", color="#4C78A8")
ax.bar(x + 0.18, fractions_prepped, width=0.36, label="with", color="#F58518")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("volume fraction")
ax.set_ylim(0, 1)
ax.set_title("habitat volume fractions")
ax.legend()
fig_bar.savefig("out/preprocess_volume_fractions.png", dpi=150, bbox_inches="tight")
plt.show()
