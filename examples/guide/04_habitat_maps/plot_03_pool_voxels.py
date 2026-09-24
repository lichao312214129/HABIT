"""
Pooling voxels across the cohort
================================

Input: a cohort of at least two subjects. Output: one shared
:class:`~habit.contracts.HabitatModel` fitted on voxels, and one
:class:`~habit.contracts.HabitatMap` per subject. The stage list is
``pool`` then ``fit``, with no ``partition``. ``direct_pooling_habitat(...)``
is a shortcut that builds the same stage list.
"""

# %%
# Load the cohort
# ---------------
# Change ``DATA`` / ``MODALITIES`` / ``ROI`` to your preprocessed layout.
# sphinx_gallery_thumbnail_number = 1
from pathlib import Path

import matplotlib.pyplot as plt

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.recipes import Study
from habit.spec import HabitatSpec, Spec, Stage
from habit.viz import plot_habitat_overlay
import numpy as np

DATA = fetch_demo()
# Three DCE phases: unenhanced, arterial, and portal-venous.
MODALITIES = ("pre_contrast", "LAP", "PVP")
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:2]
print(f"Cohort: {list(cohort.subject_ids)}")

# %%
# Fit one model on every ROI voxel
# --------------------------------
# No ``partition``: each ROI voxel is its own clustering unit. ``pool``
# puts the voxels of every subject together and one model is fit on them.
spec = HabitatSpec(
    name="direct_pooling",
    stages=(
        Stage("extract", Spec("raw", {"modalities": list(MODALITIES), "roi": ROI})),
        Stage("pool", Spec("pool")),
        Stage("fit", Spec("kmeans", {"n_habitats": 3, "n_init": 10})),
        Stage("assign", Spec("nearest_centroid")),
        Stage("volume", Spec("volume")),
    ),
    random_seed=0,
)
result = Study(spec).fit_predict(cohort)
print(result.habitat_model.summary())
print(result.features.frame)
result.features.frame

# %%
# Intensities inside each habitat
# -------------------------------
# The histogram uses the displayed ROI image, not a feature column.
Path("out").mkdir(exist_ok=True)
fig_hist, ax = plt.subplots(figsize=(6.2, 3.2))
colors = ["#4C78A8", "#F58518", "#54A24B"]
for habitat_id in sorted(
    int(v) for v in np.unique(result.habitat_maps[0].label_array) if int(v) != 0
):
    # Histogram uses the displayed ROI image (LAP), not a feature column.
    values = cohort[0].image(ROI).data[
        result.habitat_maps[0].label_array == habitat_id
    ]
    ax.hist(values, bins=30, alpha=0.6, label=f"habitat {habitat_id}", color=colors[habitat_id - 1])
ax.set_xlabel(ROI)
ax.set_ylabel("voxels")
ax.set_title("pooled voxel intensities by habitat")
ax.legend()
fig_hist.savefig("out/pooling_intensity_hist.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# One overlay per subject
# -----------------------
for subject, habitat_map in zip(cohort, result.habitat_maps):
    fig = plot_habitat_overlay(
        subject.image(ROI),
        habitat_map,
        title=f"habitats ({habitat_map.subject_id})",
        crop_to="labels",
    )
    fig.savefig(
        f"out/pooling_{habitat_map.subject_id}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()
