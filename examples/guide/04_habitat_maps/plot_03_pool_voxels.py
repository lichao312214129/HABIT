"""
Pooling voxels across the cohort
================================

Input: a cohort of at least two subjects. Output: one shared
:class:`~habit.contracts.HabitatModel` fitted on voxels, and one
:class:`~habit.contracts.HabitatMap` per subject. Stage: ``pool`` then
``fit``. There is no ``partition`` stage.
"""

# %%
# Change ``DATA`` / ``MODALITIES`` / ``ROI`` to your preprocessed layout.
from pathlib import Path

import matplotlib.pyplot as plt

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.recipes import direct_pooling_habitat
from habit.viz import plot_habitat_overlay

DATA = fetch_demo()
MODALITIES = ("LAP",)
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:2]
print(f"Cohort: {list(cohort.subject_ids)}")

result = direct_pooling_habitat(
    modalities=MODALITIES,
    n_habitats=3,
    habitat_features=("volume",),
    random_seed=0,
    roi=ROI,
).fit_predict(cohort)
print(result.habitat_model.summary())
print(result.features.frame)
result.features.frame

fig = plot_habitat_overlay(
    cohort[0].image(ROI),
    result.habitat_maps[0],
    title="habitats (pooling)",
    axis=0,
    crop_to="labels",
)
Path("out").mkdir(exist_ok=True)
fig.savefig("out/pooling_overlay.png", dpi=150, bbox_inches="tight")
plt.show()
