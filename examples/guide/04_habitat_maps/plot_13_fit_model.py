"""
Fitting a cohort habitat model
==============================

Input: one :class:`~habit.contracts.Supervoxelization` per subject.
Output: a :class:`~habit.contracts.HabitatModel`. Stage: ``fit``.
"""

# %%
# Change ``DATA`` / ``MODALITIES`` / ``ROI`` to your preprocessed layout.
from pathlib import Path

import matplotlib.pyplot as plt

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.habitat_model import KMeansHabitatModelFitter
from habit.supervoxel import SlicSupervoxelizer
from habit.voxel_features import RawVoxelFeatures

DATA = fetch_demo()
MODALITIES = ("LAP",)
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:2]

voxel = RawVoxelFeatures(modalities=list(MODALITIES), roi=ROI)
slic = SlicSupervoxelizer(n_supervoxels=8, compactness=10.0)
units = [slic(voxel(subject)) for subject in cohort]
fitter = KMeansHabitatModelFitter(n_habitats=3, n_init=3)
fitter.set_random_state(0)
model = fitter.fit(units, cohort=cohort)
print(model.summary())

fig, ax = plt.subplots()
image = ax.imshow(model.centroids, aspect="auto")
ax.set_xticks(range(len(model.feature_names)))
ax.set_xticklabels(list(model.feature_names))
ax.set_ylabel("habitat")
ax.set_title("habitat centroids")
fig.colorbar(image, ax=ax)
Path("out").mkdir(exist_ok=True)
fig.savefig("out/cohort_centroids.png", dpi=150, bbox_inches="tight")
plt.show()
