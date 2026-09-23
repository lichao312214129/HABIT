"""
Fitting a cohort habitat model
==============================

Input: one :class:`~habit.contracts.Supervoxelization` per subject.
Output: a :class:`~habit.contracts.HabitatModel`. Stage: ``fit``.
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
from habit.habitat_model import KMeansHabitatModelFitter
from habit.supervoxel import SlicSupervoxelizer
from habit.voxel_features import RawVoxelFeatures

DATA = fetch_demo()
# Three DCE phases: unenhanced, arterial, and portal-venous.
MODALITIES = ("pre_contrast", "LAP", "PVP")
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:2]
print(f"Cohort: {list(cohort.subject_ids)}")

# %%
# Partition each subject
# ----------------------
# Each element of ``units`` is one subject's supervoxels. These rows
# are what the cohort model fits on.
voxel = RawVoxelFeatures(modalities=list(MODALITIES), roi=ROI)
slic = SlicSupervoxelizer(n_supervoxels=100, compactness=10.0)
units = [slic(voxel(subject)) for subject in cohort]
for one in units:
    print(f"{one.subject_id}: {len(one.features)} supervoxels")

# %%
# Fit one shared model
# --------------------
# The fitter pools every subject's units and learns one set of centroids.
fitter = KMeansHabitatModelFitter(n_habitats=3, n_init=3)
fitter.set_random_state(0)
model = fitter.fit(units, cohort=cohort)
print(model.summary())

# %%
# First feature of each centroid
# ------------------------------
fig, ax = plt.subplots(figsize=(6.2, 3.2))
names = [f"habitat {index + 1}" for index in range(int(model.n_habitats))]
ax.bar(names, model.centroids[:, 0], color=["#4C78A8", "#F58518", "#54A24B"])
ax.set_ylabel(str(model.feature_names[0]))
ax.set_title("habitat centroids")
Path("out").mkdir(exist_ok=True)
fig.savefig("out/cohort_centroids.png", dpi=150, bbox_inches="tight")
plt.show()
