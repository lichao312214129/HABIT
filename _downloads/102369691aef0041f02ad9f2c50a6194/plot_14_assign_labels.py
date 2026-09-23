"""
Assigning habitat labels
========================

Input: a fitted :class:`~habit.contracts.HabitatModel` and one
:class:`~habit.contracts.Supervoxelization`. Output: a
:class:`~habit.contracts.HabitatMap`. Stage: ``assign`` with
``nearest_centroid``.
"""

# %%
# Change ``DATA`` / ``MODALITIES`` / ``ROI`` to your preprocessed layout.
from pathlib import Path

import matplotlib.pyplot as plt

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.habitat_model import KMeansHabitatModelFitter
from habit.supervoxel import SlicSupervoxelizer
from habit.viz import plot_habitat_overlay
from habit.voxel_features import RawVoxelFeatures

DATA = fetch_demo()
MODALITIES = ("LAP",)
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:2]
subject = cohort[0]

voxel = RawVoxelFeatures(modalities=list(MODALITIES), roi=ROI)
slic = SlicSupervoxelizer(n_supervoxels=8, compactness=10.0)
units = [slic(voxel(item)) for item in cohort]
fitter = KMeansHabitatModelFitter(n_habitats=3, n_init=3)
fitter.set_random_state(0)
model = fitter.fit(units, cohort=cohort)

habitat_map = model.assigner()(units[0])
print(habitat_map.subject_id)

fig = plot_habitat_overlay(
    subject.image(ROI),
    habitat_map,
    title="assigned habitats",
    axis=0,
    crop_to="labels",
)
Path("out").mkdir(exist_ok=True)
fig.savefig("out/assigned_habitats.png", dpi=150, bbox_inches="tight")
plt.show()
