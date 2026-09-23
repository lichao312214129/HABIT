"""
Partitioning a ROI into supervoxels
===================================

Input: a :class:`~habit.contracts.VoxelFeatureField`. Output: a
:class:`~habit.contracts.Supervoxelization`. Stage: ``partition`` with
``slic``.
"""

# %%
# Change ``DATA`` / ``MODALITIES`` / ``ROI`` to your preprocessed layout.
# sphinx_gallery_thumbnail_number = 1
from pathlib import Path

import matplotlib.pyplot as plt

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.supervoxel import SlicSupervoxelizer
from habit.viz import plot_habitat_overlay
from habit.voxel_features import RawVoxelFeatures

DATA = fetch_demo()
MODALITIES = ("LAP",)
ROI = "LAP"
subject = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[0]

field = RawVoxelFeatures(modalities=list(MODALITIES), roi=ROI)(subject)
units = SlicSupervoxelizer(n_supervoxels=100, compactness=10.0)(field)
print(f"n_supervoxels={len(units.features)}")
print(units.features.head())

fig = plot_habitat_overlay(
    subject.image(ROI),
    units,
    title="SLIC supervoxels",
    crop_to="labels",
)
Path("out").mkdir(exist_ok=True)
fig.savefig("out/slic_supervoxels.png", dpi=150, bbox_inches="tight")
plt.show()
