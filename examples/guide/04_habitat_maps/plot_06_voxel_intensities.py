"""
Extracting voxel intensities
============================

Input: one :class:`~habit.contracts.Subject`. Output: a
:class:`~habit.contracts.VoxelFeatureField`. Stage:
``extract_voxel_features`` with ``raw``.
"""

# %%
# Change ``DATA`` / ``MODALITIES`` / ``ROI`` to your preprocessed layout.
# sphinx_gallery_thumbnail_number = 1
from pathlib import Path

import matplotlib.pyplot as plt

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.viz import plot_intensity_slice
from habit.voxel_features import RawVoxelFeatures

DATA = fetch_demo()
# Three DCE phases: unenhanced, arterial, and portal-venous.
MODALITIES = ("pre_contrast", "LAP", "PVP")
ROI = "LAP"
subject = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[0]

# RawVoxelFeatures reads the ROI voxels from each phase and returns a
# VoxelFeatureField: one row per voxel, one column per modality, plus
# the 3D grid position of every row so the field can be rendered back
# into image space.
field = RawVoxelFeatures(modalities=list(MODALITIES), roi=ROI)(subject)
print(field.feature_frame().head())
field.feature_frame().head()

# The slice shows the ROI anatomy (LAP). The table above has one
# column per phase: pre_contrast, LAP, and PVP.
fig = plot_intensity_slice(
    subject.image(ROI),
    roi_mask=subject.mask(ROI),
    title="voxel intensities",
    image_label="LAP",
    roi_contour=True,
    crop_to="roi",
)
Path("out").mkdir(exist_ok=True)
fig.savefig("out/voxel_intensities.png", dpi=150, bbox_inches="tight")
plt.show()
