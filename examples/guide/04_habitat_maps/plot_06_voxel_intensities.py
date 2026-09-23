"""
Extracting voxel intensities
============================

Input: one :class:`~habit.contracts.Subject`. Output: a
:class:`~habit.contracts.VoxelFeatureField`. Stage:
``extract_voxel_features`` with ``raw``.
"""

# %%
# Load one subject
# ----------------
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
print(subject.subject_id, sorted(subject.images))

# %%
# Read one row per ROI voxel
# --------------------------
# ``RawVoxelFeatures`` returns a field with one column per phase, plus
# the grid position of every row so the field can be drawn back into
# the image.
field = RawVoxelFeatures(modalities=list(MODALITIES), roi=ROI)(subject)
print(field.feature_frame().head())
field.feature_frame().head()

# %%
# Arterial-phase slice
# --------------------
# The table above has one column per phase: pre_contrast, LAP, and PVP.
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
