"""
Extracting voxel intensities
============================

Input: one :class:`~habit.contracts.Subject`. Output: a
:class:`~habit.contracts.VoxelFeatureField`. Stage:
``extract_voxel_features`` with ``raw``.
"""

# %%
# Change ``DATA`` / ``MODALITIES`` / ``ROI`` to your preprocessed layout.
from pathlib import Path

import matplotlib.pyplot as plt

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.viz import plot_intensity_slice
from habit.voxel_features import RawVoxelFeatures

DATA = fetch_demo()
MODALITIES = ("LAP",)
ROI = "LAP"
subject = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[0]

field = RawVoxelFeatures(modalities=list(MODALITIES), roi=ROI)(subject)
print(field.feature_frame().head())
field.feature_frame().head()

fig = plot_intensity_slice(
    subject.image(ROI),
    roi_mask=subject.mask(ROI),
    title="LAP",
    roi_contour=True,
)
Path("out").mkdir(exist_ok=True)
fig.savefig("out/voxel_intensities.png", dpi=150, bbox_inches="tight")
plt.show()
