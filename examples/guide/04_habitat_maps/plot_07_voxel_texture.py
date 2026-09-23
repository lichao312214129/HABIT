"""
Extracting a voxel texture
==========================

Input: one :class:`~habit.contracts.Subject`. Output: a
:class:`~habit.contracts.VoxelFeatureField` of local entropy. Stage:
``extract_voxel_features`` with ``local_entropy``.

Kernel size, bins, and GPU paths are in
:doc:`/auto_examples/02_voxel/plot_03_voxel_texture`.
"""

# %%
# Change ``DATA`` / ``MODALITIES`` / ``ROI`` to your preprocessed layout.
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.viz import plot_voxel_texture_slice
from habit.voxel_features import LocalEntropyVoxelFeatures

DATA = fetch_demo()
MODALITIES = ("LAP",)
ROI = "LAP"
subject = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[0]
image = subject.image(ROI)

field = LocalEntropyVoxelFeatures(
    modalities=list(MODALITIES), roi=ROI, kernel_size=5, bins=32
)(subject)
print(field.feature_frame().head())
field.feature_frame().head()

entropy = np.zeros(field.geometry.shape, dtype=np.float64)
entropy[tuple(field.voxel_index.T)] = field.values[:, 0]
fig = plot_voxel_texture_slice(
    entropy, anatomy=image, roi_mask=subject.mask(ROI), axis=0, crop_to="roi"
)
Path("out").mkdir(exist_ok=True)
fig.savefig("out/voxel_texture_field.png", dpi=150, bbox_inches="tight")
plt.show()
