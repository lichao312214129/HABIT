"""
Preprocessing features before clustering
========================================

Input: a raw :class:`~habit.contracts.VoxelFeatureField`. Output: the
same rows after winsorize then min-max. Stage:
``voxel_feature_preprocessors``.
"""

# %%
# Change ``DATA`` / ``MODALITIES`` / ``ROI`` to your preprocessed layout.
# sphinx_gallery_thumbnail_number = 1
from pathlib import Path

import matplotlib.pyplot as plt

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.feature_preprocessing import MinMaxScaling, Winsorizing
from habit.voxel_features import RawVoxelFeatures

DATA = fetch_demo()
MODALITIES = ("LAP", "PVP")
ROI = "LAP"
subject = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[0]

raw = RawVoxelFeatures(modalities=list(MODALITIES), roi=ROI)(subject).feature_frame()
print("Before:")
print(raw.head())
winsor = Winsorizing(winsor_limits=(0.05, 0.05), across_features=False)
clipped = winsor.transform(raw, winsor.fit(raw))
scaler = MinMaxScaling(across_features=False)
scaled = scaler.transform(clipped, scaler.fit(clipped))
print("After winsorize + minmax:")
print(scaled.head())
scaled.head()

column = str(next(name for name in raw.columns if name not in ("z", "y", "x")))
fig, axes = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
axes[0].hist(raw[column].to_numpy(), bins=30)
axes[0].set_title(f"{column} before")
axes[1].hist(scaled[column].to_numpy(), bins=30)
axes[1].set_title(f"{column} after")
Path("out").mkdir(exist_ok=True)
fig.savefig("out/preprocess_hist.png", dpi=150, bbox_inches="tight")
plt.show()
