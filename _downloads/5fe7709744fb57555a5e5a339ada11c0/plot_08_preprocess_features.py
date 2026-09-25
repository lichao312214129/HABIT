"""
Preprocessing features before clustering
========================================

**Background.** Raw intensities differ in range between phases and
contain a few extreme voxels. The ``preprocess`` stage rescales the
feature columns before clustering so that no single column dominates
the distance.

**Purpose.** You see the feature table and the histogram of one column
before and after winsorize + min-max, then the habitat map built from
the preprocessed features.

**Key terms.**

* **preprocess** stage -- rescales / transforms feature columns (e.g.
  z-score, winsorize) so no column dominates the distance used by
  clustering. The three chains are compared on
  :doc:`/auto_examples/02b_stages/plot_09_feature_preprocessing`.
* **winsorize** -- clips each column at its 5th and 95th percentile here.
* **min-max** -- rescales each column to [0, 1].

Input: a raw :class:`~habit.contracts.VoxelFeatureField`. Output: a
:class:`~habit.contracts.HabitatMap` built from the preprocessed features.
Stage: ``voxel_feature_preprocessors``, then ``partition``, ``fit``,
``assign``.
"""

# %%
# Change ``DATA`` / ``MODALITIES`` / ``ROI`` to your preprocessed layout.
# sphinx_gallery_thumbnail_number = 2
from pathlib import Path

import matplotlib.pyplot as plt

from habit.contracts import Cohort, cohort_from_directory
from habit.datasets import fetch_demo
from habit.feature_preprocessing import CohortPreprocessingChain, MinMaxScaling, Winsorizing
from habit.habitat_model import KMeansHabitatModelFitter
from habit.supervoxel import SlicSupervoxelizer
from habit.viz import plot_habitat_overlay
from habit.voxel_features import RawVoxelFeatures

DATA = fetch_demo()
MODALITIES = ("pre_contrast", "LAP", "PVP")
ROI = "LAP"
subject = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[0]

# Raw voxel intensities: one row per ROI voxel, one column per phase.
raw_field = RawVoxelFeatures(modalities=list(MODALITIES), roi=ROI)(subject)
raw = raw_field.feature_frame()
print("Before:")
print(raw.head())

# CohortPreprocessingChain is sklearn-style: fit_transform learns the
# state and returns the transformed matrix in one call.
chain = CohortPreprocessingChain([
    Winsorizing(winsor_limits=(0.05, 0.05), across_features=False),
    MinMaxScaling(across_features=False),
])
scaled = chain.fit_transform(raw)
print("After winsorize + minmax:")
print(scaled.head())

# %%
# Histogram of the first feature before and after preprocessing.
column = str(raw.columns[0])
fig, axes = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
axes[0].hist(raw[column].to_numpy(), bins=30)
axes[0].set_title(f"{column} before")
axes[1].hist(scaled[column].to_numpy(), bins=30)
axes[1].set_title(f"{column} after")
Path("out").mkdir(exist_ok=True)
fig.savefig("out/preprocess_hist.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Continue: put the scaled features back into a VoxelFeatureField,
# partition into supervoxels, fit a habitat model, and assign labels.
scaled_field = raw_field.with_feature_frame(
    scaled,
    produced_by="cohort_feature_preprocessor",
    spec_fingerprint=chain.spec.fingerprint,
)
# Supervoxels are built on the scaled columns, so the preprocessing also
# shapes the patches, not only the final habitats.
units = SlicSupervoxelizer(n_supervoxels=100, compactness=10.0)(scaled_field)
fitter = KMeansHabitatModelFitter(n_habitats=3, n_init=3)
fitter.set_random_state(0)
model = fitter.fit([units], cohort=Cohort([subject], name="one"))
habitat_map = model.assigner()(units)
print(model.summary())

fig_map = plot_habitat_overlay(
    subject.image(ROI),
    habitat_map,
    title="habitats (preprocessed)",
    crop_to="labels",
)
fig_map.savefig("out/preprocessed_habitats.png", dpi=150, bbox_inches="tight")
plt.show()
