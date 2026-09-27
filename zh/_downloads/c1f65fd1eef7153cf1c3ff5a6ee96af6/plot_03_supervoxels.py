"""
Partitioning a ROI into supervoxels
===================================

**Background.** A tumour has tens of thousands of voxels. The two-step
design first groups them, inside each subject, into small patches of
similar voxels, so the cohort model later clusters a few rows per
subject instead of every voxel.

**Purpose.** You get about 100 supervoxels for one subject, a table with
one mean feature vector per supervoxel, and an overlay showing the
patches on the arterial image.

**Key terms.**

* **supervoxel** -- a small patch of neighbouring voxels with similar
  features, clustered inside one subject first (``partition``), so the
  cohort model clusters tens of rows per subject instead of every voxel,
  which is faster and less noisy.
* **partition** stage -- the stage that builds supervoxels; ``slic`` and
  ``kmeans`` are two registered partitioners.
* **SLIC** (simple linear iterative clustering) -- grows supervoxels from
  a regular grid of seeds; ``compactness`` trades feature similarity
  against spatial closeness (higher gives rounder, more regular patches).

Input: a :class:`~habit.contracts.VoxelFeatureField`. Output: a
:class:`~habit.contracts.Supervoxelization`. Stage: ``partition`` with
``slic``.
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
from habit.supervoxel import SlicSupervoxelizer
from habit.viz import plot_habitat_overlay
from habit.voxel_features import RawVoxelFeatures

DATA = fetch_demo()
# Three DCE phases: unenhanced, arterial, and portal-venous.
MODALITIES = ("pre_contrast", "LAP", "PVP")
ROI = "LAP"
subject = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[0]
print(subject.subject_id)

# %%
# Read intensities, then partition
# --------------------------------
# SLIC groups nearby voxels with similar intensity. Each row of
# ``units.features`` is the mean vector of one supervoxel.
field = RawVoxelFeatures(modalities=list(MODALITIES), roi=ROI)(subject)
print(f"{field.values.shape[0]} voxels, columns: {list(field.feature_names)}")
# n_supervoxels is a target, not an exact count (SLIC may return a few
# more or fewer patches); compactness=10.0 is the default value.
units = SlicSupervoxelizer(n_supervoxels=100, compactness=10.0)(field)
print(f"n_supervoxels={len(units.features)}")
print(units.features.head())

# %%
# Supervoxel overlay
# ------------------
fig = plot_habitat_overlay(
    subject.image(ROI),
    units,
    title="SLIC supervoxels",
    crop_to="labels",
)
Path("out").mkdir(exist_ok=True)
fig.savefig("out/slic_supervoxels.png", dpi=150, bbox_inches="tight")
plt.show()
