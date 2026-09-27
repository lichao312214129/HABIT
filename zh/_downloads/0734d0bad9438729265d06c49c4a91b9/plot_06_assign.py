"""
Assigning habitat labels
========================

**Background.** A fitted model only stores centroids. ``assign`` turns
that into a picture: every supervoxel of a subject gets the id of its
nearest centroid, and every voxel inherits the id of its supervoxel.

**Purpose.** You get a :class:`~habit.contracts.HabitatMap` for one
subject, an overlay of the habitats on the arterial image, and the
number of voxels in each habitat.

**Key terms.**

* **assign** -- gives every supervoxel / voxel the id of its nearest
  centroid, producing the habitat map.
* **habitat** -- a sub-region inside the tumour (the ROI) whose voxels
  behave alike across the input images; HABIT paints each ROI voxel with
  a habitat id (1, 2, 3, ...).
* **fit** / **centroid** -- see
  :doc:`/auto_examples/07_advanced/plot_05_fit`.

Input: a fitted :class:`~habit.contracts.HabitatModel` and one
:class:`~habit.contracts.Supervoxelization`. Output: a
:class:`~habit.contracts.HabitatMap`. Stage: ``assign`` with
``nearest_centroid``.
"""

# %%
# Load the cohort
# ---------------
# Change ``DATA`` / ``MODALITIES`` / ``ROI`` to your preprocessed layout.
# sphinx_gallery_thumbnail_number = 2
from pathlib import Path

import matplotlib.pyplot as plt

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.habitat_model import KMeansHabitatModelFitter
from habit.supervoxel import SlicSupervoxelizer
from habit.viz import plot_habitat_overlay
import numpy as np
from habit.voxel_features import RawVoxelFeatures

DATA = fetch_demo()
# Three DCE phases: unenhanced, arterial, and portal-venous.
MODALITIES = ("pre_contrast", "LAP", "PVP")
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:2]
subject = cohort[0]
print(f"Cohort: {list(cohort.subject_ids)}")
print("assigning:", subject.subject_id)

# %%
# Partition each subject
# ----------------------
voxel = RawVoxelFeatures(modalities=list(MODALITIES), roi=ROI)
slic = SlicSupervoxelizer(n_supervoxels=100, compactness=10.0)
units = [slic(voxel(item)) for item in cohort]
for one in units:
    print(f"{one.subject_id}: {len(one.features)} supervoxels")

# %%
# Fit one shared model
# --------------------
# Same fit as the previous page: three habitats, fixed seed, both subjects.
fitter = KMeansHabitatModelFitter(n_habitats=3, n_init=3)
fitter.set_random_state(0)
model = fitter.fit(units, cohort=cohort)
print(model.summary())

# %%
# Assign the first subject
# ------------------------
# ``assigner`` maps one subject's units to a habitat map by the nearest
# centroid. The second subject's units were used only to fit.
habitat_map = model.assigner()(units[0])
print(habitat_map.subject_id)
# Count voxels per habitat id (0 is background outside the ROI).
counts = [
    int(np.sum(habitat_map.label_array == habitat_id))
    for habitat_id in sorted(
        int(v) for v in np.unique(habitat_map.label_array) if int(v) != 0
    )
]
print("voxels per habitat:", counts)

# %%
# Overlay
# -------
fig = plot_habitat_overlay(
    subject.image(ROI),
    habitat_map,
    title="assigned habitats",
    crop_to="labels",
)
Path("out").mkdir(exist_ok=True)
fig.savefig("out/assigned_habitats.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Voxel counts
# ------------
labels = [f"habitat {index + 1}" for index in range(len(counts))]
fig_bar, ax = plt.subplots(figsize=(6.2, 3.2))
ax.bar(labels, counts, color=["#4C78A8", "#F58518", "#54A24B"])
ax.set_ylabel("voxels")
ax.set_title("assigned habitat voxel counts")
fig_bar.savefig("out/assigned_voxel_counts.png", dpi=150, bbox_inches="tight")
plt.show()
