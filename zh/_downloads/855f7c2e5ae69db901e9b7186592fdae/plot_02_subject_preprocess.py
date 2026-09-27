"""
Subject-level preprocessing
===========================

**Background.** Raw intensities differ in range between phases and
contain a few extreme voxels. Before clustering, each patient's feature
columns are winsorized (1%) and z-scored with **that patient's** own
statistics so a darker scan does not collapse into one habitat.

**Purpose.** You see one column's histogram before and after subject-level
winsorize + z-score, then a small habitat map built from the scaled field.

**When to use.** Always, for MR intensity habitats. New patients recompute
their own mean/std; training statistics are not applied to them.

**Key terms.**

* **winsorize** -- clip each column at the 1st / 99th percentile.
* **z-score** -- subtract the patient's mean and divide by the patient's std.
"""

# %%
# Load one subject and extract raw intensities
# --------------------------------------------
# sphinx_gallery_thumbnail_number = 1
from pathlib import Path

import matplotlib.pyplot as plt

from habit.contracts import Cohort, cohort_from_directory
from habit.datasets import fetch_demo
from habit.feature_preprocessing import (
    SubjectPreprocessingChain,
    Winsorizing,
    ZScoreScaling,
)
from habit.habitat_model import KMeansHabitatModelFitter
from habit.pipeline import voxel_units
from habit.viz import plot_habitat_overlay
from habit.voxel_features import RawVoxelFeatures

DATA = fetch_demo()
MODALITIES = ("pre_contrast", "LAP", "PVP", "delay_3min")
ROI = "LAP"
subject = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[0]
Path("out").mkdir(exist_ok=True)

raw_field = RawVoxelFeatures(modalities=list(MODALITIES), roi=ROI)(subject)
raw = raw_field.feature_frame()
print("Before:")
print(raw.head())

# Subject-level only: no fitted state travels to another patient.
chain = SubjectPreprocessingChain(
    [
        Winsorizing(winsor_limits=(0.01, 0.01), across_features=False),
        ZScoreScaling(across_features=False),
    ]
)
scaled = chain(raw)
print("After winsorize 1% + z-score:")
print(scaled.head())

# %%
# Histogram of the first feature before and after
# -----------------------------------------------
column = str(raw.columns[0])
fig, axes = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
axes[0].hist(raw[column].to_numpy(), bins=30)
axes[0].set_title(f"{column} before")
axes[1].hist(scaled[column].to_numpy(), bins=30)
axes[1].set_title(f"{column} after")
fig.savefig("out/subject_preprocess_hist.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Habitats on the scaled field (one subject)
# ------------------------------------------
scaled_field = raw_field.with_feature_frame(
    scaled,
    produced_by="feature_preprocessing.subject.voxel",
    spec_fingerprint=chain.spec.fingerprint(),
)
units = voxel_units(scaled_field)
fitter = KMeansHabitatModelFitter(n_habitats=3, n_init=3)
fitter.set_random_state(0)
model = fitter.fit([units], cohort=Cohort([subject], name="one"))
habitat_map = model.assigner()(units)
print(model.summary())

fig = plot_habitat_overlay(
    subject.image("LAP"),
    habitat_map,
    title=f"{subject.subject_id}: habitats after subject z-score",
    crop_to="labels",
)
fig.savefig("out/subject_preprocess_habitats.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# How to read the result
# ----------------------
# After z-score, each column is centred near 0 with unit scale **inside
# this patient**. Cohort z-score is unnecessary once that is done.

# %%
# Next
# ----
# * Supervoxels: :doc:`/auto_examples/07_advanced/plot_03_supervoxels`.
# * Two-step tutorial: :doc:`/auto_examples/00_introductory_tutorials/plot_01_two_step`.
