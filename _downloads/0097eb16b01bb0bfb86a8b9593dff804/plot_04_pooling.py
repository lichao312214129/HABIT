"""
Pooling
=======

**Background.** After each subject has been turned into clustering
units (supervoxels or voxels), ``pool`` stacks those rows into one
matrix so a single cohort model can be fitted.

**Purpose.** You will build subject-level units for two patients, pool
them, and print the pooled shape.

**When to use.** Two-step and pooled-voxel designs. Inside-each-subject
designs omit ``pool``.

**Key terms.**

* **pool** -- concatenate per-subject unit tables; no transform.
* **cohort model** -- centroids learned once on the pooled matrix.
"""

# %%
# Build units for two subjects, then pool
# ---------------------------------------
# sphinx_gallery_thumbnail_number = 1
from pathlib import Path

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.feature_preprocessing import (
    SubjectPreprocessingChain,
    Winsorizing,
    ZScoreScaling,
)
from habit.pipeline import SubjectPipeline
from habit.pipeline.pooling import fan_in
from habit.supervoxel import KMeansSupervoxelizer, MeanSupervoxelFeatures
from habit.voxel_features import RawVoxelFeatures

DATA = fetch_demo()
MODALITIES = ("pre_contrast", "LAP", "PVP", "delay_3min")
ROI = "LAP"
train = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:2]
Path("out").mkdir(exist_ok=True)

extractor = RawVoxelFeatures(modalities=list(MODALITIES), roi=ROI)
subject_chain = SubjectPreprocessingChain(
    [Winsorizing(winsor_limits=(0.01, 0.01)), ZScoreScaling()]
)
partition = KMeansSupervoxelizer(n_supervoxels=30)
partition.set_random_state(0)
summarize = MeanSupervoxelFeatures()
pipe = SubjectPipeline(
    voxel_feature_extractor=extractor,
    voxel_feature_preprocessor=subject_chain,
    supervoxelizer=partition,
    supervoxel_feature_extractor=summarize,
    habitat_assigner=None,
)

unit_list = [pipe.units(subject) for subject in train]
for subject, units in zip(train, unit_list):
    print(f"{subject.subject_id}: {units.features.shape[0]} supervoxels")

pooled = fan_in(unit_list)
print("pooled rows:", pooled.frame.shape[0], "cols:", pooled.frame.shape[1])
assert pooled.frame.shape[0] == sum(u.features.shape[0] for u in unit_list)

# %%
# How to read the result
# ----------------------
# Pooling only stacks rows. Subject z-score already put each column on a
# common scale inside each patient, so cohort z-score is not required by
# default.

# %%
# Next
# ----
# * Fit: :doc:`/auto_examples/07_advanced/plot_05_fit`.
# * Two-step tutorial: :doc:`/auto_examples/00_introductory_tutorials/plot_01_two_step`.
