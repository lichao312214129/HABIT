"""
Chaining the steps
==================

Input: one :class:`~habit.contracts.Subject`. Output: a
:class:`~habit.contracts.HabitatMap`.
:class:`~habit.pipeline.SubjectPipeline` runs extract, partition, and
assign. Swap ``RawVoxelFeatures`` for another extractor to change only
the :class:`~habit.contracts.VoxelFeatureField`.
"""

# %%
# Change ``DATA`` / ``MODALITIES`` / ``ROI`` to your preprocessed layout.
from pathlib import Path

import matplotlib.pyplot as plt

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.habitat_model import KMeansHabitatModelFitter
from habit.pipeline import SubjectPipeline
from habit.supervoxel import SlicSupervoxelizer
from habit.viz import plot_habitat_overlay
from habit.voxel_features import RawVoxelFeatures

DATA = fetch_demo()
MODALITIES = ("LAP",)
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:2]
subject = cohort[0]

voxel = RawVoxelFeatures(modalities=list(MODALITIES), roi=ROI)
slic = SlicSupervoxelizer(n_supervoxels=8, compactness=10.0)
units = [slic(voxel(item)) for item in cohort]
fitter = KMeansHabitatModelFitter(n_habitats=3, n_init=3)
fitter.set_random_state(0)
model = fitter.fit(units, cohort=cohort)

pipe = SubjectPipeline(voxel, slic, model.assigner())
habitat_map = pipe(subject)

fig = plot_habitat_overlay(
    subject.image(ROI),
    habitat_map,
    title="habitats",
    axis=0,
    crop_to="labels",
)
Path("out").mkdir(exist_ok=True)
fig.savefig("out/chained_habitats.png", dpi=150, bbox_inches="tight")
plt.show()
