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
# sphinx_gallery_thumbnail_number = 1
from pathlib import Path

import matplotlib.pyplot as plt

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.habitat_model import KMeansHabitatModelFitter
from habit.pipeline import SubjectPipeline
from habit.supervoxel import SlicSupervoxelizer
from habit.viz import plot_partition_triptych
from habit.voxel_features import RawVoxelFeatures

DATA = fetch_demo()
# Three DCE phases: unenhanced, arterial, and portal-venous.
MODALITIES = ("pre_contrast", "LAP", "PVP")
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:2]
subject = cohort[0]

voxel = RawVoxelFeatures(modalities=list(MODALITIES), roi=ROI)
slic = SlicSupervoxelizer(n_supervoxels=100, compactness=10.0)
units = [slic(voxel(item)) for item in cohort]
fitter = KMeansHabitatModelFitter(n_habitats=3, n_init=3)
fitter.set_random_state(0)
model = fitter.fit(units, cohort=cohort)

# SubjectPipeline chains the three steps: extract voxel features,
# partition into supervoxels, assign habitat labels. Calling it on one
# subject runs all three stages without any manual intermediate objects.
pipe = SubjectPipeline(voxel, slic, model.assigner())
habitat_map = pipe(subject)

fig = plot_partition_triptych(
    subject.image(ROI),
    units[0],
    habitat_map,
    axis=0,
)
Path("out").mkdir(exist_ok=True)
fig.savefig("out/chained_habitats.png", dpi=150, bbox_inches="tight")
plt.show()
