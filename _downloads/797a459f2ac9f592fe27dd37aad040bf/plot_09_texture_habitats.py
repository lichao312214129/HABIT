"""
Clustering habitats from a texture field
========================================

Input: a local-entropy :class:`~habit.contracts.VoxelFeatureField`.
Output: a :class:`~habit.contracts.HabitatMap`. Stages:
``extract_voxel_features`` with ``local_entropy``, then a per-subject
``fit``.
"""

# %%
# Change ``DATA`` / ``MODALITIES`` / ``ROI`` to your preprocessed layout.
from pathlib import Path

import matplotlib.pyplot as plt

from habit.contracts import Cohort, cohort_from_directory
from habit.datasets import fetch_demo
from habit.habitat_model import KMeansHabitatModelFitter
from habit.pipeline import voxel_units
from habit.viz import plot_habitat_overlay
from habit.voxel_features import LocalEntropyVoxelFeatures

DATA = fetch_demo()
MODALITIES = ("LAP",)
ROI = "LAP"
subject = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[0]

field = LocalEntropyVoxelFeatures(
    modalities=list(MODALITIES), roi=ROI, kernel_size=5, bins=32
)(subject)
units = voxel_units(field)
fitter = KMeansHabitatModelFitter(n_habitats=3, n_init=3)
fitter.set_random_state(0)
model = fitter.fit([units], cohort=Cohort([subject], name="one"))
habitat_map = model.assigner()(units)
print(model.summary())

fig = plot_habitat_overlay(
    subject.image(ROI),
    habitat_map,
    title="habitats (local entropy)",
    axis=0,
    crop_to="labels",
)
Path("out").mkdir(exist_ok=True)
fig.savefig("out/texture_habitats_overlay.png", dpi=150, bbox_inches="tight")
plt.show()
