"""
Clustering habitats from a derived map
======================================

Input: one :class:`~habit.contracts.Subject` with unenhanced and arterial
phases. Output: a :class:`~habit.contracts.HabitatMap`. Stage:
``extract_voxel_features`` with ``expression``, then a per-subject ``fit``.

Registration of the phases stays in
:doc:`/auto_examples/02_voxel/plot_02_custom_features`.
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
from habit.voxel_features import ExpressionVoxelFeatures

DATA = fetch_demo()
MODALITIES = ("pre_contrast", "LAP")
ROI = "LAP"
subject = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[0]

extractor = ExpressionVoxelFeatures(
    features={
        "relative_enhancement_lap": "(LAP - pre_contrast) / (pre_contrast + eps)",
    },
    roi=ROI,
)
field = extractor(subject)
print(field.feature_frame().head())
units = voxel_units(field)
fitter = KMeansHabitatModelFitter(n_habitats=3, n_init=3)
fitter.set_random_state(0)
model = fitter.fit([units], cohort=Cohort([subject], name="one"))
habitat_map = model.assigner()(units)
print(model.summary())

fig = plot_habitat_overlay(
    subject.image(ROI),
    habitat_map,
    title="habitats (relative enhancement)",
    axis=0,
    crop_to="labels",
)
Path("out").mkdir(exist_ok=True)
fig.savefig("out/derived_map_overlay.png", dpi=150, bbox_inches="tight")
plt.show()
