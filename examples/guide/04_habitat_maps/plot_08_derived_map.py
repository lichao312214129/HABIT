"""
Clustering habitats from a derived map
======================================

Input: one :class:`~habit.contracts.Subject` with unenhanced and arterial
phases. Output: a :class:`~habit.contracts.HabitatMap`. Stage:
``extract_voxel_features`` with ``expression``, then a per-subject ``fit``.
"""

# %%
# Change ``DATA`` / ``MODALITIES`` / ``ROI`` to your preprocessed layout.
# sphinx_gallery_thumbnail_number = 2
from pathlib import Path

import matplotlib.pyplot as plt

from habit.contracts import Cohort, cohort_from_directory
from habit.datasets import fetch_demo
from habit.habitat_model import KMeansHabitatModelFitter
from habit.pipeline import voxel_units
from habit.viz import plot_habitat_overlay, plot_voxel_texture_slice
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

Path("out").mkdir(exist_ok=True)
fig_field = plot_voxel_texture_slice(
    field,
    anatomy=subject.image(ROI),
    roi_mask=subject.mask(ROI),
    title="relative enhancement",
    crop_to="roi",
)
fig_field.savefig("out/relative_enhancement.png", dpi=150, bbox_inches="tight")
plt.show()

fig = plot_habitat_overlay(
    subject.image(ROI),
    habitat_map,
    title="habitats (relative enhancement)",
    crop_to="labels",
)
fig.savefig("out/derived_map_overlay.png", dpi=150, bbox_inches="tight")
plt.show()
