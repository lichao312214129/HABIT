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
# All four DCE phases: unenhanced, arterial, portal-venous, delayed.
MODALITIES = ("pre_contrast", "LAP", "PVP", "delay_3min")
ROI = "LAP"
subject = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[0]

# ExpressionVoxelFeatures evaluates arithmetic on the raw phases.
# Each key becomes one column in the VoxelFeatureField.
extractor = ExpressionVoxelFeatures(
    features={
        "rel_enh_lap": "(LAP - pre_contrast) / (pre_contrast + eps)",
        "rel_enh_pvp": "(PVP - pre_contrast) / (pre_contrast + eps)",
        "rel_enh_delay": "(delay_3min - pre_contrast) / (pre_contrast + eps)",
        "washout": "(LAP - delay_3min) / (LAP + eps)",
    },
    roi=ROI,
)
field = extractor(subject)
print(field.feature_frame().head())

# voxel_units wraps the field as one-voxel Supervoxelization units,
# so the fitter sees every voxel as its own clustering unit.
units = voxel_units(field)
fitter = KMeansHabitatModelFitter(n_habitats=3, n_init=3)
fitter.set_random_state(0)
model = fitter.fit([units], cohort=Cohort([subject], name="one"))
habitat_map = model.assigner()(units)
print(model.summary())

Path("out").mkdir(exist_ok=True)
fig_field = plot_voxel_texture_slice(
    field,
    feature=0,
    anatomy=subject.image(ROI),
    roi_mask=subject.mask(ROI),
    title="relative enhancement (LAP)",
    crop_to="roi",
)
fig_field.savefig("out/relative_enhancement.png", dpi=150, bbox_inches="tight")
plt.show()

fig = plot_habitat_overlay(
    subject.image(ROI),
    habitat_map,
    title="habitats (derived maps)",
    crop_to="labels",
)
fig.savefig("out/derived_map_overlay.png", dpi=150, bbox_inches="tight")
plt.show()
