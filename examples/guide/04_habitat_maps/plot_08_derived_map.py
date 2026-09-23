"""
Clustering habitats from a derived map
======================================

Input: one :class:`~habit.contracts.Subject` with unenhanced and arterial
phases. Output: a :class:`~habit.contracts.HabitatMap`. Stage:
``extract_voxel_features`` with ``expression``, then a per-subject ``fit``.
"""

# %%
# Load one subject
# ----------------
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
print(subject.subject_id, sorted(subject.images))

# %%
# Build the four derived maps
# ---------------------------
# Each key becomes one column. ``eps`` keeps a zero denominator from
# breaking the ratio.
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
print(f"{field.values.shape[0]} voxels, columns: {list(field.feature_names)}")
print(field.feature_frame().head())

# %%
# Fit and assign on these voxels
# ------------------------------
# ``voxel_units`` makes every ROI voxel its own clustering unit.
units = voxel_units(field)
fitter = KMeansHabitatModelFitter(n_habitats=3, n_init=3)
fitter.set_random_state(0)
model = fitter.fit([units], cohort=Cohort([subject], name="one"))
habitat_map = model.assigner()(units)
print(model.summary())

# %%
# Relative enhancement
# --------------------
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

# %%
# Habitats from the derived maps
# ------------------------------
fig = plot_habitat_overlay(
    subject.image(ROI),
    habitat_map,
    title="habitats (derived maps)",
    crop_to="labels",
)
fig.savefig("out/derived_map_overlay.png", dpi=150, bbox_inches="tight")
plt.show()
