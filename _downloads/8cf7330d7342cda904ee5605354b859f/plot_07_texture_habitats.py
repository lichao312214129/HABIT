"""
Clustering habitats from a texture field
========================================

**Background.** Texture maps can be clustered like intensities: each ROI
voxel becomes one row of texture values, and k-means groups voxels with
similar local patterns into habitats.

**Purpose.** You get three GLCM columns for one subject, three habitats
fitted directly on its voxels, a GLCM Contrast map, and the habitat
overlay.

**Key terms.**

* **GLCM Contrast** / **kernel radius** -- see
  :doc:`/auto_examples/02b_stages/plot_05_voxel_texture_gpu`.
* **Correlation / Idm** -- two more GLCM features: how linearly related
  neighbouring grey levels are, and how homogeneous the neighbourhood is.
* **per-subject fit** -- see :doc:`/auto_examples/02b_stages/plot_06_derived_map`.

Input: a per-voxel GLCM :class:`~habit.contracts.VoxelFeatureField`.
Output: a :class:`~habit.contracts.HabitatMap`. Stages:
:func:`~habit.voxel_features.extract_voxel_texture`, then a per-subject
``fit``.
"""

# %%
# Load one subject
# ----------------
# Change ``DATA`` / ``MODALITIES`` / ``ROI`` to your preprocessed layout.
# Contrast, Correlation, and Idm share one GLCM, so this is one texture
# pass rather than three.
# sphinx_gallery_thumbnail_number = 2
from pathlib import Path

import matplotlib.pyplot as plt

from habit.contracts import Cohort, cohort_from_directory
from habit.datasets import fetch_demo
from habit.habitat_model import KMeansHabitatModelFitter
from habit.pipeline import voxel_units
from habit.viz import plot_habitat_overlay, plot_voxel_texture_slice
from habit.voxel_features import extract_voxel_texture

DATA = fetch_demo()
MODALITIES = ("LAP",)
ROI = "LAP"
subject = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[0]
image = subject.image(ROI)
mask = subject.mask(ROI)
print(subject.subject_id)

# %%
# Extract three GLCM features in one pass
# ---------------------------------------
# Contrast, Correlation, and Idm share one grey-level matrix.
field = extract_voxel_texture(
    image,
    mask,
    kernel_radius=1,
    bin_width=25.0,
    feature_classes={"glcm": ["Contrast", "Correlation", "Idm"]},
)
print(f"{field.values.shape[0]} voxels, columns: {list(field.feature_names)}")
print(field.feature_frame().head())

# %%
# Fit and assign on these voxels
# ------------------------------
# No partition: every ROI voxel is its own clustering unit. The three
# columns are not rescaled here, so a column with a wider range weighs
# more in the distance (see plot_06_texture_preprocessing).
units = voxel_units(field)
fitter = KMeansHabitatModelFitter(n_habitats=3, n_init=3)
fitter.set_random_state(0)
model = fitter.fit([units], cohort=Cohort([subject], name="one"))
habitat_map = model.assigner()(units)
print(model.summary())

# %%
# GLCM Contrast
# -------------
Path("out").mkdir(exist_ok=True)
fig_field = plot_voxel_texture_slice(
    field,
    feature=0,
    anatomy=image,
    roi_mask=mask,
    title="GLCM Contrast",
    crop_to="roi",
)
fig_field.savefig("out/texture_field.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Habitats from the texture field
# -------------------------------
fig = plot_habitat_overlay(
    image,
    habitat_map,
    title="habitats (GLCM texture)",
    crop_to="labels",
)
fig.savefig("out/texture_habitats_overlay.png", dpi=150, bbox_inches="tight")
plt.show()
