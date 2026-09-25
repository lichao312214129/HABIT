"""
Extracting a voxel texture
==========================

**Background.** A voxel texture value describes how grey levels vary in
a small cube around each ROI voxel, which intensity alone does not show.
Here it is computed with one function call on an image and a mask.

**Purpose.** You get a voxel-feature table with one GLCM Contrast column
and a map of it on the arterial slice.

**When to use.** When you already hold an image and a mask and want a
single texture map; the full runtime story is on
:doc:`/auto_examples/02b_stages/plot_05_voxel_texture_gpu`.

**Key terms.**

* **voxel feature** / **extract** -- see
  :doc:`/auto_examples/02b_stages/plot_01_voxel_intensities`.
* **GLCM Contrast** / **kernel radius** / **bin width** -- see
  :doc:`/auto_examples/02b_stages/plot_05_voxel_texture_gpu`.

Input: one :class:`~habit.contracts.Subject`. Output: a
:class:`~habit.contracts.VoxelFeatureField` of per-voxel GLCM Contrast.
Stage: :func:`~habit.voxel_features.extract_voxel_texture`.

GPU paths for the same IBSI texture are in
:doc:`/auto_examples/02b_stages/plot_05_voxel_texture_gpu`.
"""

# %%
# Load one subject
# ----------------
# Change ``DATA`` / ``MODALITIES`` / ``ROI`` to your preprocessed layout.
# Radius 1 is a 3x3x3 neighbourhood; ``bin_width=25`` matches the GPU page.
# sphinx_gallery_thumbnail_number = 1
from pathlib import Path

import matplotlib.pyplot as plt

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.viz import plot_voxel_texture_slice
from habit.voxel_features import extract_voxel_texture

DATA = fetch_demo()
MODALITIES = ("LAP",)
ROI = "LAP"
subject = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[0]
image = subject.image(ROI)
mask = subject.mask(ROI)
print(subject.subject_id)

# %%
# Extract GLCM Contrast
# ---------------------
# Radius 1 is a 3×3×3 neighbourhood. ``bin_width=25`` is the grey-level width.
field = extract_voxel_texture(
    image,
    mask,
    kernel_radius=1,
    bin_width=25.0,
    feature_classes={"glcm": ["Contrast"]},
)
print(f"{field.values.shape[0]} voxels, columns: {list(field.feature_names)}")
print(field.feature_frame().head())
field.feature_frame().head()

# %%
# Contrast on the arterial slice
# ------------------------------
fig = plot_voxel_texture_slice(
    field,
    feature=0,
    anatomy=image,
    roi_mask=mask,
    crop_to="roi",
)
Path("out").mkdir(exist_ok=True)
fig.savefig("out/voxel_texture_field.png", dpi=150, bbox_inches="tight")
plt.show()
