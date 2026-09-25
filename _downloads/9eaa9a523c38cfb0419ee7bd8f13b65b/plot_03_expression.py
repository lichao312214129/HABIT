"""
Expression voxel features
=========================

**Background.** Clinicians often read DCE images as ratios (how much a
voxel enhances, how fast it washes out) rather than raw signal. An
expression extractor computes such ratios per voxel from a formula
string, without writing a class.

**Purpose.** You get a voxel-feature table with four derived columns and
a map of arterial relative enhancement inside the ROI.

**When to use.** When each new column is simple arithmetic on your
modalities; for anything more, see
:doc:`/auto_examples/01_complete/plot_10_feature_plugin`.

**Key terms.**

* **voxel feature** / **extract** -- see
  :doc:`/auto_examples/02b_stages/plot_01_voxel_intensities`.
* **derived map** -- a voxel-wise image computed from other images, such
  as relative enhancement ``(LAP - pre_contrast) / pre_contrast``.

:class:`~habit.voxel_features.ExpressionVoxelFeatures` builds one column
per formula. Names in the formula are modality keys on the subject
(``LAP``, ``pre_contrast``, ...). ``eps`` keeps a zero denominator from
breaking a ratio.

Clustering those columns is a later page
(:doc:`/auto_examples/02b_stages/plot_06_derived_map`).
"""

# %%
# Change ``DATA`` / ``MODALITIES`` / ``ROI`` to your preprocessed layout.
# sphinx_gallery_thumbnail_number = 1
from pathlib import Path

import matplotlib.pyplot as plt

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.viz import plot_voxel_texture_slice
from habit.voxel_features import ExpressionVoxelFeatures

DATA = fetch_demo()
# All four DCE phases: unenhanced, arterial, portal-venous, delayed.
MODALITIES = ("pre_contrast", "LAP", "PVP", "delay_3min")
ROI = "LAP"
subject = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[0]
print(subject.subject_id, sorted(subject.images))

# %%
# Each key becomes one column of the voxel field.
# Only ROI voxels are evaluated; the formulas use the modality keys above.
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
# Arterial relative enhancement inside the ROI.
Path("out").mkdir(exist_ok=True)
fig = plot_voxel_texture_slice(
    field,
    feature=0,
    anatomy=subject.image(ROI),
    roi_mask=subject.mask(ROI),
    title="relative enhancement (LAP)",
    crop_to="roi",
)
fig.savefig("out/expression_relative_enhancement.png", dpi=150, bbox_inches="tight")
plt.show()
