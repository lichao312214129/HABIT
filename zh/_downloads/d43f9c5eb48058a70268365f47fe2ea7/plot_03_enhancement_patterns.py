"""
Enhancement patterns
====================

**Background.** Dynamic contrast-enhanced (DCE) MRI records how a
tumour takes up and washes out contrast across several phases. Raw
intensities mix absolute brightness with that kinetic pattern.
Derived maps (ratios, differences, averages, kinetic slopes, or free
expressions) put the pattern into the feature table that habitats use.

**Purpose.** You will build every enhancement-related calculator HABIT
ships for this demo (ratio, difference, average, expression, kinetic),
inspect their columns, and fit a small per-subject habitat map on the
expression-based relative-enhancement set.

**When to use.** When habitats should follow contrast uptake / wash-out
rather than absolute signal level.

**Key terms.**

* **ratio / difference / average** -- combiners that turn intensity
  blocks into derived blocks.
* **expression** -- named formulas over phase images (AST-safe).
* **kinetic** -- wash-in / wash-out slopes from four ordered phases
  (needs per-subject acquisition times).
"""

# %%
# Load one subject with four DCE phases
# -------------------------------------
# Change DATA / MODALITIES / ROI to your preprocessed layout.
# sphinx_gallery_thumbnail_number = 1
from pathlib import Path

import matplotlib.pyplot as plt

from habit.contracts import Cohort, cohort_from_directory
from habit.datasets import fetch_demo
from habit.habitat_model import KMeansHabitatModelFitter
from habit.pipeline import voxel_units
from habit.spec import Spec
from habit.viz import plot_habitat_overlay, plot_voxel_texture_slice
from habit.voxel_features import (
    ExpressionVoxelFeatures,
    KineticVoxelFeatures,
    build_voxel_extractor,
)

DATA = fetch_demo()
MODALITIES = ("pre_contrast", "LAP", "PVP", "delay_3min")
ROI = "LAP"
subject = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[0]
print(subject.subject_id, sorted(subject.images))
Path("out").mkdir(exist_ok=True)

# %%
# Ratio, difference, and average combiners
# ----------------------------------------
# Each tree is a ``Spec``: combiners list child extractors under
# ``children``. ``ratio`` / ``difference`` need two children; ``average``
# takes one or more.
ratio_field = build_voxel_extractor(
    Spec(
        "ratio",
        {
            "children": [
                {"name": "raw", "params": {"modalities": ["LAP"], "roi": ROI}},
                {
                    "name": "raw",
                    "params": {"modalities": ["pre_contrast"], "roi": ROI},
                },
            ]
        },
    )
)(subject)
diff_field = build_voxel_extractor(
    Spec(
        "difference",
        {
            "children": [
                {"name": "raw", "params": {"modalities": ["LAP"], "roi": ROI}},
                {
                    "name": "raw",
                    "params": {"modalities": ["pre_contrast"], "roi": ROI},
                },
            ]
        },
    )
)(subject)
avg_field = build_voxel_extractor(
    Spec(
        "average",
        {
            "children": [
                {"name": "raw", "params": {"modalities": ["LAP"], "roi": ROI}},
                {"name": "raw", "params": {"modalities": ["PVP"], "roi": ROI}},
            ]
        },
    )
)(subject)
print("ratio columns:", list(ratio_field.feature_names))
print("difference columns:", list(diff_field.feature_names))
print("average columns:", list(avg_field.feature_names))

# %%
# Expression maps (relative enhancement and wash-out)
# ---------------------------------------------------
# ``eps`` avoids division by zero. Each key becomes one column.
expression = ExpressionVoxelFeatures(
    features={
        "rel_enh_lap": "(LAP - pre_contrast) / (pre_contrast + eps)",
        "rel_enh_pvp": "(PVP - pre_contrast) / (pre_contrast + eps)",
        "rel_enh_delay": "(delay_3min - pre_contrast) / (pre_contrast + eps)",
        "washout": "(LAP - delay_3min) / (LAP + eps)",
    },
    roi=ROI,
)
expr_field = expression(subject)
print("expression columns:", list(expr_field.feature_names))
print(expr_field.feature_frame().head())

# %%
# Kinetic slopes (four phases + acquisition times)
# ------------------------------------------------
# Wash-in / wash-out need true inter-phase intervals. The demo pack has
# no timestamp table, so this page supplies a small synthetic table
# (same shape a real study would load from file). Pre-contrast lead time
# is handled inside the extractor.
PHASE_TIMES = {
    subject.subject_id: {
        "LAP": "10-00-30",
        "PVP": "10-01-00",
        "delay_3min": "10-03-00",
    }
}
kinetic = KineticVoxelFeatures(
    timestamps=PHASE_TIMES,
    phases=list(MODALITIES),
    roi=ROI,
)
kin_field = kinetic(subject)
print("kinetic columns:", list(kin_field.feature_names))
print(kin_field.feature_frame().head())

# %%
# Habitats from the expression maps
# ---------------------------------
# One-subject fit: habitat ids are local to this patient.
units = voxel_units(expr_field)
fitter = KMeansHabitatModelFitter(n_habitats=3, n_init=3)
fitter.set_random_state(0)
model = fitter.fit([units], cohort=Cohort([subject], name="one"))
habitat_map = model.assigner()(units)
print(model.summary())

fig = plot_voxel_texture_slice(
    expr_field,
    feature="rel_enh_lap",
    title=f"{subject.subject_id}: relative enhancement (LAP)",
)
fig.savefig("out/enhancement_rel_enh_lap.png", dpi=150, bbox_inches="tight")
plt.show()

fig = plot_habitat_overlay(
    subject.image("LAP"),
    habitat_map,
    title=f"{subject.subject_id}: habitats from enhancement patterns",
    crop_to="labels",
)
fig.savefig("out/enhancement_habitats.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# How to read the result
# ----------------------
# Ratio, difference, average, expression, and kinetic all produced
# columns on this subject. The three habitats below are defined only
# inside this patient; do not compare their ids to another subject's map.

# %%
# Next
# ----
# * Combine intensity and texture:
#   :doc:`/auto_examples/02_features/plot_04_combining_features`.
# * Full two-step design:
#   :doc:`/auto_examples/00_introductory_tutorials/plot_01_two_step`.
