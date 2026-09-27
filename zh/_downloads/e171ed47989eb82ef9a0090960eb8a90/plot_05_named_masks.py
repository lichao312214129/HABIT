"""
Named masks
===========

**Background.** Habitat analysis needs a region of interest (ROI). The
ROI is often the tumour, but studies also use a user-drawn peritumoral
ring or other named mask that already exists on disk.

**Purpose.** You will load a subject that carries more than one mask
key, select a named mask as the ROI, and confirm the ROI voxel count.

**When to use.** When your preprocessed tree already stores a
peritumoral (or other) mask you drew yourself — HABIT does not generate
that mask on this page.

**Key terms.**

* **mask key / ROI name** -- the folder name under ``masks/<subject>/``.
* **named mask** -- any mask you supply; HABIT only reads it.
"""

# %%
# Load one subject and list mask keys
# -----------------------------------
# Change DATA / MODALITIES / ROI to your preprocessed layout. The demo
# pack stores one mask per DCE phase; use those keys as stand-ins for
# any named mask you would add (for example ``peritumoral``).
# sphinx_gallery_thumbnail_number = 1
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.viz import plot_intensity_slice

DATA = fetch_demo()
MODALITIES = ("pre_contrast", "LAP", "PVP", "delay_3min")
# Stand-in for a user-supplied named mask (e.g. peritumoral). Swap the
# string to the folder name of the mask you drew.
ROI = "LAP"
subject = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[0]
print("subject:", subject.subject_id)
print("mask keys on disk:", sorted(subject.masks))
mask = subject.mask(ROI)
n_roi = int(np.count_nonzero(np.asarray(mask.data) > 0))
print(f"ROI {ROI!r}: {n_roi} voxels")
Path("out").mkdir(exist_ok=True)

# %%
# Show the named mask on the anatomy
# ----------------------------------
fig = plot_intensity_slice(
    subject.image("LAP"),
    roi_mask=mask,
    roi_contour=True,
    image_label="LAP",
    title=f"{subject.subject_id}: named mask {ROI!r}",
)
fig.savefig("out/named_mask.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# How to read the result
# ----------------------
# The mask key is just a name. Put your peritumoral (or other) mask under
# ``masks/<subject_id>/<your_name>/`` and set ``ROI = "your_name"``.

# %%
# Next
# ----
# * Directory layout: :doc:`/auto_examples/06_manipulating_images/plot_01_directory`.
# * Two-step habitats: :doc:`/auto_examples/00_introductory_tutorials/plot_01_two_step`.
