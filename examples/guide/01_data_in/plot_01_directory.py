"""
Load from directory
===================

HABIT operators take a :class:`~habit.contracts.Subject` (or a
:class:`~habit.contracts.Cohort` of them). Build that object from a
preprocessed directory tree with :func:`~habit.contracts.cohort_from_directory`.
"""

# %%
# Directory layout
# ----------------
# :func:`~habit.datasets.fetch_demo` prints the absolute path and an
# inventory. That printed tree is what your own ``DATA`` must match.
from pathlib import Path
import os

import matplotlib.pyplot as plt

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo, inspect_preprocessed_root
from habit.viz import plot_intensity_slice

# Official pack (first call downloads; later calls reuse the cache).
# Your own data: DATA = r"D:/my_study/preprocessed"
DATA = fetch_demo()
print(inspect_preprocessed_root(DATA))
MODALITIES = ("LAP",)
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)
print(list(cohort.subject_ids), list(cohort[0].images.keys()))

# %%
# Anatomy check: greyscale LAP slice of the first subject, with the ROI
# contour. Pass the :class:`~habit.api.image.ImageVolume` (not ``.data``).
subject = cohort[0]
Path("out").mkdir(exist_ok=True)
fig_anatomy = plot_intensity_slice(
    subject.image("LAP"),
    roi_mask=subject.mask(ROI),
    title="LAP anatomy",
    roi_contour=True,
)
fig_anatomy.savefig("out/data_in_anatomy.png", dpi=150, bbox_inches="tight")
if os.environ.get("HABIT_NO_VIEW") != "1":
    plt.show()
