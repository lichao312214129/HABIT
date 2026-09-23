"""
Load from directory
===================

Build a :class:`~habit.contracts.Cohort` from a folder tree with
:func:`~habit.contracts.cohort_from_directory`. The tree is
``images/<subject>/<series>/<one file>`` and
``masks/<subject>/<roi>/<one file>``. Below: several people from the
tree, then one person taken from that cohort.
"""

# %%
# Directory layout
# ----------------
# :func:`~habit.datasets.fetch_demo` prints the absolute path and an
# inventory. That printed tree is what your own ``DATA`` must match.
from pathlib import Path

import matplotlib.pyplot as plt

from habit.contracts import Cohort, cohort_from_directory
from habit.datasets import fetch_demo, inspect_preprocessed_root
from habit.viz import plot_directory_ingest, plot_intensity_slice

# Official pack (first call downloads; later calls reuse the cache).
# Your own data: DATA = r"D:/my_study/preprocessed"
DATA = fetch_demo()
print(inspect_preprocessed_root(DATA))
# Several people: every subject folder under DATA.
many = cohort_from_directory(DATA, modalities=("LAP",), roi="LAP", name="many")
print(many)

# One person: take one Subject out and wrap it again. A plain list is not a cohort.
one = Cohort([many[0]], name="one")
print(one)

# %%
# Visual summary of this route: the directory tree becomes plottable
# Subjects (right panel zooms to the ROI; badge colours: folder / image /
# mask).
subject = one[0]
Path("out").mkdir(exist_ok=True)
fig_ingest = plot_directory_ingest(subject.image("LAP"), subject.mask("LAP"))
fig_ingest.savefig("out/data_in_ingest.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Anatomy check: greyscale LAP slice of the first subject, with the ROI
# contour. Pass the :class:`~habit.image.ImageVolume` (not ``.data``).
fig_anatomy = plot_intensity_slice(
    subject.image("LAP"),
    roi_mask=subject.mask("LAP"),
    title="LAP anatomy",
    roi_contour=True,
)
fig_anatomy.savefig("out/data_in_anatomy.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Several series share one mask folder. ``modalities`` are image folders.
# ``roi`` is the mask folder. The names can differ. DICOM is
# :doc:`/how_to/preprocess`. Loose files are
# :doc:`/auto_examples/01_data_in/plot_04_nifti_files`.
two_series = cohort_from_directory(
    DATA, modalities=("LAP", "PVP"), roi="LAP", name="two_series"
)
print(list(two_series[0].images), list(two_series[0].masks))
