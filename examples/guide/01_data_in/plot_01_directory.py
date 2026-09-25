"""
Load from directory
===================

**Background.** Before any habitat can be found, HABIT needs to know which
images and which tumour mask belong to each patient. The simplest way is a
fixed folder layout that HABIT reads in one call.

**Purpose.** You get a :class:`~habit.contracts.Cohort` (printed), a
summary figure of the loaded subject, and a greyscale slice with the ROI
contour to check that image and mask line up.

**When to use.** Use this when your preprocessed files already follow the
HABIT folder layout (the official demo pack does). For loose files or
arrays, see the other pages in this section.

**Key terms.**

* **cohort** -- HABIT's list of subjects (``Cohort``), each with its images
  and ROI mask; this is what ``fit`` / ``fit_predict`` take.
* **subject** -- one patient (``Subject``): an id plus a dictionary of images
  and a dictionary of masks.
* **series / modality** -- one image of that patient, such as the arterial
  (``LAP``) or portal-venous (``PVP``) phase; the folder name under
  ``images/<subject>/`` is the name you pass in ``modalities=``.
* **ROI / mask** -- the region of interest, usually the whole tumour, stored
  as an integer mask; only voxels inside it are analysed (0 = background).
  The folder name under ``masks/<subject>/`` is the name you pass in
  ``roi=``.

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
# ``modalities`` picks the image folders to load; ``roi`` picks the mask folder.
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
