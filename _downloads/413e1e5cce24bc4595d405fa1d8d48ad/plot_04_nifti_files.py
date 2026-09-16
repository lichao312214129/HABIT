"""
Load from NIfTI files
=====================

When you have one image file and one ROI file (NIfTI, NRRD, or MetaImage),
use HABIT's file readers or wrap the same paths as lazy
:class:`~habit.adapters.FileImageRef` handles on a
:class:`~habit.contracts.Subject`. There is no single
``load(image, roi)`` helper; these two calls are the public pair.
"""

# %%
# Immediate volumes: :func:`~habit.image.read_image` and
# :func:`~habit.image.read_mask`. Both go through SimpleITK, so ``.nii``,
# ``.nii.gz``, ``.nrrd``, ``.mha``, and ``.mhd`` work. Align the pair when
# the grids differ.
from pathlib import Path

import matplotlib.pyplot as plt

from habit.adapters import FileImageRef
from habit.contracts import Subject
from habit.datasets import fetch_demo
from habit.image import (
    GeometryPolicy,
    ImageMaskPair,
    align_image_mask,
    read_image,
    read_mask,
)
from habit.viz import plot_nifti_ingest

DATA = fetch_demo()
# Change IMAGE / MASK to your own NIfTI or NRRD paths.
IMAGE = next(
    path for path in (DATA / "images" / "subj001" / "LAP").iterdir() if path.is_file()
)
MASK = next(
    path for path in (DATA / "masks" / "subj001" / "LAP").iterdir() if path.is_file()
)
image = read_image(IMAGE, modality="LAP")
mask = read_mask(MASK)
pair = align_image_mask(
    ImageMaskPair(image, mask),
    policy=GeometryPolicy.RESAMPLE_MASK,
)
print(
    f"read_image/read_mask: image shape={pair.image.data.shape}, "
    f"mask shape={pair.mask.data.shape}, spacing={pair.image.geometry.spacing}"
)

# %%
# Pipeline object: wrap the same two paths as :class:`~habit.adapters.FileImageRef`
# and assemble a :class:`~habit.contracts.Subject`. Pixel data stay on disk until
# ``subject.image(...)`` / ``subject.mask(...)``.
subject = Subject(
    subject_id="subj001",
    images={"LAP": FileImageRef(IMAGE, is_mask=False, role_name="LAP")},
    masks={"LAP": FileImageRef(MASK, is_mask=True, role_name="LAP")},
)
volume = subject.image("LAP")
roi = subject.mask("LAP")
print(
    f"FileImageRef Subject: id={subject.subject_id}, "
    f"LAP shape={volume.data.shape}, spacing={volume.geometry.spacing}"
)

# %%
# Visual summary of this route: the image/mask file pair becomes a plottable
# Subject (right panel zooms to the ROI). Pass the
# :class:`~habit.image.ImageVolume` (not ``.data``).
Path("out").mkdir(exist_ok=True)
fig = plot_nifti_ingest(volume, roi)
fig.savefig("out/data_from_nifti_ingest.png", dpi=150, bbox_inches="tight")
plt.show()
