"""
Load from NIfTI files
=====================

Point at one image file and one mask file (``.nii``, ``.nii.gz``, ``.nrrd``,
``.mha``, ``.mhd``). Change the paths below to your own files. One person
and several people are each a :class:`~habit.contracts.Cohort`.
"""

# %%
# One person. ``role_name`` is the series or ROI name stored on the volume.
# Use the same string as the dictionary key.
from pathlib import Path

import matplotlib.pyplot as plt

from habit.adapters import FileImageRef
from habit.contracts import Cohort, Subject
from habit.datasets import fetch_demo
from habit.image import (
    GeometryPolicy,
    ImageMaskPair,
    align_image_mask,
    read_image,
    read_mask,
    validate_geometry,
)
from habit.viz import plot_nifti_ingest

DATA = fetch_demo()
# Change these two paths to your files.
IMAGE = DATA / "images" / "subj001" / "LAP" / "WATER__WATER__Ax_Dyn_LAVA_Flex+C_Series0009.nrrd"
MASK = DATA / "masks" / "subj001" / "LAP" / "WATER__BH_Ax_LAVA_Flex_10min_Series0017_mask.nrrd"

image = read_image(IMAGE, modality="LAP")
mask = read_mask(MASK)
report = validate_geometry(image, mask)
print(report.compatible, report.mismatches)
if report.compatible:
    pair = ImageMaskPair(image, mask, report)
else:
    pair = align_image_mask(
        ImageMaskPair(image, mask),
        policy=GeometryPolicy.RESAMPLE_MASK,
    )
print(pair.image.data.shape, pair.mask.data.shape)

subject = Subject(
    subject_id="subj001",
    images={"LAP": FileImageRef(IMAGE, is_mask=False, role_name="LAP")},
    masks={"LAP": FileImageRef(MASK, is_mask=True, role_name="LAP")},
)
one = Cohort([subject], name="one")
print(one)

volume = subject.image("LAP")
roi = subject.mask("LAP")

# %%
# Visual summary of this route: the image/mask file pair becomes a plottable
# Subject (right panel zooms to the ROI). Pass the
# :class:`~habit.image.ImageVolume` (not ``.data``).
Path("out").mkdir(exist_ok=True)
fig = plot_nifti_ingest(volume, roi)
fig.savefig("out/data_from_nifti_ingest.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Several people. One :class:`~habit.contracts.Subject` per person, then one
# cohort. ``subject_id`` values must be unique. Pass ``many`` to
# ``fit_predict``, not a list.
IMAGE_2 = DATA / "images" / "subj002" / "LAP" / "012_WATERWATERAxDynLAVAFlexC.nrrd"
MASK_2 = DATA / "masks" / "subj002" / "LAP" / "016_WATERWATERBHAxLAVAFlex5min_mask.nrrd"
subject_2 = Subject(
    subject_id="subj002",
    images={"LAP": FileImageRef(IMAGE_2, is_mask=False, role_name="LAP")},
    masks={"LAP": FileImageRef(MASK_2, is_mask=True, role_name="LAP")},
)
many = Cohort([subject, subject_2], name="many")
print(many)
