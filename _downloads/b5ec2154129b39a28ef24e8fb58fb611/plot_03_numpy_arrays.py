"""
Load from NumPy arrays
======================

Wrap raw NumPy arrays or deep-learning tensors with
:class:`~habit.contracts.ArrayImageRef` and explicit
:class:`~habit.contracts.Geometry`. Axis order is ``(z, y, x)``;
mask arrays must be **integer labels** (``0`` = background).
"""

# %%
# Read the same demo files, then build a Subject from arrays.
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from habit.contracts import ArrayImageRef, Geometry, Subject
from habit.datasets import fetch_demo
from habit.viz import plot_intensity_slice
from habit.voxel_features import RawVoxelFeatures

DATA = fetch_demo()
MODALITIES = ("LAP",)
ROI = "LAP"
image_path = next(
    path for path in (DATA / "images" / "subj001" / "LAP").iterdir() if path.is_file()
)
mask_path = next(
    path for path in (DATA / "masks" / "subj001" / "LAP").iterdir() if path.is_file()
)
sitk_image = sitk.ReadImage(str(image_path))
sitk_mask = sitk.ReadImage(str(mask_path))
array = sitk.GetArrayFromImage(sitk_image)
mask = np.asarray(sitk.GetArrayFromImage(sitk_mask), dtype=np.int32)
geometry = Geometry.from_array(
    array.shape,
    spacing=tuple(sitk_image.GetSpacing()),
    origin=tuple(sitk_image.GetOrigin()),
    direction=tuple(sitk_image.GetDirection()),
)
np_subject = Subject(
    subject_id="subj001",
    images={"LAP": ArrayImageRef(array=array, geometry=geometry)},
    masks={"LAP": ArrayImageRef(array=mask, geometry=geometry)},
)
field = RawVoxelFeatures(modalities=["LAP"])(np_subject)
print(
    f"NumPy Subject: id={np_subject.subject_id}, "
    f"LAP shape={np_subject.image('LAP').data.shape}, "
    f"voxels={field.values.shape[0]}"
)
print(field.feature_frame().head())
field.feature_frame().head()

# %%
# Anatomy slice with ROI contour — confirms the array-built Subject is plottable.
Path("out").mkdir(exist_ok=True)
fig = plot_intensity_slice(
    np_subject.image("LAP"),
    roi_mask=np_subject.mask("LAP"),
    roi_contour=True,
    title="NumPy-constructed Subject: LAP with ROI",
)
fig.savefig("out/numpy_subject_slice.png", dpi=150, bbox_inches="tight")
plt.show()
