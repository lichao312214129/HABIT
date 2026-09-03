#!/usr/bin/env python
"""
Get data in: directory, SimpleITK, and NumPy — all from the official demo pack.

Accompanies ``docs/source/examples/data_from_arrays.rst``.

Run from the repository root::

    python docs/source/examples/scripts/data_from_arrays_demo.py
"""

from __future__ import annotations

# BEGIN disk
from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo, inspect_preprocessed_root

# Official pack (first call downloads; later calls reuse the cache).
# Your own data: DATA = r"D:/my_study/preprocessed"
DATA = fetch_demo()
print(inspect_preprocessed_root(DATA))
MODALITIES = ("LAP",)
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)
print(list(cohort.subject_ids), list(cohort[0].images.keys()))
# END disk

# BEGIN sitk
from pathlib import Path

import SimpleITK as sitk

from habit.datasets import fetch_demo
from habit.contracts import Cohort, ImageVolume, MaskVolume, Subject
from habit.voxel_features import RawVoxelFeatures

DATA = fetch_demo()
image_path = next(
    path for path in (DATA / "images" / "subj001" / "LAP").iterdir() if path.is_file()
)
mask_path = next(
    path for path in (DATA / "masks" / "subj001" / "LAP").iterdir() if path.is_file()
)
sitk_image = sitk.ReadImage(str(image_path))
sitk_mask = sitk.ReadImage(str(mask_path))
volume = ImageVolume.from_sitk(sitk_image, modality="LAP")
roi = MaskVolume.from_sitk(sitk_mask, modality="LAP")
sitk_subject = Subject(
    subject_id="subj001",
    images={"LAP": volume},
    masks={"LAP": roi},
)
sitk_cohort = Cohort([sitk_subject], name="from_sitk")
sitk_field = RawVoxelFeatures(modalities=["LAP"])(sitk_subject)
print(
    f"SimpleITK Subject: id={sitk_subject.subject_id}, "
    f"LAP shape={volume.data.shape}, voxels={sitk_field.values.shape[0]}"
)
# END sitk

# BEGIN sitk_figures
# Paste after the SimpleITK block. Uses volume and sitk_cohort.
from pathlib import Path

from habit.recipes import one_step_habitat
from habit.viz import plot_habitat_overlay

sitk_result = one_step_habitat(
    modalities=("LAP",), n_habitats=3, random_seed=0, roi="LAP"
).fit_predict(sitk_cohort)
fig_sitk = plot_habitat_overlay(
    volume,
    sitk_result.habitat_maps[0],
    title="Habitats from SimpleITK",
)
Path("out").mkdir(exist_ok=True)
fig_sitk.savefig("out/data_from_sitk_overlay.png", dpi=150, bbox_inches="tight")
print("Wrote out/data_from_sitk_overlay.png")
# END sitk_figures

# BEGIN example
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from habit.datasets import fetch_demo
from habit.contracts import ArrayImageRef, Cohort, Geometry, Subject
from habit.voxel_features import RawVoxelFeatures

DATA = fetch_demo()
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
# Masks must be integer labels; 0 = background.
subject = Subject(
    subject_id="subj001",
    images={"LAP": ArrayImageRef(array=array, geometry=geometry)},
    masks={"LAP": ArrayImageRef(array=mask, geometry=geometry)},
)
cohort = Cohort([subject], name="from_numpy")
t1 = subject.image("LAP")
field = RawVoxelFeatures(modalities=["LAP"])(subject)
print(
    f"NumPy Subject: id={subject.subject_id}, "
    f"LAP shape={t1.data.shape}, voxels={field.values.shape[0]}"
)
# END example

# BEGIN figures
# Paste after the NumPy block. Uses cohort and t1.
from pathlib import Path

from habit.recipes import one_step_habitat
from habit.viz import plot_habitat_overlay

result = one_step_habitat(
    modalities=("LAP",), n_habitats=3, random_seed=0, roi="LAP"
).fit_predict(cohort)
fig = plot_habitat_overlay(
    t1,
    result.habitat_maps[0],
    title="Habitats from NumPy Subject",
)
Path("out").mkdir(exist_ok=True)
fig.savefig("out/data_from_arrays_overlay.png", dpi=150, bbox_inches="tight")
print("Wrote out/data_from_arrays_overlay.png")
# END figures

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _example_roi import copy_out_figures_to_gallery

    copy_out_figures_to_gallery(
        ("data_from_sitk_overlay.png", "data_from_arrays_overlay.png")
    )
