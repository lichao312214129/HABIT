"""
Load from SimpleITK
===================

Read NRRD / NIfTI with SimpleITK, then wrap volumes with
:class:`~habit.contracts.ImageVolume` and :class:`~habit.contracts.MaskVolume`.
Spacing, origin, and direction cosines are preserved.
"""

# %%
# Read demo NRRD files and assemble a :class:`~habit.contracts.Subject`.
from pathlib import Path

import matplotlib.pyplot as plt
import SimpleITK as sitk

from habit.contracts import Cohort, ImageVolume, MaskVolume, Subject
from habit.datasets import fetch_demo
from habit.recipes import one_step_habitat
from habit.viz import plot_habitat_overlay

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
volume = ImageVolume.from_sitk(sitk_image, modality="LAP")
roi = MaskVolume.from_sitk(sitk_mask, modality="LAP")
sitk_subject = Subject(
    subject_id="subj001",
    images={"LAP": volume},
    masks={"LAP": roi},
)
sitk_cohort = Cohort([sitk_subject], name="from_sitk")
print(
    f"SimpleITK Subject: id={sitk_subject.subject_id}, "
    f"LAP shape={volume.data.shape}, spacing={volume.geometry.spacing}"
)

# %%
# One-step habitats on the SimpleITK-backed subject, then overlay.
sitk_result = one_step_habitat(
    modalities=MODALITIES, n_habitats=3, random_seed=0, roi="LAP"
).fit_predict(sitk_cohort)
Path("out").mkdir(exist_ok=True)
fig_sitk = plot_habitat_overlay(
    volume,
    sitk_result.habitat_maps[0],
    title="Habitats from SimpleITK",
)
fig_sitk.savefig("out/data_from_sitk_overlay.png", dpi=150, bbox_inches="tight")
plt.show()
