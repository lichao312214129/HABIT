"""
Load from SimpleITK
===================

**Background.** Many imaging scripts already hold images as SimpleITK
objects. HABIT can wrap those objects directly, keeping their spacing,
origin and direction, so no files have to be rearranged.

**Purpose.** You get a :class:`~habit.contracts.Cohort` built from
SimpleITK images, plus a quick one-step habitat map on one patient to
show the cohort is usable.

**When to use.** Use this when your code already calls
``sitk.ReadImage`` or produces SimpleITK images; if your files follow the
HABIT folder layout, :doc:`/auto_examples/02a_data_in/plot_01_directory`
is shorter.

**Key terms.**

* **cohort / subject / ROI** -- see
  :doc:`/auto_examples/02a_data_in/plot_01_directory`.
* **one-step habitats** -- habitats clustered inside one subject only; see
  :doc:`/auto_examples/01_complete/plot_02_inside_each_subject`.

Read an image file and a mask file with SimpleITK, then put the volumes on
a :class:`~habit.contracts.Subject`. ``modality=`` is the series or ROI
name. One person and several people are each a
:class:`~habit.contracts.Cohort`.
"""

# %%
# One person. Change the two paths to your files.
from pathlib import Path

import matplotlib.pyplot as plt
import SimpleITK as sitk

from habit.contracts import Cohort, ImageVolume, MaskVolume, Subject
from habit.datasets import fetch_demo
from habit.recipes import one_step_habitat
from habit.viz import plot_habitat_overlay, plot_simpleitk_ingest

DATA = fetch_demo()
IMAGE = DATA / "images" / "subj001" / "LAP" / "WATER__WATER__Ax_Dyn_LAVA_Flex+C_Series0009.nrrd"
MASK = DATA / "masks" / "subj001" / "LAP" / "WATER__BH_Ax_LAVA_Flex_10min_Series0017_mask.nrrd"
# from_sitk keeps spacing / origin / direction, so voxel sizes stay physical.
volume = ImageVolume.from_sitk(sitk.ReadImage(str(IMAGE)), modality="LAP")
roi = MaskVolume.from_sitk(sitk.ReadImage(str(MASK)), modality="LAP")
# Dictionary keys are the names later stages use (modalities=..., roi=...).
subject = Subject(
    subject_id="subj001",
    images={"LAP": volume},
    masks={"LAP": roi},
)
one = Cohort([subject], name="one")
print(one)

# %%
# One-step habitats on this one person, then overlay. This only proves the
# cohort works end to end; habitat stages are explained in
# :doc:`/auto_examples/02b_stages/index`.
sitk_result = one_step_habitat(
    modalities=("LAP",), n_habitats=3, random_seed=0, roi="LAP"
).fit_predict(one)
Path("out").mkdir(exist_ok=True)

# Visual summary of this route: SimpleITK image objects in, habitats out
# (right panel zooms to the habitat bounding box).
fig_ingest = plot_simpleitk_ingest(volume, sitk_result.habitat_maps[0])
fig_ingest.savefig("out/data_from_sitk_ingest.png", dpi=150, bbox_inches="tight")
plt.show()

fig_sitk = plot_habitat_overlay(
    volume,
    sitk_result.habitat_maps[0],
    title="Habitats from SimpleITK",
)
fig_sitk.savefig("out/data_from_sitk_overlay.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Several people. Same construction, one Subject per person.
IMAGE_2 = DATA / "images" / "subj002" / "LAP" / "012_WATERWATERAxDynLAVAFlexC.nrrd"
MASK_2 = DATA / "masks" / "subj002" / "LAP" / "016_WATERWATERBHAxLAVAFlex5min_mask.nrrd"
volume_2 = ImageVolume.from_sitk(sitk.ReadImage(str(IMAGE_2)), modality="LAP")
roi_2 = MaskVolume.from_sitk(sitk.ReadImage(str(MASK_2)), modality="LAP")
subject_2 = Subject(
    subject_id="subj002",
    images={"LAP": volume_2},
    masks={"LAP": roi_2},
)
many = Cohort([subject, subject_2], name="many")
print(many)
