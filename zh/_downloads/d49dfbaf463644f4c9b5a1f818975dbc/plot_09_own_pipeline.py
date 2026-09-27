"""
HABIT inside your own pipeline
==============================

**Background.** Most imaging projects already have their own loader: a
PACS export script, a MONAI dataset, a segmentation network that
returns a NumPy mask. HABIT does not need its folder layout to run.
Its analysis objects take images that are already in memory, and they
give back a label array with the geometry of the input image, which you
can turn into whatever image type your pipeline uses.

**Purpose.** You will read the demo images yourself with SimpleITK (as
your own loader would), keep the masks as plain NumPy arrays (as a
segmentation network would hand them over), build the subjects and the
cohort in memory, run the approved two-step ``Spec`` with ``Study``,
and hand each habitat map back as a SimpleITK image on the input grid.
A last check runs the same spec on the same patients loaded with
``cohort_from_directory`` and prints whether the maps are identical.

**When to use.** HABIT is one step of a larger pipeline and your images
come from your own code, not from a ``DATA/images/<subject>/<modality>``
folder. If your files already follow that folder layout,
:doc:`/auto_examples/06_manipulating_images/plot_01_directory` is shorter.

**Analysis definition.** The approved two-step definition, unchanged
(four DCE phases, ROI ``LAP``, fitted on subj001 + subj002, seed 0).
Only the way the images reach HABIT changes.

**Key terms.** Full definitions are on :doc:`/tutorial/concepts`.

* **geometry** -- spacing, origin and direction that place an array in
  physical space. An array alone has none; it must be supplied with it.
* **ImageVolume / MaskVolume** -- an in-memory image or mask array plus
  its geometry; built here from a SimpleITK image and from a NumPy
  array.
* **Subject / Cohort** -- one patient's named images and masks, and a
  list of subjects; both live in memory only.
* **HabitatMap** -- one subject's habitat label array (``0`` =
  background) plus the geometry it refers to.
"""

# %%
# Your own loader
# ---------------
# Stand-in for the loader your project already has. It reads files with
# SimpleITK and returns SimpleITK images for the phases and the mask as
# a NumPy array with its geometry, the way a segmentation step often
# does. Nothing below depends on where these files live; replace the
# body with your own code.
# sphinx_gallery_thumbnail_number = 1
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from habit.contracts import Cohort, ImageVolume, MaskVolume, Subject, cohort_from_directory
from habit.datasets import fetch_demo
from habit.execution import backend_from_policy
from habit.recipes import Study
from habit.spec import HabitatSpec, Spec, Stage
from habit.spec.policy import RunPolicy
from habit.viz import plot_habitat_overlay

# Change DATA / MODALITIES / ROI to your preprocessed layout.
DATA = fetch_demo()
MODALITIES = ("pre_contrast", "LAP", "PVP", "delay_3min")
ROI = "LAP"
SUBJECT_IDS = ("subj001", "subj002")
Path("out").mkdir(exist_ok=True)


def my_loader(subject_id: str) -> Tuple[Dict[str, sitk.Image], np.ndarray, sitk.Image]:
    """
    Load one patient the way an existing project might.

    Args:
        subject_id: Patient identifier.

    Returns:
        ``(images, mask_array, mask_reference)``: the DCE phases as
        SimpleITK images keyed by phase name, the ROI as an integer NumPy
        array in ``(z, y, x)`` order, and the SimpleITK image that array
        came from (only used for its geometry).
    """
    images = {
        phase: sitk.ReadImage(str(next((DATA / "images" / subject_id / phase).glob("*.nrrd"))))
        for phase in MODALITIES
    }
    mask_reference = sitk.ReadImage(str(next((DATA / "masks" / subject_id / ROI).glob("*.nrrd"))))
    # GetArrayFromImage returns (z, y, x); masks must be integer labels.
    mask_array = sitk.GetArrayFromImage(mask_reference).astype(np.int32)
    return images, mask_array, mask_reference


# %%
# Build subjects and the cohort in memory
# ---------------------------------------
# Images: ``ImageVolume.from_sitk`` keeps each image's spacing, origin
# and direction. Mask: ``MaskVolume.from_array`` needs the geometry to be
# passed explicitly, because a NumPy array has none; pass the geometry
# your segmentation step reports for it. In the demo the mask headers
# have a different origin and z direction from their images while the
# arrays have the same shape; HABIT then keeps the mask voxels as they
# are, index for index, and puts them on the image grid (it logs the
# mismatch). A mask of a different shape is never resampled silently:
# it raises an error that tells you to regrid it first.
subjects = []
input_images: Dict[str, sitk.Image] = {}
input_masks: Dict[str, np.ndarray] = {}
for subject_id in SUBJECT_IDS:
    images, mask_array, mask_reference = my_loader(subject_id)
    subjects.append(
        Subject(
            subject_id=subject_id,
            # The keys are the names later stages use (modalities=, roi=).
            images={phase: ImageVolume.from_sitk(image, modality=phase) for phase, image in images.items()},
            masks={
                ROI: MaskVolume.from_array(
                    mask_array,
                    spacing=tuple(mask_reference.GetSpacing()),
                    origin=tuple(mask_reference.GetOrigin()),
                    direction=tuple(mask_reference.GetDirection()),
                    modality=ROI,
                )
            },
        )
    )
    # Keep the LAP image (the result is handed back on its geometry) and
    # the mask array (for the check at the end).
    input_images[subject_id] = images["LAP"]
    input_masks[subject_id] = mask_array
cohort = Cohort(subjects, name="my_pipeline")
print(cohort)

# %%
# Run the approved spec
# ---------------------
# The same ``HabitatSpec`` as
# :doc:`/auto_examples/01_building_habitat_maps/plot_01_two_step_spec`. ``Study`` only
# sees the in-memory cohort; it neither reads nor writes any folder.
spec = HabitatSpec(
    name="own_pipeline_two_step",
    stages=(
        # extract: one raw intensity column per DCE phase inside the ROI.
        Stage("extract", Spec("raw", {"modalities": list(MODALITIES), "roi": ROI})),
        # subject-level: clip to each patient's 1st..99th percentile, then 0..1.
        Stage("preprocess", Spec("winsorize", {"winsor_limits": [0.01, 0.01]})),
        Stage("preprocess2", Spec("zscore")),
        # partition: 30 supervoxels per tumour; pool: stack both patients.
        Stage("partition", Spec("kmeans", {"n_supervoxels": 30})),
        Stage("pool", Spec("pool")),
        # cohort-level: 10 equal-width bins learned on the pooled supervoxels.

        # fit: k-means with 2..10 habitats, elbow rule; assign: nearest centroid.
        Stage("fit", Spec("kmeans", {"min_habitats": 2, "max_habitats": 10, "validation": "elbow", "n_init": 10})),
        Stage("assign", Spec("nearest_centroid")),
        # quantify: one row per subject.
        Stage("volume", Spec("volume")),
        Stage("msi", Spec("msi")),
        Stage("ith", Spec("ith_score")),
        Stage("graph", Spec("graph", {"include_extended_metrics": False})),
    ),
    random_seed=0,
)
result = Study(spec).fit_predict(
    cohort,
    backend=backend_from_policy(
        RunPolicy(backend="serial", on_subject_failure="continue")
    ),
)
print("habitats:", result.habitat_model.n_habitats)

# %%
# Elbow / Kneedle caveat
# ----------------------
# HABIT validation ``elbow`` is the same rule as ``kneedle`` (Satopaa et
# al., 2011): KneeLocator on inertia, curve convex, direction decreasing —
# not Thorndike (1953) informal elbow, not pre-v1.0 discrete-curvature.
# Maps keep the Kneedle K. See :doc:`/user_guide/building_habitat_maps`.
_report = result.habitat_model.preprocessing_state["selection_report"]
_cand = [int(k) for k in _report["candidates"]]
_iner = np.asarray(_report["scores"][_report["methods"][0]], dtype=float)
_k_kn = int(result.habitat_model.n_habitats)
_k_dc = int(_cand[int(np.argmax(np.diff(_iner, n=2))) + 1]) if len(_iner) >= 3 else _k_kn
from habit.habitat_model import KMeansHabitatModelFitter as _KF

_k_table = {"elbow_kneedle": _k_kn, "discrete_curvature_elbow": _k_dc}
for _name in ("silhouette", "calinski_harabasz", "davies_bouldin"):
    _f = _KF(min_habitats=2, max_habitats=10, validation=_name, n_init=10)
    _f.set_random_state(0)
    _k_table[_name] = _f.fit(result.units, cohort=cohort).n_habitats
print("K by criterion (skip gap):")
for _n, _k in _k_table.items():
    print(f"  {_n}: {_k}")

for habitat_map in result.habitat_maps:
    print(habitat_map.subject_id, "label array", habitat_map.label_array.shape, "ids", habitat_map.habitat_ids)

subject = cohort[0]
fig = plot_habitat_overlay(
    subject.image("LAP"),
    result.habitat_maps[0],
    title=f"{subject.subject_id}: habitats from in-memory inputs",
    crop_to="labels",
)
fig.savefig("out/own_pipeline_habitats.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Hand the result back as SimpleITK images
# ----------------------------------------
# ``label_array`` is a NumPy array in ``(z, y, x)`` order, indexed like
# the input image array: voxel ``[z, y, x]`` of the map is voxel
# ``[z, y, x]`` of the LAP image. ``GetImageFromArray`` turns it into a
# SimpleITK image with default geometry; ``CopyInformation`` then gives
# it the spacing, origin and direction of the input LAP image (it
# raises if the sizes differ), so the map overlays that image in any
# viewer.
#
# Two checks print True or False: the label image has the size of the
# input image, and its labelled voxels are exactly the ROI voxels of the
# mask array your loader returned (every ROI voxel got a habitat, no
# voxel outside it did).
for habitat_map in result.habitat_maps:
    reference = input_images[habitat_map.subject_id]
    label_image = sitk.GetImageFromArray(np.asarray(habitat_map.label_array).astype(np.uint8))
    label_image.CopyInformation(reference)
    labelled = sitk.GetArrayFromImage(label_image) > 0
    roi = input_masks[habitat_map.subject_id] > 0
    print(
        f"{habitat_map.subject_id}: same size as input = {label_image.GetSize() == reference.GetSize()}, "
        f"labelled voxels == ROI voxels = {np.array_equal(labelled, roi)}"
    )
    # From here on it is your pipeline's image; writing it is optional.
    sitk.WriteImage(label_image, f"out/{habitat_map.subject_id}_habitats.nrrd")

# %%
# Same result as the directory-loaded run
# ---------------------------------------
# The same patients loaded with ``cohort_from_directory`` and the same
# spec. The in-memory route should change nothing but the loading, so
# the habitat maps must be identical voxel for voxel.
directory_cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[: len(SUBJECT_IDS)]
directory_result = Study(spec).fit_predict(
    directory_cohort,
    backend=backend_from_policy(
        RunPolicy(backend="serial", on_subject_failure="continue")
    ),
)
for own_map, dir_map in zip(result.habitat_maps, directory_result.habitat_maps):
    same = own_map.subject_id == dir_map.subject_id and np.array_equal(own_map.label_array, dir_map.label_array)
    print(f"{own_map.subject_id}: in-memory == directory-loaded: {same}")
print("same number of habitats:", result.habitat_model.n_habitats == directory_result.habitat_model.n_habitats)

# %%
# How to read the result
# ----------------------
# When this page was built:
#
# * The in-memory cohort of subj001 + subj002 gave K = 4 habitats, and
#   each HabitatMap was a ``(200, 360, 360)`` label array with habitat
#   ids 1..4.
# * Both hand-back checks printed True for both patients: the SimpleITK
#   label image has the input LAP image's size, and exactly the ROI
#   voxels of the loader's mask array carry a habitat.
# * The directory-loaded run gave identical maps (``True`` for both
#   patients) and the same number of habitats, so building the subjects
#   yourself changes only how the data arrive, not the result.
#
# Apart from the final cross-check, HABIT itself never read a folder:
# only ``my_loader`` touched files (here the demo files, in your project
# whatever your loader reads), and the ``.nrrd`` written to ``out/`` is
# just one way your pipeline could store the returned SimpleITK images.

# %%
# Where to go next
# ----------------
# * Loading from SimpleITK and from NumPy on their own:
#   :doc:`/auto_examples/06_manipulating_images/plot_03_simpleitk` and
#   :doc:`/auto_examples/06_manipulating_images/plot_04_numpy`.
# * Calling each analysis step yourself instead of ``Study``:
#   :doc:`/auto_examples/01_building_habitat_maps/plot_02_atomic_two_step`.
# * Saving the model and labelling new patients from your own loader:
#   :doc:`/auto_examples/05_validation_and_reuse/plot_01_train_save_predict`.
