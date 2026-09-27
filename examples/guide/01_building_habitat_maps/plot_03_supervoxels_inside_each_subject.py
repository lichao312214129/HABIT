"""
Supervoxels inside each subject
===============================

**Background.** One-step habitats are fit separately in every patient, so
the habitat ids are not shared. Cutting each tumour into supervoxels
first makes a habitat a group of patches. Those patches can be described
by the mean of the voxel features, or by texture measured on the
original images.

**Purpose.** On all five demo patients you will:

1. cut supervoxels from a voxel table that holds only PVP local GLCM
   (Contrast, JointEntropy), then cluster each patient's supervoxel
   means;
2. in a second analysis, cut from four-phase intensity plus the same
   PVP GLCM, replace the means with per-supervoxel radiomics on the
   four original phases, and cluster those whole-patch texture rows.

Each analysis prints habitat volume fractions and draws every patient's
habitat map. Nothing is pooled. There is no cohort model.

The same steps written as individual components, checked against this
page voxel for voxel, are in
:doc:`plot_03_supervoxels_inside_each_subject_atomic`.

**When to use.** When habitats should describe regions inside one
tumour. If habitat 1 must mean the same thing in every patient, use the
two-step design instead.

**Key terms.**

* **one-step** -- no ``pool`` stage. ``fit`` runs once per subject.
* **partition** -- the supervoxel step. With no ``pool`` after it, the
  supervoxels stay inside the subject that produced them.
* **voxel features (mean path)** -- only PVP local GLCM (Contrast,
  JointEntropy; kernel radius 3, bin width 12). No raw intensities and
  no first-order voxel texture.
* **supervoxel features** -- one row per supervoxel.
  ``mean_voxel_features`` averages the voxel table.
  ``supervoxel_radiomics`` runs PyRadiomics on the whole patch of the
  original images (no kernel): first-order Mean/Variance plus the same
  two GLCM names, once per phase.
"""

# %%
# Load the five demo patients
# ---------------------------
# Change DATA / MODALITIES / ROI to your preprocessed layout.
# sphinx_gallery_thumbnail_number = 1
from pathlib import Path

import matplotlib.pyplot as plt

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.recipes import Study
from habit.spec import HabitatSpec, Spec, Stage
from habit.viz import plot_habitat_overlay

DATA = fetch_demo()
MODALITIES = ("pre_contrast", "LAP", "PVP", "delay_3min")
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)
print(cohort)

Path("out").mkdir(exist_ok=True)

# PVP voxel texture only: two GLCM features in a 7x7x7 neighbourhood.
# Not the bundled full voxel radiomics preset; no first-order block.
voxel_texture_params = {
    "imageType": {"Original": {}},
    "featureClass": {"glcm": ["Contrast", "JointEntropy"]},
    "setting": {"binWidth": 12.0, "normalize": False},
}
# Whole-supervoxel radiomics on the four original phases (no kernel).
region_radiomics_params = {
    "imageType": {"Original": {}},
    "featureClass": {
        "firstorder": ["Mean", "Variance"],
        "glcm": ["Contrast", "JointEntropy"],
    },
    "setting": {"binWidth": 12.0, "normalize": False},
}
# Mean-path extract is inlined below: PVP local GLCM only (no raw).

# %%
# Cluster supervoxel means inside each patient
# --------------------------------------------
# The voxel table is only PVP local GLCM Contrast and JointEntropy
# (7x7x7 neighbourhood, bin width 12). Winsorize and z-score run per
# patient before the cut. ``kmeans`` with 150 supervoxels groups voxels
# by that texture profile. ``mean_voxel_features`` is the row each
# supervoxel sends to the habitat ``kmeans``. ``fit`` has no ``pool``
# before it, so it runs once per patient.
mean_result = Study(
    HabitatSpec(
        name="one_step_supervoxel_means",
        stages=(
            Stage(
                "extract",
                Spec(
                    "voxel_radiomics",
                    {
                        "modality": "PVP",
                        "roi": ROI,
                        "kernel_radius": 3,
                        "params": voxel_texture_params,
                    },
                ),
            ),
            Stage("preprocess", Spec("winsorize", {"winsor_limits": [0.01, 0.01]})),
            Stage("preprocess2", Spec("zscore")),
            Stage("partition", Spec("kmeans", {"n_supervoxels": 150, "n_init": 3})),
            Stage("supervoxel_features", Spec("mean_voxel_features")),
            Stage("fit", Spec("kmeans", {"n_habitats": 3, "n_init": 10})),
            Stage("assign", Spec("nearest_centroid")),
            Stage("quantify", Spec("volume")),
        ),
        random_seed=0,
    )
).fit_predict(cohort)
print("cohort model:", mean_result.habitat_model)
fractions = mean_result.features.frame.set_index("subject")
print(fractions.filter(like="volume_fraction").round(3).to_string())

# %%
# Mean-feature habitat map for every patient
# ------------------------------------------
# The underlay is the arterial phase (LAP). Habitats used only PVP
# local GLCM. Colours are per patient: habitat 1 here is not habitat 1
# in the next patient.
for subject in cohort:
    habitat_map = next(
        item for item in mean_result.habitat_maps if item.subject_id == subject.subject_id
    )
    fig = plot_habitat_overlay(
        subject.image(ROI),
        habitat_map,
        title=f"{subject.subject_id}: habitats on PVP local GLCM means",
        crop_to="labels",
    )
    fig.savefig(
        f"out/one_step_supervoxel_means_{subject.subject_id}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()

# %%
# Cluster supervoxel texture instead
# ----------------------------------
# Separate extract for the cut: four-phase intensity plus PVP GLCM
# (this path is independent of the mean-path texture-only extract).
# Each supervoxel is then a whole-patch PyRadiomics row on the four
# original phases: first-order Mean and Variance, plus GLCM Contrast
# and JointEntropy, once per phase. There is no sliding kernel.
# ``binWidth`` 12 discretizes the original intensities. A patch with a
# single gray level leaves Contrast and JointEntropy undefined.
# ``impute`` fills only those cells with that patient's median.
texture_result = Study(
    HabitatSpec(
        name="one_step_supervoxel_texture",
        stages=(
            Stage(
                "extract",
                Spec(
                    "concat",
                    {
                        "extractors": [
                            {
                                "name": "raw",
                                "params": {"modalities": list(MODALITIES)},
                            },
                            {
                                "name": "voxel_radiomics",
                                "params": {
                                    "modality": "PVP",
                                    "kernel_radius": 3,
                                    "params": voxel_texture_params,
                                },
                            },
                        ],
                        "roi": ROI,
                    },
                ),
            ),
            Stage("preprocess", Spec("winsorize", {"winsor_limits": [0.01, 0.01]})),
            Stage("preprocess2", Spec("zscore")),
            Stage("partition", Spec("kmeans", {"n_supervoxels": 150, "n_init": 3})),
            Stage(
                "supervoxel_features",
                Spec(
                    "supervoxel_radiomics",
                    {"modalities": list(MODALITIES), "params": region_radiomics_params},
                ),
            ),
            Stage("impute_texture", Spec("impute", {"strategy": "median"})),
            Stage("fit", Spec("kmeans", {"n_habitats": 3, "n_init": 10})),
            Stage("assign", Spec("nearest_centroid")),
            Stage("quantify", Spec("volume")),
        ),
        random_seed=0,
    )
).fit_predict(cohort)
print("cohort model:", texture_result.habitat_model)
fractions = texture_result.features.frame.set_index("subject")
print(fractions.filter(like="volume_fraction").round(3).to_string())

for subject in cohort:
    habitat_map = next(
        item
        for item in texture_result.habitat_maps
        if item.subject_id == subject.subject_id
    )
    fig = plot_habitat_overlay(
        subject.image(ROI),
        habitat_map,
        title=f"{subject.subject_id}: habitats on supervoxel texture",
        crop_to="labels",
    )
    fig.savefig(
        f"out/one_step_supervoxel_texture_{subject.subject_id}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()

# %%
# How to read the result
# ----------------------
# * ``habitat_model`` is ``None``. Each patient has its own definition
#   in ``subject_models``.
# * A supervoxel is one habitat: every voxel in that patch gets the same
#   label.
# * The two maps of one patient can disagree. They cluster different
#   columns (means of PVP local GLCM vs whole-patch radiomics).
# * Habitat numbers are private to each patient. Do not line up habitat
#   1 across rows or across figures.
# * Five demo patients and a fixed K of 3 show the workflow, not a
#   clinical subtype.

# %%
# Where to go next
# ----------------
# * The same analysis from individual components:
#   :doc:`plot_03_supervoxels_inside_each_subject_atomic`.
# * The same one-step design on voxels:
#   :doc:`plot_03_inside_each_subject_spec`.
# * Habitat ids shared by every patient:
#   :doc:`plot_01_two_step_spec`.
