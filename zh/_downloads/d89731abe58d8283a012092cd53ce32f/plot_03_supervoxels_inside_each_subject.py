"""
Supervoxels inside each subject
===============================

**Background.** One-step habitats are fit separately in every patient, so
the habitat ids are not shared. The voxel form of that design clusters
every ROI voxel. Cutting the tumour into supervoxels first makes each
habitat a whole patch, and the rows that are clustered can be texture
features instead of the mean intensity.

**Purpose.** On all five demo patients you will:

1. partition each ROI, describe each supervoxel by the mean of the voxel
   features, and cluster those means into habitats;
2. keep the same partition settings, replace the mean with per-supervoxel
   radiomics (first-order and GLCM), and cluster those texture rows.

For each analysis the page prints every patient's habitat volume
fractions and draws every patient's habitat map. Nothing is pooled.
There is no cohort model.

**When to use.** When habitats should describe regions inside one tumour,
and a voxel-wise map is too speckled, or when the scientific question is
about texture rather than mean intensity. If habitat 1 must mean the
same thing in every patient, use the two-step design instead.

**Key terms.**

* **one-step** -- no ``pool`` stage. ``fit`` runs once per subject.
* **partition** -- the supervoxel step. With no ``pool`` after it, those
  supervoxels stay inside the subject that produced them.
* **supervoxel features** -- one row per supervoxel. ``mean_voxel_features``
  averages the voxel table. ``supervoxel_radiomics`` runs PyRadiomics on
  the original image inside each supervoxel mask. Omit the stage to keep
  the means the partitioner already attached.

The voxel-only form of this design is
:doc:`/auto_examples/01_building_habitat_maps/plot_03_inside_each_subject_spec`.
"""

# %%
# Load the five demo patients
# ---------------------------
# Change DATA / MODALITIES / ROI to your preprocessed layout. One phase
# keeps this page short; add names to ``MODALITIES`` to cluster on more
# series. Radiomics then runs once per named modality.
# sphinx_gallery_thumbnail_number = 1
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.recipes import Study
from habit.recipes.result import StudyResult
from habit.spec import HabitatSpec, Spec, Stage
from habit.viz import plot_habitat_overlay

DATA = fetch_demo()
MODALITIES = ("LAP",)
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)
print(cohort)

Path("out").mkdir(exist_ok=True)

# Shared by both analyses below. More than 100 supervoxels per tumour, so
# a habitat is a group of small patches rather than a few large blocks.
# Fixed K keeps the demo on the partition and the feature stage; swap in
# min_habitats / max_habitats and validation="elbow" when you want each
# patient to choose its own K.
_PARTITION = Spec("kmeans", {"n_supervoxels": 150, "n_init": 3})
_FIT = Spec("kmeans", {"n_habitats": 3, "n_init": 10})


def volume_fraction_table(result: StudyResult) -> pd.DataFrame:
    """Return each subject's own habitat volume fractions.

    Args:
        result: One-step study result. ``result.features`` is the volume
            table written by ``Stage("quantify", Spec("volume"))``.

    Returns:
        One row per subject. Columns are ``habitat_<id>_volume_fraction``.
        The same column in two rows is not the same kind of tissue: each
        subject was clustered on its own.
    """
    frame = result.features.frame.set_index("subject")
    fraction_columns = sorted(
        (column for column in frame.columns if str(column).endswith("_volume_fraction")),
        key=lambda name: int(str(name).split("_")[1]),
    )
    table = frame.loc[:, fraction_columns].copy()
    table.index.name = "subject"
    return table


def show_every_habitat_map(result: StudyResult, *, title_suffix: str, file_prefix: str) -> None:
    """Draw one habitat overlay for every subject in ``cohort``.

    Args:
        result: Study result whose ``habitat_maps`` cover ``cohort``.
        title_suffix: English phrase after the subject id in the title.
        file_prefix: Stem used under ``out/``; the subject id is appended.
    """
    maps_by_id = {habitat_map.subject_id: habitat_map for habitat_map in result.habitat_maps}
    for subject in cohort:
        habitat_map = maps_by_id[subject.subject_id]
        # ImageVolume + HabitatMap: the overlay follows the label geometry.
        fig = plot_habitat_overlay(
            subject.image(ROI),
            habitat_map,
            title=f"{subject.subject_id}: {title_suffix}",
            crop_to="labels",
        )
        fig.savefig(
            f"out/{file_prefix}_{habitat_map.subject_id}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.show()


# %%
# Cluster supervoxel means inside each patient
# --------------------------------------------
# ``partition`` then ``fit``, and no ``pool``. The first ``kmeans`` cuts
# supervoxels. ``mean_voxel_features`` is the row each supervoxel sends
# to the second ``kmeans``, which fits habitats for that patient only.
# Assignment paints every voxel of a supervoxel with that habitat id.
mean_spec = HabitatSpec(
    name="one_step_supervoxel_means",
    stages=(
        Stage("extract", Spec("raw", {"modalities": list(MODALITIES), "roi": ROI})),
        Stage("preprocess", Spec("winsorize", {"winsor_limits": [0.01, 0.01]})),
        Stage("preprocess2", Spec("zscore")),
        Stage("partition", _PARTITION),
        Stage("supervoxel_features", Spec("mean_voxel_features")),
        Stage("fit", _FIT),
        Stage("assign", Spec("nearest_centroid")),
        Stage("quantify", Spec("volume")),
    ),
    random_seed=0,
)
mean_result = Study(spec=mean_spec).fit_predict(cohort)
mean_model = next(iter(mean_result.subject_models.values()))
print(
    f"mean features: subjects={len(mean_result.subject_models)}, "
    f"columns per supervoxel={mean_model.centroids.shape[1]}, "
    f"cohort model={mean_result.habitat_model}"
)
print("Habitat volume fractions (ids are private to each row):")
print(volume_fraction_table(mean_result).round(3).to_string())

# %%
# Mean-feature habitat map for every patient
# ------------------------------------------
# Colours are per patient. Habitat 1 in subj001 is not habitat 1 in
# subj002. Each patch of one colour is one or more supervoxels that the
# per-patient model assigned to the same habitat.
show_every_habitat_map(
    mean_result,
    title_suffix="habitats on supervoxel means",
    file_prefix="one_step_supervoxel_means",
)

# %%
# Cluster supervoxel texture instead
# ----------------------------------
# Same patients, same partition, same per-subject fit. The feature stage
# is now PyRadiomics on the original intensities inside each supervoxel,
# not the mean of the preprocessed voxel table. ``binWidth`` discretizes
# the GLCM. First-order Mean and Variance sit next to two GLCM features
# so the habitat model can separate patches by texture as well as level.
# A patch smaller than the GLCM neighbourhood, or one with a single gray
# level, leaves Contrast and JointEntropy undefined. ``impute`` replaces
# only those cells with that patient's median of the finite values, so
# k-means still receives a complete row per supervoxel.
texture_params = {
    "imageType": {"Original": {}},
    "featureClass": {
        "firstorder": ["Mean", "Variance"],
        "glcm": ["Contrast", "JointEntropy"],
    },
    "setting": {"binWidth": 25.0, "normalize": False},
}
texture_spec = HabitatSpec(
    name="one_step_supervoxel_texture",
    stages=(
        Stage("extract", Spec("raw", {"modalities": list(MODALITIES), "roi": ROI})),
        Stage("preprocess", Spec("winsorize", {"winsor_limits": [0.01, 0.01]})),
        Stage("preprocess2", Spec("zscore")),
        Stage("partition", _PARTITION),
        Stage(
            "supervoxel_features",
            Spec(
                "supervoxel_radiomics",
                {"modality": ROI, "params": texture_params},
            ),
        ),
        # After the texture columns exist. Undefined GLCM cells become that
        # patient's median; defined cells are left unchanged.
        Stage("impute_texture", Spec("impute", {"strategy": "median"})),
        Stage("fit", _FIT),
        Stage("assign", Spec("nearest_centroid")),
        Stage("quantify", Spec("volume")),
    ),
    random_seed=0,
)
texture_result = Study(spec=texture_spec).fit_predict(cohort)
texture_model = next(iter(texture_result.subject_models.values()))
print(
    f"texture features: subjects={len(texture_result.subject_models)}, "
    f"columns per supervoxel={texture_model.centroids.shape[1]}, "
    f"cohort model={texture_result.habitat_model}"
)
print("Habitat volume fractions (ids are private to each row):")
print(volume_fraction_table(texture_result).round(3).to_string())

# %%
# Texture habitat map for every patient
# -------------------------------------
# Read each figure against that patient's row in the table above. A
# colour that dominates the map is the habitat with the large volume
# fraction in that row only.
show_every_habitat_map(
    texture_result,
    title_suffix="habitats on supervoxel texture",
    file_prefix="one_step_supervoxel_texture",
)

# %%
# How to read the result
# ----------------------
# * ``habitat_model`` is ``None``. Each of the five patients has its own
#   definition in ``subject_models``.
# * A supervoxel is one habitat: the assigner looks up the supervoxel id
#   and paints every voxel in that patch with the same label.
# * The two maps of one patient can disagree. They cluster different
#   columns (means of the preprocessed intensities, versus PyRadiomics on
#   the original image).
# * Habitat numbers are private to each patient. Do not line up habitat 1
#   across rows of the volume table, or across patients in the figures.
# * Five demo patients and a fixed K of 3: the figures show the workflow,
#   not a clinical subtype.

# %%
# Where to go next
# ----------------
# * The same one-step design on voxels, with the elbow rule:
#   :doc:`/auto_examples/01_building_habitat_maps/plot_03_inside_each_subject_spec`.
# * Supervoxel method, count, and a radiomics-versus-mean benchmark:
#   :doc:`/auto_examples/03_clustering/plot_01_supervoxel_method_and_count`
#   and :doc:`/auto_examples/03_clustering/plot_02_what_supervoxel_summarizes`.
# * Habitat ids shared by every patient:
#   :doc:`/auto_examples/01_building_habitat_maps/plot_01_two_step_spec`.
