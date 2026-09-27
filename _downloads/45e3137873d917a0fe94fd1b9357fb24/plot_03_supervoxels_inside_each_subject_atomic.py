"""
Supervoxels inside each subject, from components
=================================================

**Background.** :doc:`plot_03_supervoxels_inside_each_subject` declares
this analysis as a :class:`~habit.spec.HabitatSpec`. The same objects
exist one step at a time: a voxel extractor, a subject-level
normaliser, a supervoxelizer, a supervoxel-feature extractor, a
habitat fitter and an assigner.

**Purpose.** On the same five demo patients, build both habitat maps
without ``Study``, then run ``Study`` and check that every voxel label
matches. The mean-path voxel table is only PVP local GLCM Contrast and
JointEntropy (kernel radius 3, bin width 12). The region-radiomics path
still cuts from four-phase intensity plus that PVP GLCM, then measures
whole-patch PyRadiomics on the four original phases (no kernel).

**When to use.** When one of these steps has to sit inside your own
loop. For a normal study, ``Study`` is shorter and records the manifest.
"""

# %%
# Load the five demo patients
# ---------------------------
# Change DATA / MODALITIES / ROI to your preprocessed layout.
# sphinx_gallery_thumbnail_number = 1
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.feature_preprocessing import Impute, SubjectPreprocessingChain, Winsorizing, ZScoreScaling
from habit.habitat_features import HabitatVolumeFeatures
from habit.habitat_model import KMeansHabitatModelFitter
from habit.recipes import Study
from habit.spec import HabitatSpec, Spec, Stage
from habit.supervoxel import KMeansSupervoxelizer, SupervoxelRadiomicsFeatures
from habit.viz import plot_habitat_overlay
from habit.voxel_features import ConcatVoxelFeatures, VoxelRadiomicsFeatures

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
# Mean path: PVP local GLCM only (same params dict as the Study Spec).
extractor = VoxelRadiomicsFeatures(
    modality="PVP",
    roi=ROI,
    kernel_radius=3,
    params=voxel_texture_params,
)
# Region-path cut only: four-phase intensity plus the same PVP GLCM.
region_cut_extractor = ConcatVoxelFeatures(
    extractors=[
        {"name": "raw", "params": {"modalities": list(MODALITIES)}},
        {
            "name": "voxel_radiomics",
            "params": {
                "modality": "PVP",
                "kernel_radius": 3,
                "params": voxel_texture_params,
            },
        },
    ],
    roi=ROI,
)
# Per patient, before the cut. Imputation is inserted at the front of
# the chain so a non-finite voxel cannot reach k-means.
voxel_chain = SubjectPreprocessingChain(
    [Winsorizing(winsor_limits=(0.01, 0.01)), ZScoreScaling()]
)
# 150 supervoxels from the PVP local GLCM profile (mean path).
# k-means here does not use voxel coordinates.
partitioner = KMeansSupervoxelizer(n_supervoxels=150, n_init=3)
partitioner.set_random_state(0)
# Fixed K. A new fitter per patient keeps one patient's centres from
# leaking into the next call.
volume = HabitatVolumeFeatures()


def habitat_fitter() -> KMeansHabitatModelFitter:
    """Return a seeded 3-habitat k-means fitter.

    Returns:
        Fitter with ``n_habitats=3`` and ``n_init=10``. The seed matches
        ``HabitatSpec(random_seed=0)`` on the Study page.
    """
    fitter = KMeansHabitatModelFitter(n_habitats=3, n_init=10)
    fitter.set_random_state(0)
    return fitter


# %%
# Means: one patient at a time
# ----------------------------
# The supervoxelizer stores the mean of the normalized PVP GLCM columns
# on each supervoxel. That table is what the habitat model clusters.
# There is no pool and no second patient in ``fit``.
mean_maps = []
mean_rows = []
for subject in cohort:
    field = extractor(subject)
    print(
        subject.subject_id,
        "voxel columns",
        field.values.shape[1],
    )
    field = field.with_feature_frame(
        voxel_chain(field.feature_frame()),
        produced_by="feature_preprocessing.subject.voxel",
        spec_fingerprint=voxel_chain.spec.fingerprint(),
    )
    units = partitioner(field)
    model = habitat_fitter().fit([units])
    habitat_map = model.assigner("nearest_centroid")(units)
    mean_maps.append(habitat_map)
    mean_rows.append(volume(subject, habitat_map).frame)
    print(subject.subject_id, "supervoxels", len(units.features), "K", model.n_habitats)

# %%
# Means: check against Study
# --------------------------
mean_spec = HabitatSpec(
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
mean_study = Study(mean_spec).fit_predict(cohort)
for habitat_map in mean_maps:
    study_map = next(
        item for item in mean_study.habitat_maps if item.subject_id == habitat_map.subject_id
    )
    print(
        habitat_map.subject_id,
        "labels identical =",
        np.array_equal(habitat_map.label_array, study_map.label_array),
    )
fractions = mean_study.features.frame.set_index("subject")
print(fractions.filter(like="volume_fraction").round(3).to_string())

# %%
# Mean-feature habitat map for every patient
# ------------------------------------------
for subject, habitat_map in zip(cohort, mean_maps):
    fig = plot_habitat_overlay(
        subject.image(ROI),
        habitat_map,
        title=f"{subject.subject_id}: habitats on PVP local GLCM means",
        crop_to="labels",
    )
    fig.savefig(
        f"out/atomic_supervoxel_means_{subject.subject_id}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()

# %%
# Texture: one patient at a time
# ------------------------------
# Radiomics reads the original images inside each supervoxel, one row
# per phase. The mask is the whole supervoxel, so there is no kernel.
# bin width is 12. Median imputation replaces only undefined GLCM cells.
# The cut for this path uses intensity + PVP GLCM (not the mean-path
# texture-only table).
texture = SupervoxelRadiomicsFeatures(
    modalities=list(MODALITIES),
    params=region_radiomics_params,
)
texture_chain = SubjectPreprocessingChain([Impute(strategy="median")])
texture_maps = []
for subject in cohort:
    field = region_cut_extractor(subject)
    field = field.with_feature_frame(
        voxel_chain(field.feature_frame()),
        produced_by="feature_preprocessing.subject.voxel",
        spec_fingerprint=voxel_chain.spec.fingerprint(),
    )
    units = texture(
        subject,
        partitioner(field),
    )
    units = units.with_feature_frame(
        texture_chain(units.feature_frame()),
        produced_by="feature_preprocessing.subject.supervoxel",
        spec_fingerprint=texture_chain.spec.fingerprint(),
    )
    model = habitat_fitter().fit([units])
    habitat_map = model.assigner("nearest_centroid")(units)
    texture_maps.append(habitat_map)
    print(
        subject.subject_id,
        "region radiomics columns",
        units.features.shape[1],
        "K",
        model.n_habitats,
    )

texture_spec = HabitatSpec(
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
texture_study = Study(texture_spec).fit_predict(cohort)
for habitat_map in texture_maps:
    study_map = next(
        item
        for item in texture_study.habitat_maps
        if item.subject_id == habitat_map.subject_id
    )
    print(
        habitat_map.subject_id,
        "labels identical =",
        np.array_equal(habitat_map.label_array, study_map.label_array),
    )
fractions = texture_study.features.frame.set_index("subject")
print(fractions.filter(like="volume_fraction").round(3).to_string())

for subject, habitat_map in zip(cohort, texture_maps):
    fig = plot_habitat_overlay(
        subject.image(ROI),
        habitat_map,
        title=f"{subject.subject_id}: habitats on supervoxel texture",
        crop_to="labels",
    )
    fig.savefig(
        f"out/atomic_supervoxel_texture_{subject.subject_id}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()

# %%
# Where to go next
# ----------------
# * The same analysis as one specification:
#   :doc:`plot_03_supervoxels_inside_each_subject`.
# * Voxel one-step from components:
#   :doc:`plot_04_atomic_inside_each_subject`.
