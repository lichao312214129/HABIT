"""
Subject z-score on voxel texture
=================================

Per-subject z-score uses each lesion's own mean and standard deviation.
The atomic calls and the :class:`~habit.spec.HabitatSpec` below are the
same analysis: four DCE series, GLCM / first-order / GLRLM texture,
radius 3, ``binWidth`` 12, then habitats fit on the pooled voxels.
"""

# %%
# Load the cohort
# ---------------
# Four series, arterial-phase ROI. ``print(cohort)`` lists the ids.
# sphinx_gallery_thumbnail_number = 3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.execution import SerialBackend
from habit.feature_preprocessing import SubjectPreprocessingChain, ZScoreScaling
from habit.habitat_model import KMeansHabitatModelFitter
from habit.pipeline import voxel_units
from habit.viz import plot_habitat_overlay, plot_voxel_texture_slice
from habit.voxel_features import VoxelRadiomicsFeatures

DATA = fetch_demo()
MODALITIES = ("pre_contrast", "LAP", "PVP", "delay_3min")
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:2]
print(cohort)
Path("out").mkdir(exist_ok=True)
# A hit skips PyRadiomics. voxel_batch and the GPU knobs are not part of the key.
CACHE = str((Path("out") / "voxel_texture_cache").resolve())

# %%
# Extract voxel texture
# ---------------------
# Radius 3 is a 7×7×7 neighbourhood. ``binWidth`` is the PyRadiomics
# grey-level width. Each feature is repeated on every series, so a column
# looks like ``original_glcm_Contrast-LAP``.
#
# The four runtime arguments are set explicitly. They do not change the
# feature definition. ``voxel_batch="auto"`` sizes the batch from VRAM;
# ``-1`` sends the whole ROI in one batch. ``torch_device="cpu"`` with
# ``use_torch_radiomics=False`` and ``use_gpu_matrices=False`` is the CPU path.
RADIOMICS_PARAMS = {
    "imageType": {"Original": {}},
    "featureClass": {
        "firstorder": ["Mean", "Entropy"],
        "glcm": ["Contrast", "Correlation", "Idm", "JointEntropy"],
        "glrlm": ["ShortRunEmphasis", "LongRunEmphasis"],
    },
    "setting": {"binWidth": 12},
}
texture = VoxelRadiomicsFeatures(
    modalities=list(MODALITIES),
    roi=ROI,
    kernel_radius=3,
    params=RADIOMICS_PARAMS,
    voxel_batch=1000,
    use_torch_radiomics=True,
    torch_device="cuda:0",
    use_gpu_matrices=True,
    cache_dir=CACHE,
)
backend = SerialBackend()
fields = [slot.result() for slot in backend.map(texture, cohort)]
for field in fields:
    print(
        f"{field.subject_id}: {field.values.shape[0]} voxels, "
        f"{len(field.feature_names)} columns"
    )
print(list(fields[0].feature_names))
print(fields[0].feature_frame().head())

column = next(
    name
    for name in fields[0].feature_names
    if "Contrast" in name and name.endswith("-LAP")
)
fig_texture = plot_voxel_texture_slice(
    fields[0],
    feature=column,
    anatomy=cohort[0].image(ROI),
    roi_mask=cohort[0].mask(ROI),
    title=column,
    crop_to="roi",
)
fig_texture.savefig("out/subject_zscore_texture.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Z-score inside each subject
# ---------------------------
# Impute is inserted in front of z-score. A finished column sits near
# mean 0 and standard deviation 1, using that subject only.
zscore = SubjectPreprocessingChain([ZScoreScaling(across_features=False)])
print("preprocess:", [method.spec.name for method in zscore.methods])

scaled_fields = []
for field in fields:
    raw = field.feature_frame()
    scaled = zscore(raw)
    stats = scaled.agg(["mean", "std"]).T.round(3)
    print(field.subject_id)
    print(stats.head(8))
    scaled_fields.append(
        field.with_feature_frame(
            scaled,
            produced_by="subject_feature_preprocessor",
            spec_fingerprint=zscore.spec.fingerprint(),
        )
    )

fig_hist, axes = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
axes[0].hist(fields[0].feature_frame()[column].to_numpy(), bins=30)
axes[0].set_title(f"{column} before")
axes[1].hist(scaled_fields[0].feature_frame()[column].to_numpy(), bins=30)
axes[1].set_title(f"{column} after subject z-score")
fig_hist.savefig("out/subject_zscore_hist.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Fit on the pooled voxels
# ------------------------
# Every ROI voxel is one unit. Fitting stays in this process: it has to
# see the whole cohort. The card is the habitat definition.
fitter = KMeansHabitatModelFitter(n_habitats=3, n_init=3)
fitter.set_random_state(0)
units = [voxel_units(field) for field in scaled_fields]
for one in units:
    print(f"{one.subject_id}: {len(one.features)} voxel units")
model = fitter.fit(units, cohort=cohort)
print(model.summary())
print("centroids:", model.centroids.shape)

# %%
# Assign habitats
# ---------------
# Assignment reuses these units. It does not extract texture again.
# Label 0 is background.
maps = [slot.result() for slot in backend.map(model.assigner(), units)]
for habitat_map in maps:
    labels, counts = np.unique(habitat_map.label_array, return_counts=True)
    present = {
        int(label): int(count)
        for label, count in zip(labels, counts)
        if int(label) != 0
    }
    print(habitat_map.subject_id, "voxels per habitat:", present)

fig_map = plot_habitat_overlay(
    cohort[0].image(ROI),
    maps[0],
    title="habitats (texture, subject z-score)",
    crop_to="labels",
)
fig_map.savefig("out/subject_zscore_habitats.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Same analysis as a HabitatSpec
# ------------------------------
# ``stages`` builds the same extractor, the same subject z-score, and the
# same fitter. ``pool`` with no partition is the pooled-voxel fit above.
# ``random_seed=0`` is ``set_random_state(0)``.
from habit.recipes import Study
from habit.spec import HabitatSpec, Spec, Stage

spec = HabitatSpec(
    name="texture_subject_zscore",
    stages=(
        Stage(
            "extract_voxel_features",
            Spec(
                "voxel_radiomics",
                {
                    "modalities": list(MODALITIES),
                    "roi": ROI,
                    "kernel_radius": 3,
                    "params": RADIOMICS_PARAMS,
                    "voxel_batch": 1000,
                    "use_torch_radiomics": True,
                    "torch_device": "cuda:0",
                    "use_gpu_matrices": True,
                    "cache_dir": CACHE,
                },
            ),
            role="extract_voxel_features",
        ),
        Stage(
            "preprocess_voxels",
            Spec("zscore", {"across_features": False}),
            role="preprocess",
        ),
        Stage("pool", Spec("pool"), role="pool"),
        Stage(
            "fit",
            Spec("kmeans", {"n_habitats": 3, "n_init": 3}),
            role="fit",
        ),
        Stage("assign", Spec("nearest_centroid"), role="assign"),
    ),
    random_seed=0,
)
print("stages:", [(stage.name, stage.role) for stage in spec.stages])

# %%
# Run that spec
# -------------
# ``Study.fit_predict`` builds the same chain. The voxel counts match
# the atomic assignment above.
result = Study(spec).fit_predict(cohort, backend=backend)
print(result.habitat_model.summary())
for habitat_map in result.habitat_maps:
    labels, counts = np.unique(habitat_map.label_array, return_counts=True)
    present = {
        int(label): int(count)
        for label, count in zip(labels, counts)
        if int(label) != 0
    }
    print(habitat_map.subject_id, "voxels per habitat:", present)
