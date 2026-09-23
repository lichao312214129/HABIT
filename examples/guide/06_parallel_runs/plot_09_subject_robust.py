"""
Subject robust scaling on voxel texture
=======================================

Robust scaling centres each column on that subject's median and divides
by its interquartile range. The texture definition matches the subject
z-score page; only the scaler changes.
"""

# %%
# Load the cohort
# ---------------
# sphinx_gallery_thumbnail_number = 2
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.execution import SerialBackend
from habit.feature_preprocessing import RobustScaling, SubjectPreprocessingChain
from habit.habitat_model import KMeansHabitatModelFitter
from habit.pipeline import voxel_units
from habit.viz import plot_habitat_overlay
from habit.voxel_features import VoxelRadiomicsFeatures

DATA = fetch_demo()
MODALITIES = ("pre_contrast", "LAP", "PVP", "delay_3min")
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:2]
print(cohort)
Path("out").mkdir(exist_ok=True)
CACHE = str((Path("out") / "voxel_texture_cache").resolve())

# %%
# Extract voxel texture
# ---------------------
# Same R3B12 texture as the other pages in this section. Runtime knobs are
# explicit and do not change the feature values.
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

# %%
# Robust-scale inside each subject
# --------------------------------
# Impute is inserted first. After scaling, the median of a column is near 0.
robust = SubjectPreprocessingChain([RobustScaling(across_features=False)])
print("preprocess:", [method.spec.name for method in robust.methods])
column = next(
    name
    for name in fields[0].feature_names
    if "Contrast" in name and name.endswith("-LAP")
)

scaled_fields = []
for field in fields:
    raw = field.feature_frame()
    scaled = robust(raw)
    print(
        field.subject_id,
        column,
        "median",
        round(float(scaled[column].median()), 3),
    )
    scaled_fields.append(
        field.with_feature_frame(
            scaled,
            produced_by="subject_feature_preprocessor",
            spec_fingerprint=robust.spec.fingerprint(),
        )
    )

fig_hist, axes = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
axes[0].hist(fields[0].feature_frame()[column].to_numpy(), bins=30)
axes[0].set_title(f"{column} before")
axes[1].hist(scaled_fields[0].feature_frame()[column].to_numpy(), bins=30)
axes[1].set_title(f"{column} after robust scaling")
fig_hist.savefig("out/subject_robust_hist.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Fit and assign
# --------------
# Fitting stays in this process. Assignment reuses the voxel units.
fitter = KMeansHabitatModelFitter(n_habitats=3, n_init=3)
fitter.set_random_state(0)
units = [voxel_units(field) for field in scaled_fields]
model = fitter.fit(units, cohort=cohort)
print(model.summary())
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
    title="habitats (texture, subject robust scaling)",
    crop_to="labels",
)
fig_map.savefig("out/subject_robust_habitats.png", dpi=150, bbox_inches="tight")
plt.show()
