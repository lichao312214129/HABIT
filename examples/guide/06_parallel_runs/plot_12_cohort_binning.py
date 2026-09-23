"""
Cohort binning on voxel texture
===============================

Bin edges are learned on the pooled training rows. The same bin index
then means the same texture range in every subject. This is not a
rescaling: the stored values are bin indices.
"""

# %%
# Load the cohort
# ---------------
# sphinx_gallery_thumbnail_number = 2
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.execution import SerialBackend
from habit.feature_preprocessing import Binning, CohortPreprocessingChain
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
# Fit bin edges on the pooled rows
# --------------------------------
# Six uniform bins. The value counts are the bin occupancy of one column
# in the first subject after the shared edges are applied.
binning = CohortPreprocessingChain(
    [Binning(n_bins=6, bin_strategy="uniform", across_features=False)]
)
pooled = pd.concat(
    [field.feature_frame() for field in fields],
    axis=0,
    ignore_index=True,
)
binning.fit(pooled)
print("fitted:", binning.is_fitted)
column = next(
    name
    for name in fields[0].feature_names
    if "Contrast" in name and name.endswith("-LAP")
)

scaled_fields = []
for field in fields:
    scaled = binning.transform(field.feature_frame())
    print(field.subject_id, column)
    print(scaled[column].value_counts().sort_index())
    scaled_fields.append(
        field.with_feature_frame(
            scaled,
            produced_by="cohort_feature_preprocessor",
            spec_fingerprint=binning.spec.fingerprint(),
        )
    )

fig_hist, axes = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
axes[0].hist(fields[0].feature_frame()[column].to_numpy(), bins=30)
axes[0].set_title(f"{column} before")
axes[1].hist(
    scaled_fields[0].feature_frame()[column].to_numpy(),
    bins=6,
)
axes[1].set_title(f"{column} after cohort binning")
fig_hist.savefig("out/cohort_binning_hist.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Fit on the pooled voxels
# ------------------------
# Fitting stays in this process. It has to see every subject.
# Bin indices from the cell above are the values being clustered.
fitter = KMeansHabitatModelFitter(n_habitats=3, n_init=3)
fitter.set_random_state(0)
units = [voxel_units(field) for field in scaled_fields]
model = fitter.fit(units, cohort=cohort)
print(model.summary())

# %%
# Assign habitats
# ---------------
# Assignment reuses these units. It does not extract texture again.
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
    title="habitats (texture, cohort binning)",
    crop_to="labels",
)
fig_map.savefig("out/cohort_binning_habitats.png", dpi=150, bbox_inches="tight")
plt.show()
