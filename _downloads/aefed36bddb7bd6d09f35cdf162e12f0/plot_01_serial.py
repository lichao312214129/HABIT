"""
Running a cohort serially
=========================

:class:`~habit.execution.SerialBackend` runs the texture extraction and
the habitat assignment in this process, one subject after another.
Z-scoring and fitting stay here as well: the scaler is per subject, and
the fit has to see every subject. The same chain written as a
:class:`~habit.spec.HabitatSpec` is
:doc:`/auto_examples/02_voxel/plot_06_texture_preprocessing`.
"""

# %%
# Load the cohort
# ---------------
# sphinx_gallery_thumbnail_number = 1
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.execution import SerialBackend
from habit.feature_preprocessing import SubjectPreprocessingChain, ZScoreScaling
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
# Build the extractor and the serial backend
# ------------------------------------------
# Radius 3 is a 7×7×7 neighbourhood. ``binWidth`` 12 is the PyRadiomics
# grey-level width. The device is left at ``"auto"``: CUDA when this
# machine has it, otherwise CPU. This cell does not extract texture.
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
    cache_dir=CACHE,
)
zscore = SubjectPreprocessingChain([ZScoreScaling(across_features=False)])
fitter = KMeansHabitatModelFitter(n_habitats=3, n_init=3)
fitter.set_random_state(0)
backend = SerialBackend()
print(type(backend).__name__)
print("feature classes:", sorted(RADIOMICS_PARAMS["featureClass"]))

# %%
# Extract texture
# ---------------
# ``SerialBackend.map`` yields one slot per subject, in cohort order.
fields = [slot.result() for slot in backend.map(texture, cohort)]
for field in fields:
    print(
        f"{field.subject_id}: {field.values.shape[0]} voxels, "
        f"{len(field.feature_names)} columns"
    )

# %%
# Z-score inside each subject
# ---------------------------
# Mean and standard deviation are computed on that subject only.
column = next(
    name
    for name in fields[0].feature_names
    if "Contrast" in name and name.endswith("-LAP")
)
scaled_fields = []
for field in fields:
    scaled = zscore(field.feature_frame())
    print(
        field.subject_id,
        column,
        "mean",
        round(float(scaled[column].mean()), 3),
        "std",
        round(float(scaled[column].std()), 3),
    )
    scaled_fields.append(
        field.with_feature_frame(
            scaled,
            produced_by="subject_feature_preprocessor",
            spec_fingerprint=zscore.spec.fingerprint(),
        )
    )

# %%
# Fit on the pooled voxels
# ------------------------
# Fitting stays in this process: it has to see every subject.
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
    title="habitats (serial backend)",
    crop_to="labels",
)
fig_map.savefig("out/serial_habitats.png", dpi=150, bbox_inches="tight")
plt.show()
