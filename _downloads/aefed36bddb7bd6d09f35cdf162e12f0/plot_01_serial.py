"""
Running a cohort serially
=========================

:class:`~habit.execution.SerialBackend` runs the texture extraction and
the habitat assignment in this process, one subject after another.
Z-scoring and fitting stay here as well: the scaler is per subject, and
the fit has to see every subject. The matching :class:`~habit.spec.HabitatSpec`
is on the subject z-score page.
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
# Extract, z-score, fit, assign
# -----------------------------
# ``SerialBackend.map`` yields one slot per subject, in cohort order.
# The printed card and the voxel counts are the result of that chain.
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
zscore = SubjectPreprocessingChain([ZScoreScaling(across_features=False)])
fitter = KMeansHabitatModelFitter(n_habitats=3, n_init=3)
fitter.set_random_state(0)
backend = SerialBackend()

fields = [slot.result() for slot in backend.map(texture, cohort)]
for field in fields:
    print(
        f"{field.subject_id}: {field.values.shape[0]} voxels, "
        f"{len(field.feature_names)} columns"
    )
scaled_fields = []
for field in fields:
    scaled = zscore(field.feature_frame())
    scaled_fields.append(
        field.with_feature_frame(
            scaled,
            produced_by="subject_feature_preprocessor",
            spec_fingerprint=zscore.spec.fingerprint(),
        )
    )
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
    title="habitats (serial backend)",
    crop_to="labels",
)
fig_map.savefig("out/serial_habitats.png", dpi=150, bbox_inches="tight")
plt.show()
