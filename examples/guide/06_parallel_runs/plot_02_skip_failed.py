"""
Skipping a subject that fails
=============================

``on_subject_failure="continue"`` keeps the batch. One subject here has
no arterial-phase image, so texture extraction raises. The other subject
still receives a habitat map. The error is stored on that subject's slot.
"""

# %%
# Load one real subject and one incomplete copy
# ---------------------------------------------
# The copy drops ``LAP``. The extractor asks for every series, so that
# subject fails inside the real call rather than in a stand-in function.
# sphinx_gallery_thumbnail_number = 1
import dataclasses
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from habit.contracts import Cohort, cohort_from_directory
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
source = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[0]
missing_lap = dataclasses.replace(
    source,
    subject_id="missing_lap",
    images={name: ref for name, ref in source.images.items() if name != "LAP"},
)
cohort = Cohort([source, missing_lap], name="skip")
print(cohort)
print("missing_lap images:", sorted(missing_lap.images))
Path("out").mkdir(exist_ok=True)
CACHE = str((Path("out") / "voxel_texture_cache").resolve())

# %%
# Build the extractor
# -------------------
# The same four series are requested for every subject. ``missing_lap``
# cannot satisfy that request. This cell does not extract texture.
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
backend = SerialBackend(on_subject_failure="continue")
print(type(backend).__name__, backend.on_subject_failure)

# %%
# Extract, and keep the batch after the failure
# ---------------------------------------------
# The successful slot is a texture field. The failed slot carries the
# exception. Nothing is fit in this cell.
slots = list(backend.map(texture, cohort))
ok_fields = []
ok_subjects = []
for slot, subject in zip(slots, cohort):
    if slot.error is None:
        field = slot.result()
        print(
            slot.subject_id,
            f"{field.values.shape[0]} voxels, {len(field.feature_names)} columns",
        )
        ok_fields.append(field)
        ok_subjects.append(subject)
    else:
        print(slot.subject_id, type(slot.error).__name__, slot.error)

# %%
# Z-score the subjects that produced a field
# ------------------------------------------
# The incomplete subject is not in this list.
zscore = SubjectPreprocessingChain([ZScoreScaling(across_features=False)])
column = next(
    name
    for name in ok_fields[0].feature_names
    if "Contrast" in name and name.endswith("-LAP")
)
scaled_fields = []
for field in ok_fields:
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
# Fit on the successful subjects
# ------------------------------
fitter = KMeansHabitatModelFitter(n_habitats=3, n_init=3)
fitter.set_random_state(0)
units = [voxel_units(field) for field in scaled_fields]
model = fitter.fit(units, cohort=Cohort(ok_subjects, name="skip-ok"))
print(model.summary())

# %%
# Assign the subject that has every series
# ----------------------------------------
habitat_map = model.assigner()(units[0])
labels, counts = np.unique(habitat_map.label_array, return_counts=True)
present = {
    int(label): int(count)
    for label, count in zip(labels, counts)
    if int(label) != 0
}
print(habitat_map.subject_id, "voxels per habitat:", present)

# %%
# Which slots succeeded
# ---------------------
names = [slot.subject_id for slot in slots]
colors = ["#54A24B" if slot.error is None else "#E45756" for slot in slots]
fig, ax = plt.subplots(figsize=(6.4, 2.6))
ax.barh(list(reversed(names)), [1, 1], color=list(reversed(colors)))
ax.set_xlabel("slot")
ax.set_title("continue after a missing series")
fig.savefig("out/skip_failed.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Habitat map of the complete subject
# -----------------------------------
fig_map = plot_habitat_overlay(
    source.image(ROI),
    habitat_map,
    title="habitats (subject with all series)",
    crop_to="labels",
)
fig_map.savefig("out/skip_failed_habitats.png", dpi=150, bbox_inches="tight")
plt.show()
