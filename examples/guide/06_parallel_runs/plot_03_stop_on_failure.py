"""
Stopping at the first failure
=============================

``on_subject_failure="fail_fast"`` re-raises the first subject error and
does not run the rest. The incomplete subject is first, so the real
extraction of the complete subject never starts.
"""

# %%
# Put the incomplete subject first
# --------------------------------
# Dropping ``LAP`` makes texture extraction raise as soon as that series
# is requested. The complete subject is second and must stay unvisited.
# sphinx_gallery_thumbnail_number = 1
import dataclasses
from pathlib import Path

import matplotlib.pyplot as plt

from habit.contracts import Cohort, Subject, cohort_from_directory
from habit.datasets import fetch_demo
from habit.execution import SerialBackend
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
cohort = Cohort([missing_lap, source], name="stop")
print([subject.subject_id for subject in cohort])
Path("out").mkdir(exist_ok=True)

# %%
# Build the extractor
# -------------------
# No cache directory: this page should raise, not reuse a finished field.
# This cell does not call the extractor.
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
)
print("feature classes:", sorted(RADIOMICS_PARAMS["featureClass"]))
seen: list[str] = []


def extract(subject: Subject):
    """Record the id, then run the real texture extractor."""
    seen.append(subject.subject_id)
    return texture(subject)


# %%
# Stop on the missing series
# --------------------------
# ``seen`` records who actually entered the extractor. The second id is
# absent, and the exception type is printed.
try:
    list(SerialBackend(on_subject_failure="fail_fast").map(extract, cohort))
except Exception as exc:
    print(type(exc).__name__)
    print(exc)
print("visited:", seen)

# %%
# Who was visited
# ---------------
# Grey is the subject the backend did not start.
labels = [subject.subject_id for subject in cohort]
colors = ["#4C78A8" if subject_id in seen else "#D0D0D0" for subject_id in labels]
fig, ax = plt.subplots(figsize=(6.4, 2.4))
ax.barh(list(reversed(labels)), [1, 1], color=list(reversed(colors)))
ax.set_xlabel("visited")
ax.set_title("stop at the first failure")
fig.savefig("out/stop_on_failure.png", dpi=150, bbox_inches="tight")
plt.show()
