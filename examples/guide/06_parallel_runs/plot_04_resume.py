"""
Resuming finished subjects
==========================

A :class:`~habit.execution.CheckpointStore` records each success. The
checkpoint key includes the texture specification, so changing the
feature list or ``binWidth`` does not reuse the old field. The second
``map`` returns ``from_cache=True`` and does not recompute texture.
"""

# %%
# Load the cohort
# ---------------
# sphinx_gallery_thumbnail_number = 1
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt

from habit.contracts import Subject, cohort_from_directory
from habit.datasets import fetch_demo
from habit.execution import CheckpointStore, SerialBackend
from habit.voxel_features import VoxelRadiomicsFeatures

DATA = fetch_demo()
MODALITIES = ("pre_contrast", "LAP", "PVP", "delay_3min")
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:2]
print(cohort)
Path("out").mkdir(exist_ok=True)
CACHE = str((Path("out") / "voxel_texture_cache").resolve())

# %%
# Build the extractor
# -------------------
# ``cache_key`` below folds in ``texture.spec.fingerprint()``, which
# changes when the series, radius, or ``binWidth`` change. This cell
# does not extract texture.
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


class CachedTexture:
    """Texture extractor with a spec-scoped checkpoint key."""

    def __init__(self, extractor: VoxelRadiomicsFeatures) -> None:
        self.extractor = extractor

    def __call__(self, subject: Subject):
        """Return this subject's voxel-texture field."""
        return self.extractor(subject)

    def cache_key(self, subject: Subject) -> str:
        """Return a key that changes when the texture specification changes."""
        digest = self.extractor.spec.fingerprint()
        return f"voxel_radiomics:{digest}:{subject.subject_id}"


operator = CachedTexture(texture)
print(operator.cache_key(cohort[0]))

# %%
# First pass writes the checkpoint
# --------------------------------
# A new store is empty, so both subjects are computed.
store = CheckpointStore(Path(tempfile.mkdtemp(prefix="habit_ckpt_")))
backend = SerialBackend()
first = list(backend.map(operator, cohort, checkpoint=store))
print("first from_cache:", [slot.from_cache for slot in first])

# %%
# Second pass reads the checkpoint
# --------------------------------
# ``from_cache`` is true, and the field is the one stored on the first pass.
second = list(backend.map(operator, cohort, checkpoint=store))
print("second from_cache:", [slot.from_cache for slot in second])
for slot in second:
    field = slot.result()
    print(
        slot.subject_id,
        f"{field.values.shape[0]} voxels, {len(field.feature_names)} columns",
    )

# %%
# Computed pass versus cached pass
# --------------------------------
labels = [slot.subject_id for slot in second]
positions = range(len(labels))
fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.bar(
    [index - 0.18 for index in positions],
    [0 if slot.from_cache else 1 for slot in first],
    width=0.36,
    label="computed",
    color="#4C78A8",
)
ax.bar(
    [index + 0.18 for index in positions],
    [1 if slot.from_cache else 0 for slot in second],
    width=0.36,
    label="from cache",
    color="#54A24B",
)
ax.set_xticks(list(positions))
ax.set_xticklabels(labels)
ax.set_ylabel("pass")
ax.set_title("resume skips finished texture fields")
ax.legend()
fig.savefig("out/resume.png", dpi=150, bbox_inches="tight")
plt.show()
