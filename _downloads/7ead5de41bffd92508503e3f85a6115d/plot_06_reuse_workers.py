"""
Reusing workers across passes
=============================

A cold process pool pays spawn and import on the first ``map``.
``with backend.reuse_workers():`` keeps those workers for the assignment
``map`` that follows. Z-scoring and fitting run in this process while the
workers stay up.
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
from habit.execution import backend_from_policy
from habit.feature_preprocessing import SubjectPreprocessingChain, ZScoreScaling
from habit.habitat_model import KMeansHabitatModelFitter
from habit.pipeline import voxel_units
from habit.spec import RunPolicy
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
# Build one pool
# --------------
# Workers are not started until the session below is entered.
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
policy = RunPolicy(
    workers=2,
    backend="process",
    parallel_mode="persistent",
    auto_retry_rounds=0,
)
backend = backend_from_policy(policy)
print(type(backend).__name__, backend.policy.parallel_mode)

# %%
# Open one worker session
# -----------------------
# The session stays open through scaling, fitting, and the assignment
# map. It closes in the last cell. Isolated mode would ignore this.
if __name__ == "__main__":
    worker_session = backend.reuse_workers()
    worker_session.__enter__()
    print("worker session open, workers:", backend.workers)

# %%
# Extract texture on those workers
# --------------------------------
if __name__ == "__main__":
    fields = [slot.result() for slot in backend.map(texture, cohort)]
    for field in fields:
        print(
            f"{field.subject_id}: {field.values.shape[0]} voxels, "
            f"{len(field.feature_names)} columns"
        )

# %%
# Z-score and fit while the workers stay up
# ------------------------------------------
# Both steps run in this process. The worker session is still open.
if __name__ == "__main__":
    scaled_fields = []
    for field in fields:
        scaled_fields.append(
            field.with_feature_frame(
                zscore(field.feature_frame()),
                produced_by="subject_feature_preprocessor",
                spec_fingerprint=zscore.spec.fingerprint(),
            )
        )
    units = [voxel_units(field) for field in scaled_fields]
    model = fitter.fit(units, cohort=cohort)
    print(model.summary())

# %%
# Assign on the same workers, then close the session
# --------------------------------------------------
if __name__ == "__main__":
    try:
        maps = [slot.result() for slot in backend.map(model.assigner(), units)]
        for habitat_map in maps:
            labels, counts = np.unique(habitat_map.label_array, return_counts=True)
            present = {
                int(label): int(count)
                for label, count in zip(labels, counts)
                if int(label) != 0
            }
            print(habitat_map.subject_id, "voxels per habitat:", present)
    finally:
        worker_session.__exit__(None, None, None)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(
        ["texture map", "assign map"],
        [backend.workers, backend.workers],
        color="#4C78A8",
    )
    ax.set_ylabel("workers")
    ax.set_title("workers stay up across both maps")
    fig.savefig("out/reuse_workers.png", dpi=150, bbox_inches="tight")
    plt.show()
    fig_map = plot_habitat_overlay(
        cohort[0].image(ROI),
        maps[0],
        title="habitats (reused workers)",
        crop_to="labels",
    )
    fig_map.savefig("out/reuse_workers_habitats.png", dpi=150, bbox_inches="tight")
    plt.show()
