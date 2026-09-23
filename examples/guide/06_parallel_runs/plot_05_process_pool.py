"""
Running subjects in separate processes
======================================

``RunPolicy(workers=2, backend="process")`` builds a
:class:`~habit.execution.ProcessPoolBackend`. Texture extraction and
habitat assignment run in that pool. Z-scoring and fitting stay in this
process. On Windows the ``map`` calls sit under ``__main__`` so a worker
does not start another pool.
"""

# %%
# Load the cohort
# ---------------
# Building the backend in the next cell does not start workers. ``map`` does.
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
# Build the pool
# --------------
# ``subject_timeout_sec`` is enforced only on this backend. The short
# wall-clock page sets a limit that texture cannot meet; this page leaves
# the default so both subjects finish. Workers are not started here.
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
    on_subject_failure="continue",
    auto_retry_rounds=0,
)
backend = backend_from_policy(policy)
print(type(backend).__name__, backend.workers, backend.policy.parallel_mode)

# %%
# Extract texture in the pool
# ---------------------------
# On Windows this guard stops a spawned worker from starting another pool.
if __name__ == "__main__":
    fields = [slot.result() for slot in backend.map(texture, cohort)]
    for field in fields:
        print(
            f"{field.subject_id}: {field.values.shape[0]} voxels, "
            f"{len(field.feature_names)} columns"
        )

# %%
# Z-score in this process
# -----------------------
# The pool is idle. Each subject is scaled on its own rows.
if __name__ == "__main__":
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
# Fit in this process
# -------------------
# The fitter has to see every subject, so it does not go through ``map``.
if __name__ == "__main__":
    units = [voxel_units(field) for field in scaled_fields]
    model = fitter.fit(units, cohort=cohort)
    print(model.summary())

# %%
# Assign habitats in the pool
# ---------------------------
if __name__ == "__main__":
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
        title="habitats (process pool)",
        crop_to="labels",
    )
    fig_map.savefig("out/process_pool_habitats.png", dpi=150, bbox_inches="tight")
    plt.show()
