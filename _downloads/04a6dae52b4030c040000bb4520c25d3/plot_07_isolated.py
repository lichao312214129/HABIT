"""
One fresh process per subject
=============================

``parallel_mode="isolated"`` starts a new process for every subject
instead of keeping a worker alive. The texture definition and the
subject z-score match the persistent process-pool page. Spawn cost is
higher; the habitat counts are the same chain.
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
# Select one fresh process per subject
# ------------------------------------
# ``isolated`` does not keep a worker after a subject finishes.
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
policy = RunPolicy(
    workers=2,
    backend="process",
    parallel_mode="isolated",
    auto_retry_rounds=0,
)
backend = backend_from_policy(policy)
print(type(backend).__name__, backend.policy.parallel_mode)

# %%
# Extract, fit here, assign in a fresh process per subject
# --------------------------------------------------------
# On Windows the pool is started under ``__main__`` so a spawned worker
# does not start another pool. Copy this cell together with the two above.
if __name__ == "__main__":
    fields = [slot.result() for slot in backend.map(texture, cohort)]
    for field in fields:
        print(
            f"{field.subject_id}: {field.values.shape[0]} voxels, "
            f"{len(field.feature_names)} columns"
        )
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
    # Completion order is not cohort order.
    maps = {item.subject_id: item for item in (slot.result() for slot in backend.map(model.assigner(), units))}
    for subject in cohort:
        habitat_map = maps[subject.subject_id]
        labels, counts = np.unique(habitat_map.label_array, return_counts=True)
        present = {
            int(label): int(count)
            for label, count in zip(labels, counts)
            if int(label) != 0
        }
        print(habitat_map.subject_id, "voxels per habitat:", present)
    fig_map = plot_habitat_overlay(
        cohort[0].image(ROI),
        maps[cohort[0].subject_id],
        title="habitats (isolated processes)",
        crop_to="labels",
    )
    fig_map.savefig("out/isolated_habitats.png", dpi=150, bbox_inches="tight")
    plt.show()
