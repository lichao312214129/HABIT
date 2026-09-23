"""
Per-subject wall-clock limit
============================

``subject_timeout_sec`` is enforced only by the process backend. The
limit on this page is 2 seconds, below a real texture extraction, and
the extractor is not allowed to read the on-disk texture cache.
``parallel_mode="isolated"`` and ``workers=1`` run one subject at a
time in its own process, so the limit is recorded on every slot. Both
finish as :class:`~habit.execution.SubjectTimeoutError`. A study that
should finish uses a limit on the order of minutes.
"""

# %%
# Load the cohort
# ---------------
# sphinx_gallery_thumbnail_number = 1
from pathlib import Path

import matplotlib.pyplot as plt

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.execution import SubjectTimeoutError, backend_from_policy
from habit.spec import RunPolicy
from habit.voxel_features import VoxelRadiomicsFeatures

DATA = fetch_demo()
MODALITIES = ("pre_contrast", "LAP", "PVP", "delay_3min")
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:2]
print(cohort)
Path("out").mkdir(exist_ok=True)

# %%
# Set a 2 second limit and refuse the texture cache
# -------------------------------------------------
# A cache hit would finish before the limit. ``workers=1`` runs one
# subject at a time, so one timeout does not take down the other subject.
RADIOMICS_PARAMS = {
    "imageType": {"Original": {}},
    "featureClass": {
        "firstorder": ["Mean", "Entropy"],
        "glcm": ["Contrast", "Correlation", "Idm", "JointEntropy"],
        "glrlm": ["ShortRunEmphasis", "LongRunEmphasis"],
    },
    "setting": {"binWidth": 12},
}
# No cache_dir: a cache hit would finish before the 2 second limit.
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
policy = RunPolicy(
    workers=1,
    backend="process",
    parallel_mode="isolated",
    on_subject_failure="continue",
    subject_timeout_sec=2.0,
    auto_retry_rounds=0,
)
backend = backend_from_policy(policy)
print(
    type(backend).__name__,
    "timeout",
    backend.policy.subject_timeout_sec,
)


# %%
# Both subjects hit the limit
# ---------------------------
if __name__ == "__main__":
    slots = list(backend.map(texture, cohort))
    for slot in slots:
        print(slot.subject_id, type(slot.error).__name__ if slot.error else "ok")
        if slot.error is not None:
            print(slot.error)

# %%
# Mark every timed-out slot
# -------------------------
if __name__ == "__main__":
    labels = [slot.subject_id for slot in slots]
    timed_out = [
        slot.error is not None and isinstance(slot.error, SubjectTimeoutError)
        for slot in slots
    ]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ax.barh(
        list(reversed(labels)),
        [1, 1],
        color=["#E45756" if flag else "#54A24B" for flag in reversed(timed_out)],
    )
    ax.set_xlabel("slot")
    ax.set_title("wall-clock limit 2 s")
    fig.savefig("out/wall_clock.png", dpi=150, bbox_inches="tight")
    plt.show()
