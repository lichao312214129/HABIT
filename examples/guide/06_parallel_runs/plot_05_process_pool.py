"""
Running subjects in separate processes
======================================

``RunPolicy(workers=2, backend="process")`` builds a
:class:`~habit.execution.ProcessPoolBackend`: each subject is sent to one
of two worker processes. This page runs the same texture extraction and
habitat assignment serially and in the pool, checks that the results
agree, and times both.

What a pool buys depends on where the work runs:

* **CPU-only machine**: each worker uses its own cores, so subjects run
  side by side.
* **One GPU**: only worker 0 gets the card. The other workers are pinned
  to CPU PyRadiomics so they do not fight over GPU memory, and a CPU
  worker is far slower than the GPU. ``HABIT_GPU_OVERSUBSCRIBE=wrap``
  lets every worker share the one card instead.
* **Several GPUs**: one worker per card,
  :doc:`/auto_examples/06_parallel_runs/plot_09_several_gpus`.

Every pool start pays process spawn and imports (a few seconds). With two
small subjects that overhead is larger than the work, so the numbers
below are about correctness and cost, not about a speed-up.

On Windows a process pool must be started under
``if __name__ == "__main__":`` (a spawned worker re-imports this file).
The whole run is therefore one guarded block that can be copied as is.
"""

# %%
# Set up the cohort, the extractor, and the pool
# ----------------------------------------------
# Building the backend does not start workers; ``map`` does. The
# extractor has no cache here, so both backends really compute texture.
# The device is left at ``"auto"``: CUDA when available, otherwise CPU.
# sphinx_gallery_thumbnail_number = 2
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.execution import SerialBackend, backend_from_policy
from habit.feature_preprocessing import SubjectPreprocessingChain, ZScoreScaling
from habit.habitat_model import KMeansHabitatModelFitter
from habit.pipeline import voxel_units
from habit.spec import RunPolicy
from habit.viz import plot_habitat_overlay
from habit.voxel_features import VoxelRadiomicsFeatures

# Change DATA / MODALITIES / ROI to your preprocessed layout.
DATA = fetch_demo()
MODALITIES = ("LAP",)
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:2]
print(cohort)
Path("out").mkdir(exist_ok=True)

# A light workload (radius 1, three GLCM features) keeps the CPU worker short.
texture = VoxelRadiomicsFeatures(
    modalities=list(MODALITIES),
    roi=ROI,
    kernel_radius=1,
    params={
        "imageType": {"Original": {}},
        "featureClass": {"glcm": ["Contrast", "Correlation", "JointEntropy"]},
        "setting": {"binWidth": 12},
    },
)
policy = RunPolicy(
    workers=2,
    backend="process",
    parallel_mode="persistent",
    auto_retry_rounds=0,
)


def by_subject(results: list) -> dict:
    """Key results by subject id: the pool yields subjects as they finish."""
    return {result.subject_id: result for result in results}

# %%
# Run serially and in the pool, then compare
# ------------------------------------------
# 1. texture serially, then in the pool (default single-GPU pinning), then
#    in the pool with ``HABIT_GPU_OVERSUBSCRIBE=wrap``; each is timed;
# 2. the largest absolute difference between the serial and pool fields;
# 3. z-score and fit once in this process (fitting sees every subject);
# 4. assignment serially and in the pool; the label maps must be identical.
#
# The pool returns subjects in the order they finish, not the input order,
# so results are matched by ``subject_id`` (``by_subject`` above).
#
# On a one-GPU machine the default pool computes the second subject on the
# CPU, so step 2 is also a CPU-versus-GPU check. The documented tolerance
# between the two paths is in :doc:`/how_to/voxel_texture`.
if __name__ == "__main__":
    try:
        import torch

        n_gpus = torch.cuda.device_count()
    except ImportError:
        n_gpus = 0
    print("visible CUDA devices:", n_gpus, "| CPU cores:", os.cpu_count())

    serial = SerialBackend()
    timings = {}

    start = time.perf_counter()
    fields_serial = [slot.result() for slot in serial.map(texture, cohort)]
    timings["serial"] = time.perf_counter() - start

    start = time.perf_counter()
    fields_pool = by_subject(slot.result() for slot in backend_from_policy(policy).map(texture, cohort))
    timings["pool"] = time.perf_counter() - start

    # Workers read this variable when they start, so set it before the map.
    os.environ["HABIT_GPU_OVERSUBSCRIBE"] = "wrap"
    try:
        start = time.perf_counter()
        fields_wrap = by_subject(slot.result() for slot in backend_from_policy(policy).map(texture, cohort))
        timings["pool, wrap"] = time.perf_counter() - start
    finally:
        del os.environ["HABIT_GPU_OVERSUBSCRIBE"]

    for name, seconds in timings.items():
        print(f"texture {name:>11}: {seconds:6.1f} s")
    for a in fields_serial:
        b, c = fields_pool[a.subject_id], fields_wrap[a.subject_id]
        print(
            f"{a.subject_id}: max |serial - pool| = {np.abs(a.values - b.values).max():.2e}, "
            f"max |serial - pool, wrap| = {np.abs(a.values - c.values).max():.2e}"
        )

    zscore = SubjectPreprocessingChain([ZScoreScaling(across_features=False)])
    units = [
        voxel_units(
            field.with_feature_frame(
                zscore(field.feature_frame()),
                produced_by="subject_feature_preprocessor",
                spec_fingerprint=zscore.spec.fingerprint(),
            )
        )
        for field in fields_serial
    ]
    fitter = KMeansHabitatModelFitter(n_habitats=3, n_init=3)
    fitter.set_random_state(0)
    model = fitter.fit(units, cohort=cohort)

    start = time.perf_counter()
    maps_serial = [slot.result() for slot in serial.map(model.assigner(), units)]
    timings["assign serial"] = time.perf_counter() - start
    start = time.perf_counter()
    maps_pool = by_subject(slot.result() for slot in backend_from_policy(policy).map(model.assigner(), units))
    timings["assign pool"] = time.perf_counter() - start
    for a in maps_serial:
        same = np.array_equal(a.label_array, maps_pool[a.subject_id].label_array)
        print(f"{a.subject_id}: serial labels == pool labels: {same}")
    print(f"assign serial {timings['assign serial']:.1f} s, pool {timings['assign pool']:.1f} s")

    fig, ax = plt.subplots(figsize=(6.4, 3.0), constrained_layout=True)
    ax.barh(list(timings), list(timings.values()), color="#4C78A8")
    ax.invert_yaxis()
    ax.set_xlabel("wall-clock seconds (2 subjects)")
    ax.set_title(f"serial vs process pool ({n_gpus} visible GPU)")
    fig.savefig("out/process_pool_timing.png", dpi=150, bbox_inches="tight")
    plt.show()

    fig_map = plot_habitat_overlay(
        cohort[0].image(ROI),
        maps_pool[cohort[0].subject_id],
        title="habitats assigned in the pool",
        crop_to="labels",
    )
    fig_map.savefig("out/process_pool_habitats.png", dpi=150, bbox_inches="tight")
    plt.show()
