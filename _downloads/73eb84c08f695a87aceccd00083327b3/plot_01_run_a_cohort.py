"""
Running a cohort in parallel
============================

The study is the one from :doc:`/auto_quickstart/plot_quickstart_python`.
Only the ``backend=`` argument changes: serially, subjects run one after
the other in this process; with ``RunPolicy(workers=4, backend="process")``
four worker processes take subjects side by side. The habitat maps must
be identical either way, and this page checks that.

Which backend when:

=========================================  ==========================================
Where the work runs                        What to use
=========================================  ==========================================
CPU (raw intensities, most kernels)        a process pool, ``workers`` = cores to spare
One GPU (voxel texture)                    serial, or a pool with
                                           ``HABIT_GPU_OVERSUBSCRIBE=wrap``
Several GPUs (voxel texture)               one worker per card,
                                           :doc:`plot_03_several_gpus`
=========================================  ==========================================

On one GPU only worker 0 gets the card; the other workers are pinned to
CPU so they do not fight over GPU memory. ``wrap`` lets them share it.
``parallel_mode="isolated"`` starts a fresh process for every subject:
same results, more start-up cost, useful when a subject can crash the
interpreter.

On Windows a process pool must start under ``if __name__ == "__main__":``
(a spawned worker re-imports this file), so the whole run is one guarded
block that can be copied as is.
"""

# %%
# Serial, then a pool of four workers
# -----------------------------------
# Starting a pool costs process spawn and imports. ``reuse_workers()``
# keeps the same workers up across calls, so only the first call pays it.
# The pool returns maps in cohort order.
# sphinx_gallery_thumbnail_number = 1
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.execution import SerialBackend, backend_from_policy
from habit.recipes import two_step_habitat
from habit.spec import RunPolicy

if __name__ == "__main__":
    # Change DATA / MODALITIES / ROI to your preprocessed layout.
    DATA = fetch_demo()
    MODALITIES = ("pre_contrast", "LAP", "PVP", "delay_3min")
    ROI = "LAP"
    cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)
    study = two_step_habitat(modalities=MODALITIES, roi=ROI, n_supervoxels=30, random_seed=0)

    timings = {}
    start = time.perf_counter()
    serial = study.fit_predict(cohort, backend=SerialBackend())
    timings["serial"] = time.perf_counter() - start

    backend = backend_from_policy(RunPolicy(workers=4, backend="process"))
    with backend.reuse_workers():
        start = time.perf_counter()
        pooled = study.fit_predict(cohort, backend=backend)
        timings["pool, first call"] = time.perf_counter() - start
        start = time.perf_counter()
        pooled = study.fit_predict(cohort, backend=backend)
        timings["pool, workers reused"] = time.perf_counter() - start

    for name, seconds in timings.items():
        print(f"{name:>21}: {seconds:5.1f} s")
    for a, b in zip(serial.habitat_maps, pooled.habitat_maps):
        print(f"{a.subject_id}: serial labels == pool labels: {np.array_equal(a.label_array, b.label_array)}")

    # With five small tumours the first pool call is about as slow as the
    # serial run: spawning four workers costs what they save. The gain
    # shows once workers are reused, and grows with more or larger
    # subjects.
    Path("out").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 2.6), constrained_layout=True)
    ax.barh(list(timings), list(timings.values()), color="#4C78A8")
    ax.invert_yaxis()
    ax.set_xlabel(f"wall-clock seconds ({len(cohort)} subjects)")
    ax.set_title("serial vs process pool (4 workers)")
    fig.savefig("out/run_a_cohort_timing.png", dpi=150, bbox_inches="tight")
    plt.show()
