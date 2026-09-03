"""
Parallel and checkpoints
========================

:class:`~habit.spec.HabitatSpec` declares what to compute;
:class:`~habit.spec.RunPolicy` declares how to schedule it. Pass a
:class:`~habit.execution.process_pool.ProcessPoolBackend` into
:meth:`~habit.recipes.Study.fit_predict`. With a fixed ``random_seed``
the maps match serial execution.

.. important::
   **Windows + process pool:** spawning workers re-imports your script.
   Put the run inside ``if __name__ == "__main__":`` (already done below).
   See also :doc:`/tutorial/quickstart`.

Pick a backend with :func:`~habit.execution.backend_from_policy`:

* Debug one subject: ``RunPolicy(workers=1, backend="serial", ...)``
* Cohort default: ``RunPolicy(workers=2, backend="process", ...)``
* Fresh process per subject: same, plus ``parallel_mode="isolated"``
"""

# %%
# Shared two-step spec and a small demo cohort.
from pathlib import Path
import time

import matplotlib.pyplot as plt
import pandas as pd

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.execution.process_pool import ProcessPoolBackend
from habit.recipes import Study
from habit.spec import HabitatSpec, RunPolicy, Spec, Stage
from habit.viz import plot_habitat_overlay

import habit.voxel_features  # registers local_entropy for the texture timing block

DATA = fetch_demo()
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=("LAP",), roi=ROI)[:5]
Path("out").mkdir(exist_ok=True)
print(f"Cohort: {list(cohort.subject_ids)}")

raw_spec = HabitatSpec(
    name="parallel_demo",
    stages=(
        Stage("extract_voxel_features", Spec("raw", {"modalities": ["LAP"]})),
        Stage(
            "preprocess1",
            Spec("winsorize", {"winsor_limits": (0.05, 0.05), "across_features": False}),
        ),
        Stage("preprocess2", Spec("minmax", {"across_features": False})),
        Stage("partition", Spec("kmeans", {"n_supervoxels": 6, "n_init": 3})),
        Stage("pool", Spec("pool")),
        Stage(
            "fit",
            Spec(
                "kmeans",
                {
                    "min_habitats": 2,
                    "max_habitats": 3,
                    "validation": "elbow",
                    "n_init": 3,
                },
            ),
        ),
        Stage("assign", Spec("nearest_centroid")),
        Stage("quantify", Spec("volume")),
    ),
    random_seed=21,
)

# %%
# Serial baseline (no process pool).
if __name__ == "__main__":
    t0 = time.perf_counter()
    serial_result = Study(spec=raw_spec).fit_predict(cohort)
    serial_s = time.perf_counter() - t0
    print(f"serial: {serial_s:.2f}s, n_maps={len(serial_result.habitat_maps)}")
    print(serial_result.features.frame.head())
    fig = plot_habitat_overlay(
        cohort[0].image(ROI),
        serial_result.habitat_maps[0],
        title="habitats",
    )
    fig.savefig("out/parallel_execution_overlay.png", dpi=150, bbox_inches="tight")
    plt.show()
    serial_result.features.frame.head()

# %%
# Persistent process pool (workers=2). Labels must match serial.
if __name__ == "__main__":
    policy_persistent = RunPolicy(
        workers=2,
        backend="process",
        on_subject_failure="continue",
        parallel_mode="persistent",
    )
    t0 = time.perf_counter()
    persistent_result = Study(spec=raw_spec).fit_predict(
        cohort, backend=ProcessPoolBackend.from_policy(policy_persistent)
    )
    persistent_s = time.perf_counter() - t0
    mismatches_persistent = sum(
        1
        for left, right in zip(serial_result.habitat_maps, persistent_result.habitat_maps)
        if not (left.label_array == right.label_array).all()
    )
    print(
        f"process_pool persistent: {persistent_s:.2f}s, "
        f"label_mismatches_vs_serial={mismatches_persistent}"
    )

# %%
# Isolated process pool (fresh worker per subject).
if __name__ == "__main__":
    policy_isolated = RunPolicy(
        workers=2,
        backend="process",
        on_subject_failure="continue",
        parallel_mode="isolated",
    )
    t0 = time.perf_counter()
    isolated_result = Study(spec=raw_spec).fit_predict(
        cohort, backend=ProcessPoolBackend.from_policy(policy_isolated)
    )
    isolated_s = time.perf_counter() - t0
    mismatches_isolated = sum(
        1
        for left, right in zip(serial_result.habitat_maps, isolated_result.habitat_maps)
        if not (left.label_array == right.label_array).all()
    )
    scheduler_table = pd.DataFrame(
        [
            {
                "config": "serial",
                "wall_time_s": serial_s,
                "n_maps": len(serial_result.habitat_maps),
                "label_mismatches_vs_serial": 0,
            },
            {
                "config": "process_pool_persistent_workers=2",
                "wall_time_s": persistent_s,
                "n_maps": len(persistent_result.habitat_maps),
                "label_mismatches_vs_serial": mismatches_persistent,
            },
            {
                "config": "process_pool_isolated_workers=2",
                "wall_time_s": isolated_s,
                "n_maps": len(isolated_result.habitat_maps),
                "label_mismatches_vs_serial": mismatches_isolated,
            },
        ]
    )
    print(scheduler_table.to_string(index=False))
    scheduler_table

# %%
# Heavier voxel field (local entropy): timing only, not a new definition.
if __name__ == "__main__":
    texture_spec = HabitatSpec(
        name="parallel_texture_input",
        stages=(
            Stage(
                "extract_voxel_features",
                Spec(
                    "local_entropy",
                    {"modalities": ["LAP"], "kernel_size": 3, "bins": 32},
                ),
            ),
            Stage(
                "preprocess1",
                Spec(
                    "winsorize",
                    {"winsor_limits": (0.05, 0.05), "across_features": False},
                ),
            ),
            Stage("preprocess2", Spec("minmax", {"across_features": False})),
            Stage("partition", Spec("kmeans", {"n_supervoxels": 6, "n_init": 3})),
            Stage("pool", Spec("pool")),
            Stage(
                "fit",
                Spec(
                    "kmeans",
                    {
                        "min_habitats": 2,
                        "max_habitats": 3,
                        "validation": "elbow",
                        "n_init": 3,
                    },
                ),
            ),
            Stage("assign", Spec("nearest_centroid")),
            Stage("quantify", Spec("volume")),
        ),
        random_seed=21,
    )
    texture_cohort = cohort[:1]
    t0 = time.perf_counter()
    texture_serial = Study(spec=texture_spec).fit_predict(texture_cohort)
    texture_serial_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    texture_parallel = Study(spec=texture_spec).fit_predict(
        texture_cohort, backend=ProcessPoolBackend.from_policy(policy_persistent)
    )
    texture_parallel_s = time.perf_counter() - t0
    texture_table = pd.DataFrame(
        [
            {
                "config": "texture_serial",
                "wall_time_s": texture_serial_s,
                "n_maps": len(texture_serial.habitat_maps),
            },
            {
                "config": "texture_process_pool_persistent_workers=2",
                "wall_time_s": texture_parallel_s,
                "n_maps": len(texture_parallel.habitat_maps),
            },
        ]
    )
    print("Texture-input timing (scheduler cost only):")
    print(texture_table.to_string(index=False))
    texture_table
