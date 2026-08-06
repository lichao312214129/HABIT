#!/usr/bin/env python
"""Serial vs parallel timing: direct_pooling recipe, voxel texture on torch GPU.

Pipeline: voxel_radiomics (7x7x7, bundled R3B12 preset, T1+T2 by default,
torch GPU, ROI mask from T1) -> winsorize/minmax -> (no supervoxels) ->
cohort binning -> kmeans habitats -> assign -> volume. Every ROI voxel is
a clustering unit.

The recipe isolates per-subject failures itself (v0.1 continue semantics via
``raise_on_failure=False``), so sub1's deterministic GeometryError is skipped,
not fatal. ``_backend_session`` reuses the persistent pool across the units
stage; the label stage reuses in-memory units (no second radiomics pass).

Safety defaults (avoid laptop freeze on a single 8 GB GPU):
* ``N_SUBJECTS`` small, ``workers`` capped to 2 and to the GPU pool
* heavy clustering HTML/PNG plots off during timing
* override with env: ``HABIT_TIMING_N=5``, ``HABIT_TIMING_WORKERS=2``,
  ``HABIT_TIMING_MODALITIES=T1,T2``, ``HABIT_TIMING_PLOTS=1``

Run::

    python habit/domain/voxel_features/mytest_timing_pooling.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Running this file as a script puts ``.../voxel_features`` on sys.path[0],
# which can make Python prefer a stale site-packages ``habit`` over the
# workspace checkout. Force the repo root first.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from habit import HabitatSpec, RunPolicy, Spec, cohort_from_directory
from habit.execution.process_pool import ProcessPoolBackend
import habit.recipes as recipes

DATA = Path(r"F:\work\habit_project\.cursor\test\resample_02")
OUT_ROOT = Path("out/timing_pooling")
# Single mask folder key under masks/<subject>/<roi>/ (not a modality list).
ROI = "T1"
N_SUBJECTS = int(os.environ.get("HABIT_TIMING_N", "10"))
# Image modalities: both T1 and T2 by default; ROI mask still comes from T1.
_MODALITY_ENV = os.environ.get("HABIT_TIMING_MODALITIES", "T1,T2")
MODALITIES = tuple(
    part.strip() for part in _MODALITY_ENV.split(",") if part.strip()
)
_WRITE_PLOTS = os.environ.get("HABIT_TIMING_PLOTS", "0").strip() in {
    "1",
    "true",
    "True",
    "yes",
}


if __name__ == "__main__":
    # Cap BLAS threads in the parent too (serial path + kmeans silhouette).
    for _key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(_key, "1")

    cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)
    subjects = tuple(list(cohort)[:N_SUBJECTS])
    # Match v0.1 pooling YAML default (processes: 2); never exceed subject count.
    workers = 2
    print(f"data={DATA}")
    print(
        f"cohort={len(subjects)}/{len(cohort)} subjects, workers={workers}, "
        f"modalities={MODALITIES}, plots={_WRITE_PLOTS}",
        flush=True,
    )

    spec = HabitatSpec(
        name="timing_pooling_texture",
        voxel_feature_extractor=Spec(
            "voxel_radiomics",
            {
                "modalities": list(MODALITIES),
                "kernel_radius": 3,
                "voxel_batch": 2000,
                "use_torch_radiomics": "auto",
            },
        ),
        voxel_feature_preprocessors=(
            Spec("winsorize", {"winsor_limits": (0.05, 0.05), "across_features": False}),
            Spec("minmax", {"across_features": False}),
        ),
        supervoxelizer=None,  # direct pooling: every ROI voxel is a unit
        cohort_feature_preprocessors=(
            Spec("binning", {"n_bins": 8, "bin_strategy": "uniform", "across_features": False}),
        ),
        habitat_model_fitter=Spec(
            "kmeans",
            # Pooling units are per-VOXEL (~17k rows): silhouette_score is
            # O(n^2) (pairwise distance matrix, GB-scale, minutes per k).
            # calinski_harabasz is O(n) and keeps k-selection working.
            {"min_habitats": 2, "max_habitats": 3, "validation": "elbow", "n_init": 3},
        ),
        habitat_assigner=Spec("nearest_centroid"),
        habitat_features=(Spec("volume"),),
        random_seed=42,
    )

    # Subset cohort view (keeps the first N_SUBJECTS in cohort order).
    from habit.contracts import Cohort

    cohort = Cohort(subjects=subjects, name=f"{cohort.name or 'cohort'}_first{N_SUBJECTS}")

    print("\n=== SERIAL ===", flush=True)
    t0 = time.perf_counter()
    serial_result = recipes.direct_pooling(cohort, spec)
    t_serial = time.perf_counter() - t0
    t0 = time.perf_counter()
    serial_dir = serial_result.save(
        str(OUT_ROOT / "serial"),
        map_format="nii.gz",
        write_cluster_plots=_WRITE_PLOTS,
        write_cluster_plots_3d=_WRITE_PLOTS,
        write_interactive_cluster_plots=_WRITE_PLOTS,
    )
    s_serial = time.perf_counter() - t0
    print(
        f"SERIAL compute {t_serial:.2f}s + save {s_serial:.2f}s | "
        f"habitats={serial_result.habitat_model.n_habitats} -> {serial_dir}",
        flush=True,
    )

    print("\n=== PARALLEL ===", flush=True)
    # Must be a bool. A non-empty string like "False" is truthy in Python and
    # would still enable GPU-pool capping (1 GPU -> workers forced to 1).
    # Set True on single-GPU hosts if you want to avoid multiple CUDA contexts.
    cap_gpu = False
    backend = ProcessPoolBackend.from_policy(
        RunPolicy(
            workers=workers,
            backend="process",
            on_subject_failure="continue",
            parallel_mode="persistent",
            cap_workers_to_gpu_pool=cap_gpu,
            # Deterministic geometry failure: retrying just re-spawns the pool.
            auto_retry_rounds=0,
        )
    )
    print(f"effective_workers={backend.workers}", flush=True)
    t0 = time.perf_counter()
    parallel_result = recipes.direct_pooling(cohort, spec, backend=backend)
    t_parallel = time.perf_counter() - t0
    t0 = time.perf_counter()
    parallel_dir = parallel_result.save(
        str(OUT_ROOT / "parallel"),
        map_format="nii.gz",
        write_cluster_plots=_WRITE_PLOTS,
        write_cluster_plots_3d=_WRITE_PLOTS,
        write_interactive_cluster_plots=_WRITE_PLOTS,
    )
    s_parallel = time.perf_counter() - t0
    print(
        f"PARALLEL compute {t_parallel:.2f}s + save {s_parallel:.2f}s | "
        f"habitats={parallel_result.habitat_model.n_habitats} -> {parallel_dir}",
        flush=True,
    )

    print("\n--- summary ---")
    print(f"serial   : compute {t_serial:.2f} s  (save {s_serial:.2f} s)")
    print(
        f"parallel : compute {t_parallel:.2f} s  "
        f"(save {s_parallel:.2f} s, workers={backend.workers})"
    )
    if t_parallel > 0:
        print(f"speedup  : {t_serial / t_parallel:.2f}x")
    print(f"model_id serial={serial_result.habitat_model.model_id}")
    print(f"model_id parallel={parallel_result.habitat_model.model_id}")
