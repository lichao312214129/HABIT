"""
Parallel and checkpoints
========================

:class:`~habit.spec.HabitatSpec` declares what to compute;
:class:`~habit.spec.RunPolicy` declares how to schedule it. Pass a
backend from :func:`~habit.execution.backend_from_policy` into
:meth:`~habit.recipes.Study.fit_predict`. With a fixed ``random_seed``
the maps match serial execution.

.. important::
   **Windows + process pool:** spawning workers re-imports your script.
   Put the run inside ``if __name__ == "__main__":`` (already done below).
   See also :doc:`/tutorial/quickstart`.

Backend selection (timeout uncoupled from spawn)
------------------------------------------------

:func:`~habit.execution.backend_from_policy` selects
:class:`~habit.execution.ProcessPoolBackend` when any of:

* ``backend == "process"``
* ``workers > 1``
* ``parallel_mode == "isolated"``

A positive ``subject_timeout_sec`` alone does **not** force spawn (the
library default is ``900.0``). True in-process serial is simply
``RunPolicy(workers=1, backend="serial")``.

Why parallel can look *slower* on a tiny demo cohort
----------------------------------------------------

On a 5-subject demo with a light ``raw`` two-step spec, process-pool
wall time is often **higher** than serial. Typical causes (not a bug in
the habitat definition):

* **Spawn overhead (Windows ``spawn`` / cold Linux workers):** each
  worker re-imports HABIT / NumPy / SimpleITK before any subject runs.
* **Tiny work per subject:** when per-subject compute is a few seconds,
  pool startup dominates Amdahl's law.
* **Oversubscription:** ``workers`` larger than useful CPU cores, or
  several workers contending for one GPU when a texture / GPU path is
  active. Prefer a **CPU-only** fair comparison below
  (``CUDA_VISIBLE_DEVICES=""``). On a single-GPU host HABIT defaults to
  CPU radiomics for worker slots beyond 0 (set
  ``HABIT_GPU_OVERSUBSCRIBE=wrap`` to share the card).
* **Isolated mode** pays spawn cost **per subject** — useful for
  timeout / OOM isolation, not for small demos.

When per-subject work is realistic (larger volumes, dense voxel texture
or more supervoxels / habitat search), acceleration and parallelism shine:

* **GPU dense voxel texture extraction:** cuts 3D texture computation
  (90 features across 54,913 ROI voxels) from **19.85s** on pure CPU (Route A)
  to **1.70s** with hybrid GPU (Route B) and **0.71s** with full end-to-end
  GPU (Route C, **~28× acceleration**).
* **GPU cohort acceleration and process-pool parallelism:** on a 16-subject
  cohort (~878k ROI voxels, 90 features), GPU feature extraction slashes wall time
  from **263.5s** on serial CPU (3.6 subjects/min) down to **11.6s** on multi-GPU
  (82.7 subjects/min, **~23× cohort speedup** vs serial CPU, **4.2× faster** than
  an 8-worker CPU pool).

Pick a backend:

* Debug one subject: ``RunPolicy(workers=1, backend="serial", ...)``
* Cohort default: ``RunPolicy(workers=2, backend="process", ...)``
* Fresh process per subject: same, plus ``parallel_mode="isolated"``
"""

# %%
# Shared two-step CPU-only spec and a small demo cohort.
from __future__ import annotations

import os
from pathlib import Path
import shutil
import time
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

# Fair CPU comparison: hide GPUs from this process and its children.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.execution import CheckpointStore, backend_from_policy
from habit.recipes import Study
from habit.spec import HabitatSpec, RunPolicy, Spec, Stage
from habit.viz import plot_habitat_overlay

DATA = fetch_demo()
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=("LAP",), roi=ROI)[:5]
Path("out").mkdir(exist_ok=True)
print(f"Cohort: {list(cohort.subject_ids)}")
print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}")

# Module-level operators so ProcessPoolBackend can pickle them under
# Windows ``spawn`` (nested classes inside ``__main__`` fail to unpickle).


class _BoomOnSubject:
    """Fail only for ``fail_subject_id`` (fault-tolerance demo)."""

    def __init__(self, fail_subject_id: str) -> None:
        self.fail_subject_id = str(fail_subject_id)

    def __call__(self, subject):  # type: ignore[no-untyped-def]
        if subject.subject_id == self.fail_subject_id:
            raise RuntimeError("demo intentional failure")
        return subject.subject_id


class _SlowSubject:
    """Sleep only for ``slow_subject_id`` (timeout demo)."""

    def __init__(self, slow_subject_id: str) -> None:
        self.slow_subject_id = str(slow_subject_id)

    def __call__(self, subject):  # type: ignore[no-untyped-def]
        if subject.subject_id == self.slow_subject_id:
            time.sleep(30.0)
        return subject.subject_id

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
# Serial baseline (true in-process: workers=1 + backend serial).
if __name__ == "__main__":
    policy_serial = RunPolicy(
        workers=1,
        backend="serial",
        on_subject_failure="continue",
        resume=False,
    )
    t0 = time.perf_counter()
    serial_result = Study(spec=raw_spec).fit_predict(
        cohort, backend=backend_from_policy(policy_serial)
    )
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
        subject_timeout_sec=None,
        on_subject_failure="continue",
        parallel_mode="persistent",
        resume=False,
    )
    t0 = time.perf_counter()
    persistent_result = Study(spec=raw_spec).fit_predict(
        cohort, backend=backend_from_policy(policy_persistent)
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
# Isolated process pool (fresh worker per subject — higher spawn cost).
if __name__ == "__main__":
    policy_isolated = RunPolicy(
        workers=2,
        backend="process",
        subject_timeout_sec=None,
        on_subject_failure="continue",
        parallel_mode="isolated",
        resume=False,
    )
    t0 = time.perf_counter()
    isolated_result = Study(spec=raw_spec).fit_predict(
        cohort, backend=backend_from_policy(policy_isolated)
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
    print("CPU-only scheduler timing (demo cohort, light raw spec):")
    print(scheduler_table.to_string(index=False))
    if persistent_s > serial_s:
        print(
            "Note: parallel > serial here is expected on a tiny cohort — "
            "spawn/import overhead dominates per-subject work. "
            "See the cloud table below for a heavier workload where "
            "workers=4–8 beat serial."
        )
    scheduler_table

# %%
# Fault tolerance: one subject raises; ``on_subject_failure="continue"``
# keeps the others.
if __name__ == "__main__":
    from habit.contracts.ops import SubjectResult
    from habit.execution.backends import SerialBackend

    backend_ft = SerialBackend(on_subject_failure="continue", resume=False)
    outcomes: List[SubjectResult] = list(
        backend_ft.map(_BoomOnSubject(cohort[1].subject_id), list(cohort[:3]))
    )
    fault_table = pd.DataFrame(
        [
            {
                "subject_id": r.subject_id,
                "ok": r.error is None,
                "value": r.value,
                "error": None if r.error is None else type(r.error).__name__,
            }
            for r in outcomes
        ]
    )
    print("Fault isolation (continue):")
    print(fault_table.to_string(index=False))
    fault_table

# %%
# Time control: process backend + short ``subject_timeout_sec``.
# A sleeping subject is marked failed; others still finish.
if __name__ == "__main__":
    _SLOW_SUBJECT_ID = cohort[1].subject_id
    policy_timeout = RunPolicy(
        workers=2,
        backend="process",
        subject_timeout_sec=2.0,
        subject_spawn_timeout_sec=60.0,
        graceful_shutdown_sec=2.0,
        on_subject_failure="continue",
        parallel_mode="isolated",
        resume=False,
    )
    timeout_backend = backend_from_policy(policy_timeout)
    timeout_outcomes: List[SubjectResult] = list(
        timeout_backend.map(_SlowSubject(_SLOW_SUBJECT_ID), list(cohort[:3]))
    )
    timeout_table = pd.DataFrame(
        [
            {
                "subject_id": r.subject_id,
                "ok": r.error is None,
                "error": None if r.error is None else str(r.error)[:80],
            }
            for r in timeout_outcomes
        ]
    )
    print("Per-subject timeout (2s):")
    print(timeout_table.to_string(index=False))
    timeout_table

# %%
# Checkpoint resume: first run writes successes; second run skips them.
if __name__ == "__main__":
    ckpt_dir = Path("out") / "parallel_guide_ckpt"
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    store = CheckpointStore(ckpt_dir)
    policy_ckpt = RunPolicy(
        workers=1,
        backend="serial",
        on_subject_failure="continue",
        resume=True,
    )
    backend_ckpt = backend_from_policy(policy_ckpt)
    t0 = time.perf_counter()
    first = Study(spec=raw_spec).fit_predict(
        cohort[:2], backend=backend_ckpt, checkpoint=store
    )
    first_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    second = Study(spec=raw_spec).fit_predict(
        cohort[:2], backend=backend_ckpt, checkpoint=store
    )
    second_s = time.perf_counter() - t0
    resume_table = pd.DataFrame(
        [
            {"run": "cold", "wall_time_s": first_s, "n_maps": len(first.habitat_maps)},
            {
                "run": "resume_same_checkpoint",
                "wall_time_s": second_s,
                "n_maps": len(second.habitat_maps),
            },
        ]
    )
    print("Checkpoint resume:")
    print(resume_table.to_string(index=False))
    resume_table

# %%
# Three tiers of voxel texture acceleration and numerical parity.
#
# Hardware (AutoDL west, 2026-09-04): NVIDIA GeForce RTX 4080 SUPER
# (32 GiB), Intel Xeon Platinum 8352V CPU.
# Workload: :class:`~habit.voxel_features.voxel_radiomics.VoxelRadiomicsFeatures`
# on a large lesion (54,913 ROI tumor voxels, 3D volume shape 80×80×48),
# extracting 90 three-dimensional radiomic texture features (GLCM, firstorder).
#
# HABIT supports three distinct execution tiers:
#
# 1. **Route A (Pure CPU PyRadiomics):** Single-threaded C matrix construction +
#    CPU NumPy feature quantification.
# 2. **Route B (Hybrid GPU):** PyRadiomics C matrix construction on CPU +
#    batch tensor evaluation via TorchRadiomics on GPU.
# 3. **Route C (Full End-to-End GPU):** HABIT built-in GPU matrix construction
#    (``gpumatrices``) + GPU TorchRadiomics tensor evaluation (zero H2D transfer).
#
# **Scientific parity** (feature columns name-aligned; shape ``(54913, 90)``):
# * Route B vs Route C: **bit-identical** (``max_abs_diff = 0``, ``max_rel_diff = 0``).
#   Built-in ``gpumatrices`` matches PyRadiomics C matrix construction exactly.
# * Route A vs B / A vs C: ``max_abs_diff = 0.5`` on Energy / TotalEnergy
#   (~2.45e6 → relative ~2e-7). For ``|value| >= 1e-3``, worst relative is
#   ~1.3e-2 (Skewness near small mid-ROI values). Mean abs over all cells is
#   ~0.00137. ``np.allclose(..., rtol=1e-4, atol=1e-4)`` holds. Float32
#   radiomics rounding, not a definition change.
if __name__ == "__main__":
    three_route_bench = pd.DataFrame(
        [
            {
                "route": "Route A: Pure CPU (PyRadiomics)",
                "matrix_construction": "CPU (PyRadiomics C)",
                "feature_quantification": "CPU (PyRadiomics)",
                "roi_voxels": 54913,
                "n_features": 90,
                "wall_s": 19.85,
                "speedup_vs_A": 1.00,
                "speedup_vs_B": 0.09,
            },
            {
                "route": "Route B: Hybrid GPU",
                "matrix_construction": "CPU (PyRadiomics C)",
                "feature_quantification": "GPU (TorchRadiomics)",
                "roi_voxels": 54913,
                "n_features": 90,
                "wall_s": 1.70,
                "speedup_vs_A": 11.66,
                "speedup_vs_B": 1.00,
            },
            {
                "route": "Route C: Full End-to-End GPU",
                "matrix_construction": "GPU (gpumatrices)",
                "feature_quantification": "GPU (TorchRadiomics)",
                "roi_voxels": 54913,
                "n_features": 90,
                "wall_s": 0.71,
                "speedup_vs_A": 28.06,
                "speedup_vs_B": 2.41,
            },
        ]
    )
    print("Three-tier voxel texture acceleration (54,913 ROI voxels, 90 features):")
    print(
        three_route_bench[
            [
                "route",
                "matrix_construction",
                "feature_quantification",
                "wall_s",
                "speedup_vs_A",
                "speedup_vs_B",
            ]
        ].to_string(index=False)
    )

    parity_table = pd.DataFrame(
        [
            {
                "comparison": "A vs B (CPU vs hybrid GPU)",
                "max_abs_diff": 0.5,
                "max_rel_diff": 1.26e-2,
                "note": "Abs peak Energy; rel peak Skewness (|v|>=1e-3)",
            },
            {
                "comparison": "A vs C (CPU vs full GPU)",
                "max_abs_diff": 0.5,
                "max_rel_diff": 1.26e-2,
                "note": "Same float32 rounding as A vs B (B==C)",
            },
            {
                "comparison": "B vs C (hybrid vs full GPU)",
                "max_abs_diff": 0.0,
                "max_rel_diff": 0.0,
                "note": "Bit-identical after name alignment",
            },
        ]
    )
    print("\nNumerical parity (54,913 voxels x 90 features; columns name-aligned):")
    print(parity_table.to_string(index=False))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0))

    routes = ["Route A\nPure CPU", "Route B\nHybrid GPU", "Route C\nFull GPU"]
    times = [19.85, 1.70, 0.71]
    colors = ["#7f7f7f", "#1f77b4", "#2ca02c"]

    bars1 = ax1.bar(routes, times, color=colors, width=0.55)
    ax1.set_ylabel("wall time (s)")
    ax1.set_title("Voxel texture extraction wall time\n(54,913 ROI voxels, 90 features)")
    ax1.set_ylim(0, 24)
    ax1.grid(axis="y", alpha=0.3)
    for bar in bars1:
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.4,
            f"{h:.2f}s",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    speedups = [1.0, 11.66, 28.06]
    bars2 = ax2.bar(routes, speedups, color=colors, width=0.55)
    ax2.set_ylabel("speedup vs CPU (x)")
    ax2.set_title("Acceleration factor vs Pure CPU\n(higher is faster)")
    ax2.set_ylim(0, 34)
    ax2.grid(axis="y", alpha=0.3)
    for bar in bars2:
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.6,
            f"{h:.1f}x",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig("out/parallel_voxel_radiomics_speedup.png", dpi=150, bbox_inches="tight")
    plt.show()
    three_route_bench

# %%
# Cloud multi-CPU / multi-GPU timings across a 16-subject cohort.
#
# Hardware (AutoDL west, 2026-09-04): 5× NVIDIA GeForce RTX 4080 SUPER
# (32 GiB each), 144 logical CPUs (2× Xeon Platinum 8352V), ~503 GiB RAM.
# Workload: Dense 3D voxel texture feature extraction (``voxel_radiomics``)
# across synthetic cohort n=16, shape 80×80×48 (~54,913 ROI voxels/case,
# total ~878,608 ROI voxels across cohort, extracting 90 features per voxel).
# Re-run::
#
#   python scripts/run_multi_gpu_cohort_bench.py
#
# Root cause of previous benchmark:
# Earlier runs used ``Spec("raw")`` followed by CPU k-means / MSI, which was
# 100% CPU computation with 0% GPU utilization (GPUs remained idle). That caused
# 5 GPUs to track 0 GPU multi-CPU with only ~1.7×-1.9× speedup.
#
# True GPU acceleration in HABIT:
# HABIT's true GPU engine is ``voxel_radiomics`` (TorchRadiomics + CUDA
# ``gpumatrices``). On dense 3D voxel textures (90 features across 54,913 voxels):
# * Pure CPU (PyRadiomics) takes 263.5s on serial CPU (3.64 subjects/min).
# * 8 CPU workers take 49.2s (19.53 subjects/min).
# * Single RTX 4080 SUPER cuts wall time to 19.3s (49.73 subjects/min, 13.65× speedup vs CPU).
# * 5× RTX 4080 SUPER drops wall time down to 11.6s (82.68 subjects/min, 22.70× speedup vs CPU serial,
#   and 4.23× faster than 8 CPU workers).
#
# Worker count sweet spot on small cohorts:
# On 16 subjects, 2 workers on 5 GPUs (11.61s) beat 5 workers (14.78s).
# Because single-subject GPU compute is so fast (~0.71s, ~11.4s total compute),
# going from 2 to 5 workers saves only ~2.9s of raw computation, which is
# outweighed by spawning 5 separate processes, initializing 5 CUDA primary
# contexts, and modulo load imbalance (16 is not divisible by 5).
# For small cohorts (<50 subjects), 2–4 workers are optimal; for large
# production cohorts (100+ subjects) where compute dwarfs startup, all 5 GPUs
# deliver linear multi-worker scaling.
if __name__ == "__main__":
    cloud_table = pd.DataFrame(
        [
            {
                "scenario": "0gpu_multicpu",
                "device": "CUDA=-1",
                "workers": 1,
                "wall_s": 263.49,
                "subjects_per_min": 3.64,
                "speedup_vs_w1": 1.00,
                "speedup_vs_cpu_serial": 1.00,
            },
            {
                "scenario": "0gpu_multicpu",
                "device": "CUDA=-1",
                "workers": 2,
                "wall_s": 138.03,
                "subjects_per_min": 6.96,
                "speedup_vs_w1": 1.91,
                "speedup_vs_cpu_serial": 1.91,
            },
            {
                "scenario": "0gpu_multicpu",
                "device": "CUDA=-1",
                "workers": 4,
                "wall_s": 79.23,
                "subjects_per_min": 12.12,
                "speedup_vs_w1": 3.33,
                "speedup_vs_cpu_serial": 3.33,
            },
            {
                "scenario": "0gpu_multicpu",
                "device": "CUDA=-1",
                "workers": 8,
                "wall_s": 49.16,
                "subjects_per_min": 19.53,
                "speedup_vs_w1": 5.36,
                "speedup_vs_cpu_serial": 5.36,
            },
            {
                "scenario": "1gpu_multicpu",
                "device": "CUDA=0",
                "workers": 1,
                "wall_s": 19.30,
                "subjects_per_min": 49.73,
                "speedup_vs_w1": 1.00,
                "speedup_vs_cpu_serial": 13.65,
            },
            {
                "scenario": "1gpu_multicpu",
                "device": "CUDA=0",
                "workers": 2,
                "wall_s": 17.44,
                "subjects_per_min": 55.04,
                "speedup_vs_w1": 1.11,
                "speedup_vs_cpu_serial": 15.11,
            },
            {
                "scenario": "1gpu_multicpu",
                "device": "CUDA=0",
                "workers": 4,
                "wall_s": 24.67,
                "subjects_per_min": 38.91,
                "speedup_vs_w1": 0.78,
                "speedup_vs_cpu_serial": 10.68,
            },
            {
                "scenario": "5gpu_multicpu",
                "device": "CUDA=0,1,2,3,4",
                "workers": 1,
                "wall_s": 18.47,
                "subjects_per_min": 51.96,
                "speedup_vs_w1": 1.00,
                "speedup_vs_cpu_serial": 14.27,
            },
            {
                "scenario": "5gpu_multicpu",
                "device": "CUDA=0,1,2,3,4",
                "workers": 2,
                "wall_s": 11.61,
                "subjects_per_min": 82.68,
                "speedup_vs_w1": 1.59,
                "speedup_vs_cpu_serial": 22.70,
            },
            {
                "scenario": "5gpu_multicpu",
                "device": "CUDA=0,1,2,3,4",
                "workers": 4,
                "wall_s": 14.05,
                "subjects_per_min": 68.32,
                "speedup_vs_w1": 1.31,
                "speedup_vs_cpu_serial": 18.75,
            },
            {
                "scenario": "5gpu_multicpu",
                "device": "CUDA=0,1,2,3,4",
                "workers": 5,
                "wall_s": 14.78,
                "subjects_per_min": 64.97,
                "speedup_vs_w1": 1.25,
                "speedup_vs_cpu_serial": 17.83,
            },
        ]
    )
    print(
        "Cloud timings (5× RTX 4080 SUPER, 144 CPUs; "
        "n=16 synthetic 80×80×48, ~54.9k ROI voxels/case, 90 features):"
    )
    print(cloud_table.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for scenario, group in cloud_table.groupby("scenario", sort=False):
        ax.plot(
            group["workers"],
            group["wall_s"],
            marker="o",
            label=str(scenario),
        )
    ax.set_xlabel("workers")
    ax.set_ylabel("wall time (s)")
    ax.set_title("Cohort voxel texture acceleration (16 subjects, 90 features)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("out/parallel_cloud_speedup.png", dpi=150, bbox_inches="tight")
    plt.show()
    cloud_table
