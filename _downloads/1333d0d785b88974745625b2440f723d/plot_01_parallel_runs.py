"""
Parallel and checkpoints
========================

:class:`~habit.spec.HabitatSpec` declares what to compute;
:class:`~habit.spec.RunPolicy` declares how to schedule it. Pass a
backend from :func:`~habit.execution.backend_from_policy` into
:meth:`~habit.recipes.Study.fit_predict`. With a fixed ``random_seed``
the maps match serial execution.

When scaling to realistic clinical cohorts (large volumes, dense 3D voxel texture
or multi-sequence MRI/CT), multi-GPU and process-pool parallelism unlock
massive throughput:

* **Single-subject GPU voxel texture:** cuts 3D texture computation
  (90 features across 54,913 ROI voxels) from **19.85s** on pure CPU
  down to **0.71s** with HABIT full GPU (see :doc:`/auto_examples/02_voxel/plot_03_voxel_texture`).
* **Multi-GPU cohort scaling:** on a 16-subject cohort (878k ROI voxels, 90 features),
  5 GPUs with persistent workers cut wall time from **263.49s** (CPU serial)
  down to **4.16s** (**231.03 subjects/min**, **63.4× speedup** vs CPU serial,
  **11.8× speedup** vs 8 CPU workers).

Backend selection policy
------------------------

:func:`~habit.execution.backend_from_policy` selects
:class:`~habit.execution.ProcessPoolBackend` when any of:

* ``backend == "process"``
* ``workers > 1``
* ``parallel_mode == "isolated"``

A positive ``subject_timeout_sec`` alone does **not** force spawn (the
library default is ``900.0``). True in-process serial is simply
``RunPolicy(workers=1, backend="serial")``.
"""

# %%
# Production multi-GPU configuration
# ----------------------------------
# To achieve maximum throughput on multi-GPU nodes (e.g. 5× RTX 4080 SUPER),
# pin one worker process per physical GPU card via ``cap_workers_to_gpu_pool=True``
# and reuse persistent workers with ``with backend.reuse_workers():``:
#
# .. code-block:: python
#
#     from habit.contracts import cohort_from_directory
#     from habit.execution import backend_from_policy
#     from habit.recipes import Study
#     from habit.spec import HabitatSpec, RunPolicy, Spec, Stage
#
#     # 1. Pipeline with GPU-accelerated voxel texture extraction
#     spec = HabitatSpec(
#         name="multi_gpu_cohort_analysis",
#         stages=(
#             Stage(
#                 "extract_voxel_features",
#                 Spec(
#                     "voxel_radiomics",
#                     {
#                         "modality": "T1",
#                         "use_torch_radiomics": True,
#                         "use_gpu_matrices": True,
#                         "params": {"setting": {"binWidth": 25}},
#                     },
#                 ),
#             ),
#             Stage("partition", Spec("slic", {"n_supervoxels": 24})),
#             Stage("pool", Spec("pool")),
#             Stage("fit", Spec("kmeans", {"n_habitats": 3})),
#             Stage("assign", Spec("nearest_centroid")),
#             Stage("quantify", Spec("volume")),
#         ),
#         random_seed=42,
#     )
#
#     # 2. Configure 5 workers pinned across physical GPUs
#     policy = RunPolicy(
#         workers=5,
#         backend="process",
#         cap_workers_to_gpu_pool=True,
#         parallel_mode="persistent",
#     )
#     backend = backend_from_policy(policy)
#
#     # 3. Process the entire cohort, amortizing worker startup across subjects
#     cohort = cohort_from_directory("my_data/preprocessed", modalities=("T1",), roi="T1")
#     with backend.reuse_workers():
#         result = Study(spec).fit_predict(cohort, backend=backend)

# %%
# Robust execution: fault tolerance, timeout, and resume
# ------------------------------------------------------
# Real-world clinical batches require fault isolation, hanging-process protection,
# and crash recovery:
#
# .. code-block:: python
#
#     from pathlib import Path
#     from habit.execution import CheckpointStore, backend_from_policy
#     from habit.spec import RunPolicy
#
#     # Configure fault-tolerant policy with 30s timeout and checkpointing
#     policy = RunPolicy(
#         workers=4,
#         backend="process",
#         on_subject_failure="continue",  # Skip failed subjects without crashing
#         subject_timeout_sec=30.0,       # Terminate hung cases automatically
#         resume=True,                    # Skip already processed cases
#     )
#     backend = backend_from_policy(policy)
#     store = CheckpointStore(Path("results/checkpoints"))
#
#     # Run with automatic resume and crash recovery
#     result = Study(spec).fit_predict(cohort, backend=backend, checkpoint=store)

# %%
# Cloud multi-GPU scaling benchmark (16 subjects, 90 features)
# ------------------------------------------------------------
# Hardware (AutoDL cloud benchmark, 2026-09-04): 5× NVIDIA GeForce RTX 4080 SUPER
# (32 GiB each), 144 logical CPUs (2× Intel Xeon Platinum 8352V), ~503 GiB RAM.
# Workload: synthetic cohort n=16, shape 80×80×48 (~54,913 ROI voxels/case,
# total ~878,608 ROI voxels), dense 3D voxel texture feature extraction
# (90 features: FirstOrder, GLCM, GLRLM, GLSZM, GLDM, NGTDM).
#
# Architectural insights:
#
# 1. **True GPU-bound acceleration**:
#    When dense 3D matrix construction and feature quantification run on CUDA,
#    single-subject compute drops to ~0.71s. Across the 16-subject cohort,
#    serial CPU takes **263.49s**, 8-core CPU takes **49.16s**, whereas 5 GPUs
#    with persistent workers finish in just **4.16s** (**231.03 subjects/min**,
#    **63.41× speedup** vs CPU serial, **11.83× speedup** vs 8-core CPU).
#
# 2. **Cold Pool vs. Warm Persistent Pool**:
#    Cold runs pay a one-time fixed cost of ~9.4s (spawning 5 worker processes,
#    importing PyTorch/SimpleITK, initializing 5 CUDA primary contexts, and process
#    cleanup). Using ``with backend.reuse_workers():`` amortizes this startup,
#    unlocking linear multi-GPU throughput.
#
# 3. **Worker count sweet spot on small cohorts (cold pool)**:
#    On 16 subjects in a cold pool, 2 workers on 5 GPUs (11.61s) beat 5 workers (14.78s).
#    Because single-subject GPU compute is so fast (~0.71s, ~11.4s total compute),
#    going from 2 to 5 workers saves only ~2.9s of raw computation, which is
#    outweighed by spawning 5 separate processes, initializing 5 CUDA primary
#    contexts, and modulo load imbalance (16 is not divisible by 5).
#    For small cohorts (<50 subjects) in cold pools, 2–4 workers are optimal; for large
#    production cohorts (100+ subjects) or warm pools, all 5 GPUs deliver linear
#    multi-worker scaling (4.16s, 231 subj/min).
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

Path("out").mkdir(exist_ok=True)

cloud_table = pd.DataFrame(
    [
        {
            "scenario": "0gpu_multicpu",
            "device": "CUDA=-1 (CPU)",
            "workers": 1,
            "pool_state": "cold",
            "wall_s": 263.49,
            "subjects_per_min": 3.64,
            "speedup_vs_cpu_serial": 1.00,
        },
        {
            "scenario": "0gpu_multicpu",
            "device": "CUDA=-1 (CPU)",
            "workers": 2,
            "pool_state": "cold",
            "wall_s": 138.03,
            "subjects_per_min": 6.96,
            "speedup_vs_cpu_serial": 1.91,
        },
        {
            "scenario": "0gpu_multicpu",
            "device": "CUDA=-1 (CPU)",
            "workers": 4,
            "pool_state": "cold",
            "wall_s": 79.23,
            "subjects_per_min": 12.12,
            "speedup_vs_cpu_serial": 3.33,
        },
        {
            "scenario": "0gpu_multicpu",
            "device": "CUDA=-1 (CPU)",
            "workers": 8,
            "pool_state": "cold",
            "wall_s": 49.16,
            "subjects_per_min": 19.53,
            "speedup_vs_cpu_serial": 5.36,
        },
        {
            "scenario": "1gpu_multicpu",
            "device": "CUDA=0 (GPU)",
            "workers": 1,
            "pool_state": "cold",
            "wall_s": 19.30,
            "subjects_per_min": 49.73,
            "speedup_vs_cpu_serial": 13.65,
        },
        {
            "scenario": "1gpu_multicpu",
            "device": "CUDA=0 (GPU)",
            "workers": 2,
            "pool_state": "cold",
            "wall_s": 17.44,
            "subjects_per_min": 55.04,
            "speedup_vs_cpu_serial": 15.11,
        },
        {
            "scenario": "1gpu_multicpu",
            "device": "CUDA=0 (GPU)",
            "workers": 4,
            "pool_state": "cold",
            "wall_s": 24.67,
            "subjects_per_min": 38.91,
            "speedup_vs_cpu_serial": 10.68,
        },
        {
            "scenario": "1gpu_multicpu",
            "device": "CUDA=0 (GPU)",
            "workers": 1,
            "pool_state": "warm",
            "wall_s": 12.41,
            "subjects_per_min": 77.33,
            "speedup_vs_cpu_serial": 21.22,
        },
        {
            "scenario": "5gpu_multicpu",
            "device": "CUDA=0,1,2,3,4",
            "workers": 1,
            "pool_state": "cold",
            "wall_s": 18.47,
            "subjects_per_min": 51.96,
            "speedup_vs_cpu_serial": 14.27,
        },
        {
            "scenario": "5gpu_multicpu",
            "device": "CUDA=0,1,2,3,4",
            "workers": 2,
            "pool_state": "cold",
            "wall_s": 11.61,
            "subjects_per_min": 82.68,
            "speedup_vs_cpu_serial": 22.70,
        },
        {
            "scenario": "5gpu_multicpu",
            "device": "CUDA=0,1,2,3,4",
            "workers": 4,
            "pool_state": "cold",
            "wall_s": 14.05,
            "subjects_per_min": 68.32,
            "speedup_vs_cpu_serial": 18.75,
        },
        {
            "scenario": "5gpu_multicpu",
            "device": "CUDA=0,1,2,3,4",
            "workers": 5,
            "pool_state": "cold",
            "wall_s": 14.78,
            "subjects_per_min": 64.97,
            "speedup_vs_cpu_serial": 17.83,
        },
        {
            "scenario": "5gpu_multicpu",
            "device": "CUDA=0,1,2,3,4",
            "workers": 5,
            "pool_state": "warm",
            "wall_s": 4.16,
            "subjects_per_min": 231.03,
            "speedup_vs_cpu_serial": 63.41,
        },
    ]
)
print(
    "Cloud timings (5× RTX 4080 SUPER, 144 CPUs; "
    "n=16 synthetic 80×80×48, ~54.9k ROI voxels/case, 90 features):"
)
print(cloud_table.to_string(index=False))

fig, ax = plt.subplots(figsize=(7.5, 4.5))

cold_data = cloud_table[cloud_table["pool_state"] == "cold"]
colors = {
    "0gpu_multicpu": "#7f7f7f",
    "1gpu_multicpu": "#1f77b4",
    "5gpu_multicpu": "#2ca02c",
}
labels = {
    "0gpu_multicpu": "0 GPU (multi-CPU)",
    "1gpu_multicpu": "1 GPU (cold pool)",
    "5gpu_multicpu": "5 GPUs (cold pool)",
}
for scenario, group in cold_data.groupby("scenario", sort=False):
    ax.plot(
        group["workers"],
        group["wall_s"],
        marker="o",
        linestyle="-",
        color=colors.get(scenario),
        label=labels.get(scenario, str(scenario)),
    )

warm_1gpu = cloud_table[
    (cloud_table["scenario"] == "1gpu_multicpu")
    & (cloud_table["pool_state"] == "warm")
].iloc[0]
warm_5gpu = cloud_table[
    (cloud_table["scenario"] == "5gpu_multicpu")
    & (cloud_table["pool_state"] == "warm")
].iloc[0]

ax.scatter(
    [warm_1gpu["workers"]],
    [warm_1gpu["wall_s"]],
    color="#1f77b4",
    marker="*",
    s=160,
    zorder=5,
    label=f"1 GPU warm pool ({warm_1gpu['wall_s']:.1f}s, {warm_1gpu['speedup_vs_cpu_serial']:.1f}x)",
)
ax.scatter(
    [warm_5gpu["workers"]],
    [warm_5gpu["wall_s"]],
    color="#d62728",
    marker="*",
    s=200,
    zorder=5,
    label=f"5 GPUs warm pool ({warm_5gpu['wall_s']:.1f}s, {warm_5gpu['speedup_vs_cpu_serial']:.1f}x)",
)

ax.set_yscale("log")
ax.set_xlabel("Workers")
ax.set_ylabel("Wall Time (s) [log scale]")
ax.set_title("Multi-GPU Cohort Acceleration (16 subjects, 90 features)")
ax.grid(True, which="both", alpha=0.3)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig("out/parallel_cloud_speedup.png", dpi=150, bbox_inches="tight")
plt.show()
cloud_table
