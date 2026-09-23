# sphinx_gallery_thumbnail_path = '_static/images/examples/parallel_cloud_speedup.png'
"""
Running a cohort on several GPUs
================================

``cap_workers_to_gpu_pool=True`` clamps ``workers`` to the visible GPU
count, one worker per card. Pair it with ``reuse_workers()`` so CUDA
startup is paid once. This page records a cloud run. It does not
repeat that workload when the docs are built.
"""

# %%
# The policy a 5-GPU node uses. Building it does not launch workers.
from habit.execution import backend_from_policy
from habit.spec import RunPolicy

policy = RunPolicy(
    workers=5,
    backend="process",
    parallel_mode="persistent",
    cap_workers_to_gpu_pool=True,
    on_subject_failure="fail_fast",
)
backend = backend_from_policy(policy)
print(
    type(backend).__name__,
    "requested_workers=5",
    f"cap={backend.policy.cap_workers_to_gpu_pool}",
)

# %%
# Recorded AutoDL timings (2026-09-04)
# ------------------------------------
# 5× NVIDIA GeForce RTX 4080 SUPER (32 GiB), 144 logical CPUs, ~503 GiB RAM.
# Synthetic cohort n=16, shape 80×80×48, 90 voxel-texture features.
# Warm rows used ``with backend.reuse_workers():``.
# Full driver: ``scripts/run_multi_gpu_cohort_bench.py``.
#
# .. list-table::
#    :header-rows: 1
#    :widths: 18 22 10 12 12 16 16
#
#    * - Scenario
#      - Device
#      - Workers
#      - Pool
#      - Wall (s)
#      - Subjects/min
#      - Speedup vs CPU serial
#    * - 0 GPU
#      - CUDA=-1 (CPU)
#      - 1
#      - cold
#      - 263.49
#      - 3.64
#      - 1.00×
#    * - 0 GPU
#      - CUDA=-1 (CPU)
#      - 8
#      - cold
#      - 49.16
#      - 19.53
#      - 5.36×
#    * - 1 GPU
#      - CUDA=0
#      - 1
#      - warm
#      - 12.41
#      - 77.33
#      - 21.22×
#    * - 5 GPUs
#      - CUDA=0,1,2,3,4
#      - 5
#      - cold
#      - 14.78
#      - 64.97
#      - 17.83×
#    * - 5 GPUs
#      - CUDA=0,1,2,3,4
#      - 5
#      - warm
#      - 4.16
#      - 231.03
#      - 63.41×
#
# .. figure:: /_static/images/examples/parallel_cloud_speedup.png
#    :alt: Multi-GPU cohort wall time versus worker count
#
#    Recorded AutoDL result (16 subjects, 90 features). Stars are warm
#    persistent pools. This page does not regenerate the figure.
