"""
Voxel texture and GPU
=====================

Texture maps are **inputs to clustering**, not post-label tables.
GPU is a faster implementation of the same IBSI / PyRadiomics definition —
the numbers do not change because of GPU.

Pass :class:`~habit.api.image.ImageVolume` /
:class:`~habit.api.image.MaskVolume` to the plotter (not ``.data``).
"""

# %%
# Local entropy on one demo subject. :func:`~habit.kernels.local_entropy_map`
# returns a volume; :func:`~habit.viz.plot_voxel_texture_slice` overlays it.
from pathlib import Path
import time
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import habit.voxel_features  # registers built-in voxel extractors
from habit.contracts import Subject, VoxelFeatureField, cohort_from_directory
from habit.datasets import fetch_demo
from habit.kernels import local_entropy_map
from habit.voxel_features import VoxelFeatureExtractorRegistry
from habit.viz import plot_voxel_texture_slice

DATA = fetch_demo()
MODALITY = "LAP"
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=(MODALITY,), roi=ROI)[:1]
subject = cohort[0]
image_vol = subject.image(MODALITY)
mask_vol = subject.mask(ROI)

entropy = local_entropy_map(image_vol.data, kernel_size=5, bins=32)
fig = plot_voxel_texture_slice(
    entropy, anatomy=image_vol, roi_mask=mask_vol
)
Path("out").mkdir(exist_ok=True)
fig.savefig("out/voxel_texture_overlay.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Three acceleration runtimes for voxel radiomics
# -----------------------------------------------
# When computing voxel-level texture on clinical lesions, three distinct runtime
# architectures can be used:
#
# 1. **Pure PyRadiomics (CPU)**: Single-threaded C-extension (``cMatrices``) for
#    texture matrices + NumPy for feature formulas.
# 2. **C matrices + TorchRadiomics (GPU)**: PyRadiomics C-extension on CPU for
#    matrix building + PyTorch on GPU for feature formulas (upstream pytorchradiomics).
# 3. **HABIT built-in GPU**: GPU-native matrix building (``gpumatrices``) +
#    GPU feature formulas (zero CPU-GPU transfer of intermediate matrices).
#
# Below, we define a unified extraction helper that selects among these runtimes.
RADIOMICS_PARAMS: Dict[str, Any] = {
    "imageType": {"Original": {}},
    "featureClass": {"glcm": ["Contrast"]},
    "setting": {"binWidth": 25},
}


def extract_voxel_radiomics(
    item: Subject,
    *,
    torch_device: str = "cpu",
    use_torch_radiomics: bool = False,
    use_gpu_matrices: bool = False,
) -> VoxelFeatureField:
    """Run voxel-wise GLCM Contrast with an explicit runtime configuration.

    Args:
        item: Subject whose ``MODALITY`` image and ``ROI`` mask are used.
        torch_device: ``"cpu"`` or ``"cuda"`` (and optional index).
        use_torch_radiomics: Whether TorchRadiomics is enabled.
        use_gpu_matrices: Whether TorchRadiomics texture matrices run on GPU.

    Returns:
        Voxel-by-feature field (one row per ROI voxel).
    """
    return VoxelFeatureExtractorRegistry.create(
        "voxel_radiomics",
        modality=MODALITY,
        kernel_radius=1,
        torch_device=torch_device,
        use_torch_radiomics=use_torch_radiomics,
        use_gpu_matrices=use_gpu_matrices,
        params=RADIOMICS_PARAMS,
    )(item)


# Run baseline CPU extraction on demo subject (34,694 voxels)
cpu_t0 = time.perf_counter()
cpu_field = extract_voxel_radiomics(
    subject, torch_device="cpu", use_torch_radiomics=False, use_gpu_matrices=False
)
cpu_seconds = time.perf_counter() - cpu_t0
print("CPU voxel_radiomics feature table:")
print(cpu_field.feature_frame().head())
cpu_field.feature_frame().head()

fig_glcm = plot_voxel_texture_slice(
    cpu_field, feature=0, anatomy=image_vol, roi_mask=mask_vol
)
fig_glcm.savefig("out/voxel_texture_glcm_contrast.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# GPU radiomics and the three-way runtime comparison
# --------------------------------------------------
# Install CUDA torch, then the optional extra::
#
#    pip install torch --index-url https://download.pytorch.org/whl/cu124
#    pip install "habitat-analysis[torch]"
#
# We compare the two GPU-accelerated runtimes against baseline CPU:
#
# * **Runtime 2**: C matrices + TorchRadiomics (``use_gpu_matrices=False``)
# * **Runtime 3**: HABIT built-in GPU matrices + TorchRadiomics (``use_gpu_matrices=True``)
#
# On clinical lesions, HABIT GPU eliminates both the CPU single-threaded C loop
# and the host-to-device intermediate array transfers.
try:
    import torch
except ImportError:
    torch = None

gpu_field: Optional[VoxelFeatureField] = None
gpu_seconds: Optional[float] = None
c_torch_seconds: Optional[float] = None
gpu_device = "unavailable"

if torch is None:
    print("torch not installed; skip GPU device probe (CPU numbers above still apply)")
elif not torch.cuda.is_available():
    print("CUDA not available; skip GPU path (CPU numbers still run)")
else:
    gpu_device = "cuda"

    # Runtime 2: C matrices + TorchRadiomics GPU
    c_torch_t0 = time.perf_counter()
    c_torch_field = extract_voxel_radiomics(
        subject,
        torch_device="cuda",
        use_torch_radiomics=True,
        use_gpu_matrices=False,
    )
    c_torch_seconds = time.perf_counter() - c_torch_t0

    # Runtime 3: HABIT built-in GPU matrices + TorchRadiomics GPU
    gpu_t0 = time.perf_counter()
    gpu_field = extract_voxel_radiomics(
        subject,
        torch_device="cuda",
        use_torch_radiomics=True,
        use_gpu_matrices=True,
    )
    gpu_seconds = time.perf_counter() - gpu_t0

    print("GPU voxel_radiomics feature table (HABIT built-in GPU):")
    print(gpu_field.feature_frame().head())
    gpu_field.feature_frame().head()

n_voxels = int(cpu_field.values.shape[0])
n_features = int(cpu_field.values.shape[1])
timing_rows = [
    {
        "runtime": "Pure PyRadiomics (CPU)",
        "seconds": round(cpu_seconds, 3),
        "speedup_vs_cpu": "1.0x",
        "speedup_vs_c_torch": "-",
        "n_voxels": n_voxels,
        "n_features": n_features,
    }
]
if c_torch_seconds is not None:
    timing_rows.append(
        {
            "runtime": "C matrices + TorchRadiomics (GPU)",
            "seconds": round(c_torch_seconds, 3),
            "speedup_vs_cpu": f"{cpu_seconds / c_torch_seconds:.1f}x",
            "speedup_vs_c_torch": "1.0x",
            "n_voxels": n_voxels,
            "n_features": n_features,
        }
    )
if gpu_field is not None and gpu_seconds is not None:
    timing_rows.append(
        {
            "runtime": "HABIT Built-in GPU",
            "seconds": round(gpu_seconds, 3),
            "speedup_vs_cpu": f"{cpu_seconds / gpu_seconds:.1f}x",
            "speedup_vs_c_torch": (
                f"{c_torch_seconds / gpu_seconds:.1f}x"
                if c_torch_seconds is not None
                else "-"
            ),
            "n_voxels": int(gpu_field.values.shape[0]),
            "n_features": int(gpu_field.values.shape[1]),
        }
    )
timing = pd.DataFrame(timing_rows)
print("\n--- Live Runtime Comparison on subj001 (34,694 voxels) ---")
print(timing.to_string(index=False))
timing

# %%
# Numerical parity: GPU must match CPU up to floating-point noise. Report
# max absolute difference and ``numpy.allclose``. A speedup does not
# change the IBSI / PyRadiomics definition.
if gpu_field is None:
    print("GPU skipped; no CPU vs GPU numerical table")
else:
    cpu_values = cpu_field.feature_frame().to_numpy(dtype=np.float64)
    gpu_values = gpu_field.feature_frame().to_numpy(dtype=np.float64)
    abs_diff = np.abs(cpu_values - gpu_values)
    max_abs = float(np.nanmax(abs_diff))
    rtol = 1e-5
    atol = 1e-8
    close = bool(np.allclose(cpu_values, gpu_values, rtol=rtol, atol=atol, equal_nan=True))
    parity = pd.DataFrame(
        [
            {
                "max_abs_diff": max_abs,
                "rtol": rtol,
                "atol": atol,
                "allclose": close,
            }
        ]
    )
    print(parity.to_string(index=False))
    parity

# %%
# Three tiers of voxel texture acceleration and scientific parity
# ---------------------------------------------------------------
# In real-world clinical imaging, tumor volumes can exceed tens of thousands
# of voxels. As the ROI expands, the single-threaded C loop in PyRadiomics
# becomes a critical bottleneck.
#
# Hardware (AutoDL cloud benchmark): NVIDIA GeForce RTX 4080 SUPER (32 GiB),
# Intel Xeon Platinum 8352V CPU.
# Workload: :class:`~habit.voxel_features.voxel_radiomics.VoxelRadiomicsFeatures`
# on a large lesion (54,913 ROI tumor voxels, 3D volume shape 80×80×48),
# extracting 90 three-dimensional radiomic texture features (FirstOrder, GLCM,
# GLRLM, GLSZM, GLDM, NGTDM).
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
# Scientific parity (54,913 voxels × 90 features, ~4.94M values):
#
# * **Route B vs Route C**: **Bit-identical** (``max_abs_diff = 0.0``, ``max_rel_diff = 0.0``).
#   HABIT built-in ``gpumatrices`` matches PyRadiomics C matrix construction exactly.
# * **Route A vs Route C**: ``max_abs_diff = 0.5`` on Energy / TotalEnergy
#   (~2.45e6 -> relative ~2e-7). For ``|value| >= 1e-3``, worst relative difference
#   is ~1.26e-2 on Skewness near zero. Mean absolute error over all 4.94M values is
#   ~0.00137. ``np.allclose(rtol=1e-4, atol=1e-4)`` holds across all features.
#   This is purely float32 summation rounding, not a mathematical definition change.
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
fig.savefig("out/voxel_texture_acceleration_speedup.png", dpi=150, bbox_inches="tight")
plt.show()
three_route_bench
