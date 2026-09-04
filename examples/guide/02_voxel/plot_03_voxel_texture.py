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
# Large clinical tumor benchmark summary (up to 80,000+ voxels)
# ------------------------------------------------------------
# In real-world clinical imaging, tumor volumes can exceed tens of thousands
# of voxels. As the ROI expands, the single-threaded C loop in PyRadiomics
# becomes a critical bottleneck.
#
# Below is the benchmark comparing all three runtimes on clinical liver lesions
# (NVIDIA GeForce RTX 3070 Laptop GPU, ``binWidth=25``, ``kernelRadius=1``):
#
# * **subj001 (34,694 voxels)**:
#   - GLCM Contrast: CPU 68.71 s -> C+Torch 7.54 s -> HABIT GPU 2.68 s (25.6x vs CPU, 2.8x vs C+Torch)
#   - GLCM 4 features: CPU 85.33 s -> C+Torch 7.04 s -> HABIT GPU 3.02 s (28.3x vs CPU, 2.3x vs C+Torch)
#   - GLRLM (2 features): CPU 13.25 s -> C+Torch 8.02 s -> HABIT GPU 2.61 s (5.1x vs CPU, 3.1x vs C+Torch)
# * **subj005 (80,084 voxels, massive tumor benchmark)**:
#   - Pure PyRadiomics CPU: 414.58 s (~7 minutes)
#   - C matrices + TorchRadiomics GPU: 39.62 s
#   - HABIT Built-in GPU: 7.59 s (**54.6x vs CPU, 5.2x vs C+Torch**)
#
# On an 80k-voxel volume, HABIT collapses a 7-minute CPU bottleneck down to 7.6 seconds!
benchmark_summary = pd.DataFrame(
    [
        {
            "Case / Volume": "subj001 (34,694 voxels)",
            "Task": "GLCM Contrast",
            "Pure PyRadiomics (CPU)": "68.71 s",
            "C + TorchRadiomics (GPU)": "7.54 s",
            "HABIT Built-in GPU": "2.68 s",
            "Speedup vs CPU": "25.6x",
            "Speedup vs C+Torch": "2.8x",
        },
        {
            "Case / Volume": "subj001 (34,694 voxels)",
            "Task": "GLCM 4 features",
            "Pure PyRadiomics (CPU)": "85.33 s",
            "C + TorchRadiomics (GPU)": "7.04 s",
            "HABIT Built-in GPU": "3.02 s",
            "Speedup vs CPU": "28.3x",
            "Speedup vs C+Torch": "2.3x",
        },
        {
            "Case / Volume": "subj001 (34,694 voxels)",
            "Task": "GLRLM (2 features)",
            "Pure PyRadiomics (CPU)": "13.25 s",
            "C + TorchRadiomics (GPU)": "8.02 s",
            "HABIT Built-in GPU": "2.61 s",
            "Speedup vs CPU": "5.1x",
            "Speedup vs C+Torch": "3.1x",
        },
        {
            "Case / Volume": "subj005 (80,084 voxels)",
            "Task": "GLCM Contrast",
            "Pure PyRadiomics (CPU)": "414.58 s (~7 min)",
            "C + TorchRadiomics (GPU)": "39.62 s",
            "HABIT Built-in GPU": "7.59 s",
            "Speedup vs CPU": "54.6x",
            "Speedup vs C+Torch": "5.2x",
        },
    ]
)
print("\n--- Clinical Large-Tumor Comprehensive Benchmark ---")
print(benchmark_summary.to_string(index=False))
benchmark_summary
