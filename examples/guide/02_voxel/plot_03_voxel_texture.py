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
# Same ``voxel_radiomics`` Spec on CPU and (when CUDA is present) GPU.
# ``binWidth`` and the GLCM name are identical; GPU is not a different
# IBSI feature. Time each path and compare the feature tables numerically.
RADIOMICS_PARAMS: Dict[str, Any] = {
    "imageType": {"Original": {}},
    "featureClass": {"glcm": ["Contrast"]},
    "setting": {"binWidth": 25},
}


def extract_voxel_radiomics(
    item: Subject,
    *,
    torch_device: str,
    use_gpu_matrices: bool,
) -> VoxelFeatureField:
    """Run voxel-wise GLCM Contrast with an explicit device.

    Args:
        item: Subject whose ``MODALITY`` image and ``ROI`` mask are used.
        torch_device: ``"cpu"`` or ``"cuda"`` (and optional index).
        use_gpu_matrices: Whether TorchRadiomics texture matrices run on GPU.

    Returns:
        Voxel-by-feature field (one row per ROI voxel).
    """
    return VoxelFeatureExtractorRegistry.create(
        "voxel_radiomics",
        modality=MODALITY,
        kernel_radius=1,
        torch_device=torch_device,
        use_gpu_matrices=use_gpu_matrices,
        params=RADIOMICS_PARAMS,
    )(item)


cpu_t0 = time.perf_counter()
cpu_field = extract_voxel_radiomics(subject, torch_device="cpu", use_gpu_matrices=False)
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
# GPU radiomics
# -------------
# Install CUDA torch, then the optional extra::
#
#    pip install torch --index-url https://download.pytorch.org/whl/cu124
#    pip install "habitat-analysis[torch]"
#
# Same extractor kwargs as the CPU cell. Use CUDA when available; otherwise
# skip GPU with a clear print so this page still runs.
#
# Speedup (RTX 3070 Laptop, 4096-voxel ROI), end-to-end vs PyRadiomics CPU:
#
# * First-order: 0.760 s → 0.037 s (~21×)
# * GLCM: 2.556 s → 0.197 s (~13×)
# * GLRLM: 1.567 s → 0.530 s (~3×)
# * Integer count matrices are bit-identical; see
#   :doc:`/reference/features/traditional` for IBSI Phase 1.
try:
    import torch
except ImportError:
    torch = None

gpu_field: Optional[VoxelFeatureField] = None
gpu_seconds: Optional[float] = None
gpu_device = "unavailable"
if torch is None:
    print("torch not installed; skip GPU device probe (CPU numbers above still apply)")
elif not torch.cuda.is_available():
    print("CUDA not available; skip GPU path (CPU numbers still run)")
else:
    gpu_device = "cuda"
    gpu_t0 = time.perf_counter()
    gpu_field = extract_voxel_radiomics(
        subject, torch_device="cuda", use_gpu_matrices=True
    )
    gpu_seconds = time.perf_counter() - gpu_t0
    print("GPU voxel_radiomics feature table:")
    print(gpu_field.feature_frame().head())
    gpu_field.feature_frame().head()

n_voxels = int(cpu_field.values.shape[0])
n_features = int(cpu_field.values.shape[1])
timing_rows = [
    {
        "device": "cpu",
        "seconds": cpu_seconds,
        "n_voxels": n_voxels,
        "n_features": n_features,
    }
]
if gpu_field is not None and gpu_seconds is not None:
    timing_rows.append(
        {
            "device": gpu_device,
            "seconds": gpu_seconds,
            "n_voxels": int(gpu_field.values.shape[0]),
            "n_features": int(gpu_field.values.shape[1]),
        }
    )
timing = pd.DataFrame(timing_rows)
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
