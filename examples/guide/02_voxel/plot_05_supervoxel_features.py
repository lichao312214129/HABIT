"""
Supervoxel feature extraction and acceleration
==============================================

In two-step habitat analysis, the tumor ROI is first partitioned into supervoxels.
Each supervoxel must then be summarized into a quantitative feature vector
before cohort pooling and clustering.

HABIT supports two primary approaches for supervoxel feature extraction:

1. **Statistical aggregations**: reducing the per-voxel signals inside each
   supervoxel (e.g. :class:`~habit.supervoxel.MeanSupervoxelFeatures`,
   median/percentiles via :class:`~habit.supervoxel.PercentileSupervoxelFeatures`,
   and standard deviation via :class:`~habit.supervoxel.StdSupervoxelFeatures`).
2. **Supervoxel radiomics texture**: extracting full IBSI-compliant radiomics
   features for every individual supervoxel (:class:`~habit.supervoxel.SupervoxelRadiomicsFeatures`).

High-throughput acceleration vs PyRadiomics parity
--------------------------------------------------
Extracting texture features for dozens of supervoxels sequentially using standard
PyRadiomics ``execute()`` incurs high overhead (repeated mask hashing, bounding-box
cropping, and Python loop costs).
HABIT enables an accelerated native C-extension engine by default
(``use_supervoxel_cext=True``), running directly on CPU without GPU overhead.
Furthermore, HABIT sets GPU and backend calculation precision to double precision
``float64`` by default, guaranteeing exact numerical parity within machine epsilon
against official PyRadiomics without single-precision (``float32``) quantization error.
This page benchmarks extraction time, confirms multi-fold speedup, and proves exact
numerical parity against PyRadiomics.
"""

# sphinx_gallery_thumbnail_number = 2

# %%
# Load one demo subject and generate SLIC supervoxels.
from pathlib import Path
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.supervoxel import (
    MeanSupervoxelFeatures,
    PercentileSupervoxelFeatures,
    SlicSupervoxelizer,
    StdSupervoxelFeatures,
    SupervoxelRadiomicsFeatures,
)
from habit.viz import plot_habitat_overlay, use_style
from habit.viz.labels import sanitize_label
from habit.voxel_features import RawVoxelFeatures

DATA = fetch_demo()
MODALITIES = ("LAP",)
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:1]
subject = cohort[0]
image = subject.image(MODALITIES[0])
Path("out").mkdir(exist_ok=True)
print(f"Subject: {subject.subject_id}")

# Extract raw voxel features and partition ROI into 24 SLIC supervoxels
voxel = RawVoxelFeatures(modalities=list(MODALITIES))
field = voxel(subject)
svx = SlicSupervoxelizer(n_supervoxels=24, compactness=10.0)
units = svx(field)
print(f"Generated {len(units.features)} SLIC supervoxels.")

# Visualize the SLIC supervoxels on the anatomical image
fig_svx = plot_habitat_overlay(
    image,
    units,
    title="SLIC supervoxels (n=24)",
)
fig_svx.savefig("out/supervoxel_features_slic_overlay.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Approach 1: Statistical feature extraction (Mean, Median, Std).
# Each extractor binds the voxel field and aggregates voxel intensities within each parcel.
mean_ext = MeanSupervoxelFeatures(modality=MODALITIES[0])
mean_ext.bind_fields(working=field)
mean_units = mean_ext(subject, units)

med_ext = PercentileSupervoxelFeatures(modality=MODALITIES[0], q=50.0)
med_ext.bind_fields(working=field)
med_units = med_ext(subject, units)

std_ext = StdSupervoxelFeatures(modality=MODALITIES[0])
std_ext.bind_fields(working=field)
std_units = std_ext(subject, units)

stats_table = pd.DataFrame(
    {
        "supervoxel": mean_units.features.index,
        "mean_intensity": mean_units.features[MODALITIES[0]].values,
        "median_intensity": med_units.features[f"p50-{MODALITIES[0]}"].values,
        "std_intensity": std_units.features[f"std-{MODALITIES[0]}"].values,
    }
)
print("Supervoxel statistical features (first 5 parcels):")
print(stats_table.head(5).round(2).to_string(index=False))
stats_table.head(5)

# %%
# Approach 2: Supervoxel Radiomics texture and high-throughput acceleration.
# Compare HABIT native C-extension against PyRadiomics for time and numerical parity.
texture_params = {
    "imageType": {"Original": {}},
    "featureClass": {
        "firstorder": ["Mean", "Variance", "Skewness"],
        "glcm": ["Contrast", "Correlation", "JointEntropy"],
    },
    "setting": {"binWidth": 25.0, "normalize": False},
}

# Warm up extractors to isolate steady-state execution time from module import latency.
# output_float32=False keeps full float64 double precision to inspect numerical parity.
rad_native = SupervoxelRadiomicsFeatures(
    modality=MODALITIES[0],
    params=texture_params,
    use_supervoxel_cext=True,
    output_float32=False,
)
rad_pyrad = SupervoxelRadiomicsFeatures(
    modality=MODALITIES[0],
    params=texture_params,
    use_supervoxel_cext=False,
    use_torch_radiomics=False,
    output_float32=False,
)
_ = rad_native(subject, units)
_ = rad_pyrad(subject, units)

# Benchmark Native C-extension
t0 = time.perf_counter()
res_native = rad_native(subject, units)
t_native = time.perf_counter() - t0

# Benchmark standard PyRadiomics
t0 = time.perf_counter()
res_pyrad = rad_pyrad(subject, units)
t_pyrad = time.perf_counter() - t0

speedup = t_pyrad / max(t_native, 1e-6)
print(f"\nExtraction time for {len(units.features)} supervoxels:")
print(f"  HABIT Native C-extension : {t_native:.3f} s")
print(f"  Reference PyRadiomics    : {t_pyrad:.3f} s")
print(f"  Speedup factor           : {speedup:.2f}x faster")

# Numerical parity check across all matching texture columns
common_cols = [c for c in res_native.features.columns if c in res_pyrad.features.columns]
parity_records: List[Dict[str, float]] = []
for col in common_cols:
    val_nat = res_native.features[col].to_numpy(dtype=float)
    val_pyr = res_pyrad.features[col].to_numpy(dtype=float)
    abs_diff = np.abs(val_nat - val_pyr)
    parity_records.append(
        {
            "feature": col,
            "max_abs_diff": float(np.nanmax(abs_diff)),
            "mean_abs_diff": float(np.nanmean(abs_diff)),
        }
    )

parity_df = pd.DataFrame(parity_records)
max_overall_diff = parity_df["max_abs_diff"].max()
print(f"\nMaximum absolute difference across all features: {max_overall_diff:.2e}")
print("\nFeature-level parity table:")
print(parity_df.to_string(index=False))

# %%
# Visual comparison: execution time and numerical parity correlation.
with use_style("radiology"):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), constrained_layout=True)

    # Panel 1: Execution time benchmark bar chart
    methods = ["HABIT Native (C-ext)", "PyRadiomics"]
    times = [t_native, t_pyrad]
    colors = ["#0072B2", "#E69F00"]
    bars = axes[0].bar(methods, times, color=colors, width=0.45)
    axes[0].set_ylabel("Execution time (seconds)")
    axes[0].set_title(sanitize_label(f"Supervoxel texture benchmark ({speedup:.1f}x speedup)"))
    axes[0].set_ylim(0, max(times) * 1.25)
    for bar in bars:
        height = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(times) * 0.03,
            f"{height:.2f} s",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Panel 2: Numerical parity scatter plot for GLCM JointEntropy
    sample_col = [c for c in common_cols if "JointEntropy" in c][0]
    nat_vals = res_native.features[sample_col].to_numpy(dtype=float)
    pyr_vals = res_pyrad.features[sample_col].to_numpy(dtype=float)
    axes[1].scatter(pyr_vals, nat_vals, color="#009E73", s=45, alpha=0.85, label="Supervoxel values")
    lims = [min(pyr_vals.min(), nat_vals.min()), max(pyr_vals.max(), nat_vals.max())]
    axes[1].plot(lims, lims, color="#D55E00", linestyle="--", linewidth=1.5, label="Identity (y = x)")
    axes[1].set_xlabel("PyRadiomics value")
    axes[1].set_ylabel("HABIT Native C-ext value")
    axes[1].set_title(sanitize_label(f"Numerical parity: {sample_col.split('_')[-1]}"))
    axes[1].legend(loc="lower right", frameon=True)

fig.savefig("out/supervoxel_features_benchmark_parity.png", dpi=150, bbox_inches="tight")
plt.show()
