#!/usr/bin/env python
"""
Compare batched GPU supervoxel radiomics vs PyRadiomics on real or synthetic data.

Usage (Git Bash / py310):
    # Demo subject with ALL feature classes (default paths under demo_data/)
    E:/conda/mconda/envs/py310/python.exe test_compare_supervoxel_gpu_vs_pyradiomics.py \\
        --reference pyradiomics --all-features

    # Custom paths
    E:/conda/mconda/envs/py310/python.exe test_compare_supervoxel_gpu_vs_pyradiomics.py \\
        --image path/to/image.nii.gz \\
        --supervoxel-map path/to/subj001_supervoxel.nrrd \\
        --params-file config/radiomics/params_supervoxel_radiomics.yaml \\
        --reference pyradiomics --all-features
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch

from habit.core.habitat_analysis.clustering_features.batched_supervoxel_radiomics import (
    DEFAULT_FEATURES_BY_CLASS,
    _resolve_enabled_features,
    extract_batched_supervoxel_features,
)
from habit.utils.radiomics_params_utils import (
    VOXEL_SAFE_GLCM_FEATURES,
    create_radiomics_feature_extractor,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE = (
    PROJECT_ROOT
    / "demo_data/preprocessed/resample_02/images/subj001/delay3/delay3.nii.gz"
)
DEFAULT_SUPERVOXEL_MAP = (
    PROJECT_ROOT / "demo_data/results/habitat_two_step/subj001_supervoxel.nrrd"
)
DEFAULT_PARAMS = PROJECT_ROOT / "config/radiomics/params_supervoxel_radiomics.yaml"


def _align_image_to_reference(image: sitk.Image, reference: sitk.Image) -> sitk.Image:
    """
    Resample ``image`` onto the grid of ``reference`` when sizes/spacing differ.

    Args:
        image: Source intensity image.
        reference: Target geometry (typically the supervoxel map).

    Returns:
        sitk.Image: Resampled image aligned to ``reference``.
    """
    same_size = image.GetSize() == reference.GetSize()
    same_spacing = image.GetSpacing() == reference.GetSpacing()
    same_origin = image.GetOrigin() == reference.GetOrigin()
    if same_size and same_spacing and same_origin:
        return image

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    aligned = resampler.Execute(image)
    print(
        "Resampled image",
        image.GetSize(),
        "->",
        aligned.GetSize(),
        "to match supervoxel map geometry.",
    )
    return aligned


def _load_image(path: Path) -> sitk.Image:
    """Load a NIfTI/NRRD image via SimpleITK."""
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    return sitk.ReadImage(str(path))


def _load_labels(
    sv_map: sitk.Image,
    *,
    max_labels: Optional[int],
) -> np.ndarray:
    """
    Return sorted positive supervoxel label ids.

    Args:
        sv_map: Supervoxel label map.
        max_labels: Optional cap for faster smoke tests.

    Returns:
        np.ndarray: 1D label ids.
    """
    sv_array = sitk.GetArrayFromImage(sv_map)
    labels = np.unique(sv_array)
    labels = labels[labels > 0]
    if max_labels is not None and max_labels > 0:
        labels = labels[:max_labels]
    return labels.astype(np.int64, copy=False)


def _make_synthetic_case(seed: int = 42) -> Tuple[sitk.Image, sitk.Image, np.ndarray]:
    """Build a small synthetic image with several supervoxels."""
    rng = np.random.default_rng(seed)
    image_array = rng.integers(10, 200, size=(24, 24, 24), dtype=np.int16).astype(np.float32)
    sv_array = np.zeros((24, 24, 24), dtype=np.uint8)
    sv_array[2:10, 2:10, 2:10] = 1
    sv_array[12:20, 2:10, 2:10] = 2
    sv_array[2:10, 12:20, 12:20] = 3
    sv_array[12:18, 12:18, 12:18] = 4
    sv_array[18:22, 18:22, 18:22] = 5

    image = sitk.GetImageFromArray(image_array)
    image.SetSpacing((1.0, 1.0, 1.0))
    sv_map = sitk.GetImageFromArray(sv_array)
    sv_map.CopyInformation(image)
    labels = np.unique(sv_array)
    labels = labels[labels > 0]
    return image, sv_map, labels


def _enabled_features_from_args(
    *,
    params_file: Optional[Path],
    all_features: bool,
    feature_classes: Optional[List[str]],
) -> Tuple[Dict[str, List[str]], Dict[str, object]]:
    """
    Resolve enabled feature classes and PyRadiomics settings.

    Returns:
        Tuple[enabled_features, settings]
    """
    if all_features:
        enabled = {
            feature_class: list(DEFAULT_FEATURES_BY_CLASS[feature_class])
            for feature_class in DEFAULT_FEATURES_BY_CLASS
        }
        # MCC / Imc1 / Imc2 can crash MKL or CUDA eigvals on real supervoxel ROIs.
        enabled["glcm"] = list(VOXEL_SAFE_GLCM_FEATURES)
        settings: Dict[str, object] = {}
        if params_file is not None and params_file.is_file():
            extractor = create_radiomics_feature_extractor(str(params_file))
            settings = dict(extractor.settings)
        else:
            settings = {
                "binWidth": 25,
                "voxelArrayShift": 300,
                "force2D": False,
                "geometryTolerance": 1e-3,
                "normalize": False,
            }
        return enabled, settings

    if params_file is None or not params_file.is_file():
        raise FileNotFoundError(
            "Provide --params-file or use --all-features for a full feature comparison."
        )

    extractor = create_radiomics_feature_extractor(str(params_file))
    enabled = _resolve_enabled_features(extractor.enabledFeatures)

    if feature_classes:
        enabled = {
            feature_class: enabled[feature_class]
            for feature_class in feature_classes
            if feature_class in enabled
        }

    return enabled, dict(extractor.settings)


def _run_pyradiomics_reference(
    image: sitk.Image,
    sv_map: sitk.Image,
    labels: np.ndarray,
    enabled_features: Mapping[str, List[str]],
    settings: Dict[str, object],
) -> pd.DataFrame:
    """CPU PyRadiomics reference with union-mask binning (same semantics as batched path)."""
    from habit.core.habitat_analysis.clustering_features.batched_supervoxel_radiomics import (
        extract_supervoxel_features_pyradiomics,
    )

    enabled_map = {feature_class: list(names) for feature_class, names in enabled_features.items()}
    return extract_supervoxel_features_pyradiomics(
        image,
        sv_map,
        labels,
        enabled_features=enabled_map,
        settings=settings,
        batch_size=1,
    )


def _run_sequential_torch_reference(
    image: sitk.Image,
    sv_map: sitk.Image,
    labels: np.ndarray,
    enabled_features: Mapping[str, List[str]],
    settings: Dict[str, object],
    *,
    device: str,
    dtype_name: str,
) -> pd.DataFrame:
    """Sequential TorchRadiomics with union-mask binning (batch_size=1)."""
    enabled_map = {feature_class: list(names) for feature_class, names in enabled_features.items()}
    return extract_batched_supervoxel_features(
        image,
        sv_map,
        labels,
        enabled_features=enabled_map,
        settings=settings,
        device=device,
        dtype_name=dtype_name,
        batch_size=1,
    )


def _compare_frames(
    df_gpu: pd.DataFrame,
    df_ref: pd.DataFrame,
    *,
    rtol: float,
    atol: float,
) -> Tuple[int, int, List[str], pd.DataFrame]:
    """
    Compare GPU and reference DataFrames.

    Returns:
        matched, mismatched, mismatch lines, per-feature diff table
    """
    common_cols = sorted(
        set(df_gpu.columns) & set(df_ref.columns) - {"SupervoxelID"}
    )
    merged = df_gpu.merge(df_ref, on="SupervoxelID", suffixes=("_gpu", "_ref"))
    mismatches: List[str] = []
    diff_rows: List[Dict[str, object]] = []

    for col in common_cols:
        gpu_vals = merged[f"{col}_gpu"].to_numpy(dtype=np.float64)
        ref_vals = merged[f"{col}_ref"].to_numpy(dtype=np.float64)
        abs_diff = np.abs(gpu_vals - ref_vals)
        max_diff = float(np.max(abs_diff))
        ok = bool(np.allclose(gpu_vals, ref_vals, rtol=rtol, atol=atol, equal_nan=True))
        diff_rows.append({
            "feature": col,
            "max_abs_diff": max_diff,
            "mean_abs_diff": float(np.mean(abs_diff)),
            "matched": ok,
        })
        if not ok:
            mismatches.append(f"{col}: max_abs_diff={max_diff:.6g}")

    diff_df = pd.DataFrame(diff_rows).sort_values("max_abs_diff", ascending=False)
    matched = int(diff_df["matched"].sum())
    mismatched = int((~diff_df["matched"]).sum())
    return matched, mismatched, mismatches, diff_df


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--supervoxel-map", type=Path, default=DEFAULT_SUPERVOXEL_MAP)
    parser.add_argument("--params-file", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument(
        "--feature-class",
        action="append",
        dest="feature_classes",
        default=None,
        help="Limit comparison to feature class(es), e.g. glcm, glrlm.",
    )
    parser.add_argument(
        "--all-features",
        action="store_true",
        help="Enable all six PyRadiomics feature classes (firstorder + textures).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use built-in synthetic data instead of --image/--supervoxel-map.",
    )
    parser.add_argument(
        "--reference",
        choices=("torch-sequential", "pyradiomics"),
        default="pyradiomics",
        help="Reference backend for numerical comparison.",
    )
    parser.add_argument(
        "--max-labels",
        type=int,
        default=None,
        help="Compare only the first N supervoxel labels (for quick tests).",
    )
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--report-csv",
        type=Path,
        default=None,
        help="Optional path to save per-feature diff summary CSV.",
    )
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("CUDA is not available. This script validates GPU batched extraction.")
        return 1

    if args.synthetic:
        image, sv_map, labels = _make_synthetic_case()
        print("Data source: synthetic")
    else:
        image = _load_image(args.image)
        sv_map = _load_image(args.supervoxel_map)
        image = _align_image_to_reference(image, sv_map)
        sv_map.CopyInformation(image)
        labels = _load_labels(sv_map, max_labels=args.max_labels)
        print("Data source: real")
        print("Image:", args.image)
        print("Supervoxel map:", args.supervoxel_map)

    enabled, settings = _enabled_features_from_args(
        params_file=args.params_file,
        all_features=args.all_features,
        feature_classes=args.feature_classes,
    )
    settings.setdefault("geometryTolerance", 1e-3)

    n_features = sum(len(names) for names in enabled.values())
    print("Enabled feature classes:", sorted(enabled.keys()))
    print("Total features:", n_features)
    print("Reference mode:", args.reference)
    print("Supervoxels compared:", len(labels), labels.tolist()[:10], "...")

    t0 = time.perf_counter()
    df_gpu = extract_batched_supervoxel_features(
        image,
        sv_map,
        labels,
        enabled_features=enabled,
        settings=settings,
        device="cuda:0",
        dtype_name="float64",
        batch_size=args.batch_size,
    )
    gpu_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    if args.reference == "torch-sequential":
        df_ref = _run_sequential_torch_reference(
            image,
            sv_map,
            labels,
            enabled,
            settings,
            device="cuda:0",
            dtype_name="float64",
        )
    else:
        df_ref = _run_pyradiomics_reference(
            image, sv_map, labels, enabled, settings,
        )
    ref_seconds = time.perf_counter() - t1

    matched, mismatched, details, diff_df = _compare_frames(
        df_gpu, df_ref, rtol=args.rtol, atol=args.atol,
    )
    total_cols = matched + mismatched

    print(f"\nGPU batched time: {gpu_seconds:.3f}s")
    print(f"Reference time: {ref_seconds:.3f}s")
    print(f"Compared columns: {total_cols}")
    print(f"Matched: {matched}, Mismatched: {mismatched}")

    if args.report_csv is not None:
        diff_df.to_csv(args.report_csv, index=False)
        print(f"Diff report saved: {args.report_csv}")

    if mismatched:
        print("\nTop mismatches:")
        for line in details[:30]:
            print(" ", line)
        if len(details) > 30:
            print(f"  ... and {len(details) - 30} more")
        if args.reference == "pyradiomics":
            print(
                "\nNote: GPU batched path uses global binning across all supervoxels, "
                "while PyRadiomics bins each ROI independently. "
                "Use --reference torch-sequential to validate GPU batch math only."
            )
        return 2

    print("\nAll compared features match within tolerance.")
    top = diff_df.head(10)
    if not top.empty:
        print("\nLargest abs diffs (still within tolerance):")
        for _, row in top.iterrows():
            print(
                f"  {row['feature']}: max={row['max_abs_diff']:.6g}, "
                f"mean={row['mean_abs_diff']:.6g}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
