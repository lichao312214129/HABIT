"""
Manual smoke test for vendored torchradiomics inside habit.

Uses the in-tree copy at
``habit/core/habitat_analysis/clustering_features/torchradiomics/``.

Compares voxel-based feature maps from conventional PyRadiomics against
TorchRadiomics (via ``inject_torch_radiomics``) for every supported feature
class.

Run from repo root::

    python tests/habitat/manual_torch_radiomics_voxel.py

Requires optional deps: torch, pyradiomics, SimpleITK, numpy<2 (for torch 2.4).
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import SimpleITK as sitk
import torch
from radiomics.featureextractor import RadiomicsFeatureExtractor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from habit.core.habitat_analysis.clustering_features.torchradiomics import (
    inject_torch_radiomics,
    restore_radiomics,
)
from habit.core.habitat_analysis.clustering_features.voxel_radiomics_extractor import (
    DEFAULT_VOXEL_BATCH,
)

IMAGE_PATH = PROJECT_ROOT / ".cursor/test/resample_02/images/sub164/T2/T2.nii.gz"
MASK_PATH = PROJECT_ROOT / ".cursor/test/resample_02/masks/sub164/T2/mask_T2.nii.gz"
PARAMS_FILE = PROJECT_ROOT / ".cursor/test/parameter.yaml"
# Voxel neighborhood radius; matches config_habitat_direct_pooling.yaml (not in parameter.yaml).
KERNEL_RADIUS = 3
# Feature classes that torchradiomics injection supports.
TORCH_SUPPORTED_FEATURE_CLASSES: List[str] = [
    "firstorder",
    "glcm",
    "gldm",
    "glrlm",
    "glszm",
    "ngtdm",
]


@dataclass
class FeatureDiffStats:
    """Per-feature comparison statistics within the ROI mask."""

    feature_name: str
    n_voxels: int
    max_abs_diff: float
    mean_abs_diff: float
    max_rel_diff: float
    allclose: bool


@dataclass
class TimingStats:
    """Wall-clock timing for one feature class extraction."""

    feature_class: str
    cpu_seconds: float
    gpu_seconds: float

    @property
    def speedup(self) -> float:
        """CPU time divided by GPU time; values > 1 mean GPU is faster."""
        if self.gpu_seconds <= 0.0:
            return float("inf")
        return self.cpu_seconds / self.gpu_seconds


def _check_numpy_version() -> None:
    major = int(np.__version__.split(".", maxsplit=1)[0])
    if major >= 2:
        print(
            f"WARNING: numpy {np.__version__} may break torch 2.4; "
            "use `pip install \"numpy<2\"` in this environment.",
            file=sys.stderr,
        )


def get_enabled_torch_feature_classes(params_file: Path) -> List[str]:
    """
    Read enabled feature classes from a PyRadiomics parameter YAML file.

    Only classes supported by torchradiomics injection are returned, in a
    stable sorted order.

    Args:
        params_file: Path to the PyRadiomics parameter YAML file.

    Returns:
        List[str]: Enabled and torch-supported feature class names.
    """
    extractor = RadiomicsFeatureExtractor(str(params_file))
    enabled = set(extractor.enabledFeatures.keys())
    return sorted(enabled & set(TORCH_SUPPORTED_FEATURE_CLASSES))


def apply_voxel_settings(
    extractor: RadiomicsFeatureExtractor,
    kernel: int,
    device: str | None = None,
) -> RadiomicsFeatureExtractor:
    """
    Apply voxel-based extraction settings on top of a params-file extractor.

    Args:
        extractor: Extractor loaded from ``parameter.yaml``.
        kernel: Neighborhood radius in voxels.
        device: Torch device string for injection mode; ``None`` keeps CPU
            PyRadiomics classes.

    Returns:
        RadiomicsFeatureExtractor: The same extractor instance, updated in place.
    """
    extractor.settings.update(
        {
            "voxelBased": True,
            "padDistance": kernel,
            "kernelRadius": kernel,
            "maskedKernel": False,
            "voxelBatch": DEFAULT_VOXEL_BATCH,
            "geometryTolerance": 1e-3,
        }
    )
    if device is not None:
        extractor.settings["dtype"] = torch.float64
        extractor.settings["device"] = device
    return extractor


def load_image_mask(
    image_path: Path,
    mask_path: Path,
) -> Tuple[sitk.Image, sitk.Image]:
    """
    Load image/mask and align mask geometry with the image.

    Args:
        image_path: Path to the input image.
        mask_path: Path to the ROI mask.

    Returns:
        Tuple[sitk.Image, sitk.Image]: (image, mask).
    """
    image = sitk.ReadImage(str(image_path))
    mask = sitk.ReadImage(str(mask_path))
    mask.CopyInformation(image)
    return image, mask


def get_comparison_mask(
    image: sitk.Image,
    mask: sitk.Image,
    params_file: Path,
    kernel: int,
) -> sitk.Image:
    """
    Build the preprocessed ROI mask used for feature-map comparison.

    Both CPU and Torch paths run ``RadiomicsFeatureExtractor.execute()``, which
    applies the same internal preprocessing; this helper exposes that mask for
    ROI statistics.

    Args:
        image: Raw input image.
        mask: Raw ROI mask.
        params_file: Path to the PyRadiomics parameter YAML file.
        kernel: kernelRadius / padDistance for voxel extraction.

    Returns:
        sitk.Image: Preprocessed mask aligned with feature maps.
    """
    rf_ext = RadiomicsFeatureExtractor(str(params_file))
    apply_voxel_settings(rf_ext, kernel)
    _, mask_norm = rf_ext.loadImage(image, mask, None, **rf_ext.settings)
    return mask_norm


def create_feature_extractor(
    params_file: Path,
    kernel: int,
    feature_class: str,
    device: str | None = None,
) -> RadiomicsFeatureExtractor:
    """
    Create a ``RadiomicsFeatureExtractor`` from ``parameter.yaml`` for one class.

    Args:
        params_file: Path to the PyRadiomics parameter YAML file.
        kernel: Neighborhood radius in voxels.
        feature_class: Feature class name, e.g. ``"glcm"``.
        device: Torch device string for injection mode; ``None`` keeps CPU
            PyRadiomics classes.

    Returns:
        RadiomicsFeatureExtractor: Configured extractor instance.
    """
    extractor = RadiomicsFeatureExtractor(str(params_file))
    apply_voxel_settings(extractor, kernel, device)
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName(feature_class)
    return extractor


def extract_feature_maps(
    image: sitk.Image,
    mask: sitk.Image,
    params_file: Path,
    kernel: int,
    feature_class: str,
    device: str | None = None,
) -> Tuple[Dict[str, sitk.Image], float]:
    """
    Extract voxel-based feature maps via ``RadiomicsFeatureExtractor``.

    When ``device`` is set, temporarily inject TorchRadiomics classes into
    PyRadiomics and restore them afterward.

    Args:
        image: Raw input image.
        mask: Raw ROI mask.
        params_file: Path to the PyRadiomics parameter YAML file.
        kernel: Neighborhood radius in voxels.
        feature_class: Feature class name, e.g. ``"glcm"``.
        device: Torch device string for injection mode; ``None`` uses CPU
            PyRadiomics.

    Returns:
        Tuple[Dict[str, sitk.Image], float]:
            Feature name to SimpleITK feature map, and ``execute()`` wall time
            in seconds (CUDA synchronized when applicable).
    """
    use_torch = device is not None
    if use_torch:
        inject_torch_radiomics()

    try:
        extractor = create_feature_extractor(params_file, kernel, feature_class, device)
        start = time.perf_counter()
        result = extractor.execute(image, mask, voxelBased=True)
        if use_torch and device is not None and device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_seconds = time.perf_counter() - start
        features = {
            name: feature_map
            for name, feature_map in result.items()
            if isinstance(feature_map, sitk.Image) and not name.startswith("diagnostics")
        }
        return features, elapsed_seconds
    finally:
        if use_torch:
            restore_radiomics()


def compare_feature_maps(
    pyrad_features: Dict[str, sitk.Image],
    torch_features: Dict[str, sitk.Image],
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> Tuple[List[FeatureDiffStats], Dict[str, Any]]:
    """
    Compare PyRadiomics and TorchRadiomics feature maps inside the ROI.

    Both sides use ``RadiomicsFeatureExtractor.execute()``, which may crop the
    output to the tumor bounding box. Comparison is therefore performed on the
    shared cropped grid using finite voxel values.

    Args:
        pyrad_features: Feature maps from conventional PyRadiomics.
        torch_features: Feature maps from injected TorchRadiomics.
        rtol: Relative tolerance for ``np.allclose``.
        atol: Absolute tolerance for ``np.allclose``.

    Returns:
        Tuple[List[FeatureDiffStats], Dict[str, Any]]:
            Per-feature stats and overall summary dictionary.
    """
    pyrad_keys = set(pyrad_features.keys())
    torch_keys = set(torch_features.keys())
    common_keys = sorted(pyrad_keys & torch_keys)
    only_pyrad = sorted(pyrad_keys - torch_keys)
    only_torch = sorted(torch_keys - pyrad_keys)

    per_feature_stats: List[FeatureDiffStats] = []
    max_abs_diff_overall = 0.0
    mean_abs_diff_overall = 0.0
    max_rel_diff_overall = 0.0
    all_close_overall = True

    for feature_name in common_keys:
        pyrad_array = sitk.GetArrayFromImage(pyrad_features[feature_name])
        torch_array = sitk.GetArrayFromImage(torch_features[feature_name])
        if pyrad_array.shape != torch_array.shape:
            raise ValueError(
                f"Shape mismatch for {feature_name}: "
                f"{pyrad_array.shape} vs {torch_array.shape}"
            )

        valid = np.isfinite(pyrad_array) & np.isfinite(torch_array)
        pyrad_values = pyrad_array[valid]
        torch_values = torch_array[valid]

        abs_diff = np.abs(pyrad_values - torch_values)
        max_abs_diff = float(np.nanmax(abs_diff)) if abs_diff.size else 0.0
        mean_abs_diff = float(np.nanmean(abs_diff)) if abs_diff.size else 0.0

        with np.errstate(divide="ignore", invalid="ignore"):
            rel_diff = abs_diff / np.maximum(np.abs(pyrad_values), np.finfo(float).eps)
        max_rel_diff = float(np.nanmax(rel_diff)) if rel_diff.size else 0.0

        feature_allclose = bool(
            np.allclose(
                pyrad_values,
                torch_values,
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )
        )

        per_feature_stats.append(
            FeatureDiffStats(
                feature_name=feature_name,
                n_voxels=int(pyrad_values.size),
                max_abs_diff=max_abs_diff,
                mean_abs_diff=mean_abs_diff,
                max_rel_diff=max_rel_diff,
                allclose=feature_allclose,
            )
        )

        max_abs_diff_overall = max(max_abs_diff_overall, max_abs_diff)
        mean_abs_diff_overall = max(mean_abs_diff_overall, mean_abs_diff)
        max_rel_diff_overall = max(max_rel_diff_overall, max_rel_diff)
        all_close_overall = all_close_overall and feature_allclose

    summary = {
        "n_common_features": len(common_keys),
        "n_only_pyradiomics": len(only_pyrad),
        "n_only_torch": len(only_torch),
        "only_pyradiomics": only_pyrad,
        "only_torch": only_torch,
        "max_abs_diff": max_abs_diff_overall,
        "max_mean_abs_diff": mean_abs_diff_overall,
        "max_rel_diff": max_rel_diff_overall,
        "allclose": all_close_overall,
    }
    return per_feature_stats, summary


def print_feature_class_comparison(
    class_name: str,
    per_feature_stats: List[FeatureDiffStats],
    summary: Dict[str, Any],
) -> None:
    """Print comparison results for one feature class."""
    print(f"\n=== {class_name} ===")
    print(
        f"features: common={summary['n_common_features']}, "
        f"only_pyradiomics={summary['n_only_pyradiomics']}, "
        f"only_torch={summary['n_only_torch']}"
    )

    if summary["only_pyradiomics"]:
        print(f"  only in PyRadiomics: {summary['only_pyradiomics']}")
    if summary["only_torch"]:
        print(f"  only in TorchRadiomics: {summary['only_torch']}")

    for stats in per_feature_stats:
        print(
            f"  {stats.feature_name}: "
            f"n={stats.n_voxels}, "
            f"max_abs={stats.max_abs_diff:.6e}, "
            f"mean_abs={stats.mean_abs_diff:.6e}, "
            f"max_rel={stats.max_rel_diff:.6e}, "
            f"allclose={stats.allclose}"
        )

    print(
        f"  overall: max_abs={summary['max_abs_diff']:.6e}, "
        f"max_mean_abs={summary['max_mean_abs_diff']:.6e}, "
        f"max_rel={summary['max_rel_diff']:.6e}, "
        f"allclose={summary['allclose']}"
    )


def print_timing_summary(timing_stats: List[TimingStats]) -> None:
    """Print per-class and total CPU vs GPU timing."""
    print("\n=== Timing (execute only) ===")
    print(f"{'class':<12} {'cpu_s':>10} {'gpu_s':>10} {'speedup':>10}")
    total_cpu = 0.0
    total_gpu = 0.0
    for stats in timing_stats:
        total_cpu += stats.cpu_seconds
        total_gpu += stats.gpu_seconds
        print(
            f"{stats.feature_class:<12} "
            f"{stats.cpu_seconds:10.3f} "
            f"{stats.gpu_seconds:10.3f} "
            f"{stats.speedup:10.2f}x"
        )
    overall_speedup = total_cpu / total_gpu if total_gpu > 0.0 else float("inf")
    print(
        f"{'TOTAL':<12} {total_cpu:10.3f} {total_gpu:10.3f} {overall_speedup:10.2f}x"
    )


def main() -> None:
    _check_numpy_version()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    feature_class_names = get_enabled_torch_feature_classes(PARAMS_FILE)
    if not feature_class_names:
        raise ValueError(
            f"No torch-supported feature classes enabled in {PARAMS_FILE}. "
            f"Enable at least one of: {TORCH_SUPPORTED_FEATURE_CLASSES}"
        )

    print(f"torch={torch.__version__}, cuda={torch.cuda.is_available()}, device={device}")
    print(f"params_file={PARAMS_FILE}")
    print(f"kernelRadius={KERNEL_RADIUS}")
    print(f"enabled feature classes: {feature_class_names}")

    image, mask = load_image_mask(IMAGE_PATH, MASK_PATH)
    mask_norm = get_comparison_mask(image, mask, PARAMS_FILE, KERNEL_RADIUS)

    mask_voxel_count = int(np.sum(sitk.GetArrayFromImage(mask_norm) > 0))
    print(f"mask non-zero voxels: {mask_voxel_count}")

    global_allclose = True
    global_max_abs_diff = 0.0
    global_max_rel_diff = 0.0
    total_common_features = 0
    timing_stats: List[TimingStats] = []

    for class_name in feature_class_names:
        print(f"\nExtracting {class_name} ...")
        pyrad_features, cpu_seconds = extract_feature_maps(
            image,
            mask,
            PARAMS_FILE,
            KERNEL_RADIUS,
            class_name,
            device=None,
        )
        torch_features, gpu_seconds = extract_feature_maps(
            image,
            mask,
            PARAMS_FILE,
            KERNEL_RADIUS,
            class_name,
            device=device,
        )
        timing_stats.append(
            TimingStats(
                feature_class=class_name,
                cpu_seconds=cpu_seconds,
                gpu_seconds=gpu_seconds,
            )
        )

        print(
            f"  PyRadiomics: {len(pyrad_features)} feature map(s), "
            f"TorchRadiomics (injected): {len(torch_features)} feature map(s)"
        )
        print(f"  time: cpu={cpu_seconds:.3f}s, gpu={gpu_seconds:.3f}s, speedup={cpu_seconds / gpu_seconds:.2f}x")

        per_feature_stats, summary = compare_feature_maps(
            pyrad_features,
            torch_features,
        )
        print_feature_class_comparison(class_name, per_feature_stats, summary)

        global_allclose = global_allclose and bool(summary["allclose"])
        global_max_abs_diff = max(global_max_abs_diff, float(summary["max_abs_diff"]))
        global_max_rel_diff = max(global_max_rel_diff, float(summary["max_rel_diff"]))
        total_common_features += int(summary["n_common_features"])

    print_timing_summary(timing_stats)

    print("\n=== Overall ===")
    print(f"compared feature classes: {len(feature_class_names)}")
    print(f"total common features: {total_common_features}")
    print(f"global max_abs_diff: {global_max_abs_diff:.6e}")
    print(f"global max_rel_diff: {global_max_rel_diff:.6e}")
    print(f"global allclose: {global_allclose}")


if __name__ == "__main__":
    main()
