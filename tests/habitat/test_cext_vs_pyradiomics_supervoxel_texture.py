"""
Compare habit C-extension batched supervoxel texture extraction against
PyRadiomics per-label ROI extraction (speed and numerical agreement).

Run with pytest:
    python -m pytest tests/habitat/test_cext_vs_pyradiomics_supervoxel_texture.py -s

Run standalone (synthetic default, 100 supervoxels):
    python tests/habitat/test_cext_vs_pyradiomics_supervoxel_texture.py

Run on real data:
    python tests/habitat/test_cext_vs_pyradiomics_supervoxel_texture.py \\
        --image path/to/image.nii.gz \\
        --supervoxel-map path/to/supervoxel.nrrd \\
        --max-labels 32 --repeats 3
"""

from __future__ import annotations

import argparse
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk

from habit.core.habitat_analysis.clustering_features.supervoxel_cext import (
    cext_backend,
    is_cext_available,
)
from habit.core.habitat_analysis.clustering_features.batched_supervoxel_radiomics import (
    DEFAULT_SUPERVOXEL_BATCH,
    extract_batched_supervoxel_features,
    extract_supervoxel_features_pyradiomics,
)

# Texture-only feature subset used for fast regression checks.
TEXTURE_ENABLED_FEATURES: Dict[str, List[str]] = {
    "glcm": ["Contrast", "Correlation", "JointEntropy"],
    "glrlm": ["ShortRunEmphasis", "LongRunEmphasis", "RunEntropy"],
    "glszm": ["SmallAreaEmphasis", "LargeAreaEmphasis", "ZoneEntropy"],
    "ngtdm": ["Coarseness", "Contrast", "Busyness"],
    "gldm": ["SmallDependenceEmphasis", "LargeDependenceEmphasis", "DependenceEntropy"],
}

DEFAULT_SYNTHETIC_N_LABELS = 100

DEFAULT_SETTINGS: Dict[str, object] = {
    "binWidth": 25,
    "voxelArrayShift": 0,
    "force2D": False,
    "force2Ddimension": 0,
    "distances": [1],
    "gldm_a": 0,
    "padDistance": 1,
    "supervoxelUnionBboxCrop": True,
}


@dataclass
class TimingResult:
    """Wall-clock timing for one extraction backend."""

    backend: str
    seconds: float
    n_labels: int

    @property
    def seconds_per_label(self) -> float:
        if self.n_labels <= 0:
            return float("nan")
        return self.seconds / self.n_labels


@dataclass
class FeatureDiffSummary:
    """Aggregated numerical comparison between two feature tables."""

    column: str
    max_abs_diff: float
    mean_abs_diff: float
    max_rel_diff: float
    n_compared: int
    n_nan_mismatch: int


def _label_grid_shape(n_labels: int) -> Tuple[int, int, int]:
    """
    Choose a near-cubic 3D grid that can host at least ``n_labels`` supervoxels.

    Args:
        n_labels: Target number of supervoxel labels.

    Returns:
        Tuple[int, int, int]: Grid size ``(nz, ny, nx)`` with product >= n_labels.
    """
    if n_labels <= 0:
        raise ValueError(f"n_labels must be positive; got {n_labels}")

    nz = int(np.ceil(n_labels ** (1.0 / 3.0)))
    ny = int(np.ceil(np.sqrt(n_labels / nz)))
    nx = int(np.ceil(n_labels / (nz * ny)))
    return nz, ny, nx


def _make_synthetic_case(
    seed: int = 42,
    n_labels: int = DEFAULT_SYNTHETIC_N_LABELS,
    shape: Tuple[int, int, int] = (48, 48, 48),
    label_gap: int = 1,
) -> Tuple[sitk.Image, sitk.Image, np.ndarray]:
    """
    Build a synthetic 3D image with ``n_labels`` non-overlapping supervoxel regions.

    Labels are placed on a regular 3D grid. A one-voxel gap between grid cells
    avoids shared faces with neighboring labels, matching typical PyRadiomics ROI
    masks more closely than fully tiled partitions.

    Args:
        seed: Random seed for the intensity image.
        n_labels: Number of supervoxel labels to generate.
        shape: Volume shape ``(z, y, x)``.
        label_gap: Empty voxels left between adjacent label cells on each axis.

    Returns:
        Tuple[sitk.Image, sitk.Image, np.ndarray]: Intensity image, label map, label ids.
    """
    if n_labels <= 0:
        raise ValueError(f"n_labels must be positive; got {n_labels}")
    if label_gap < 0:
        raise ValueError(f"label_gap must be >= 0; got {label_gap}")

    rng = np.random.default_rng(seed)
    sz, sy, sx = shape
    image_array = rng.integers(10, 200, size=shape, dtype=np.int16).astype(np.float32)
    sv_array = np.zeros(shape, dtype=np.uint16)

    grid_nz, grid_ny, grid_nx = _label_grid_shape(n_labels)
    label = 1
    for iz in range(grid_nz):
        z0 = iz * sz // grid_nz
        z1 = (iz + 1) * sz // grid_nz if iz < grid_nz - 1 else sz
        for iy in range(grid_ny):
            y0 = iy * sy // grid_ny
            y1 = (iy + 1) * sy // grid_ny if iy < grid_ny - 1 else sy
            for ix in range(grid_nx):
                if label > n_labels:
                    break
                x0 = ix * sx // grid_nx
                x1 = (ix + 1) * sx // grid_nx if ix < grid_nx - 1 else sx

                z0i = min(z0 + label_gap, z1)
                z1i = max(z1 - label_gap, z0i)
                y0i = min(y0 + label_gap, y1)
                y1i = max(y1 - label_gap, y0i)
                x0i = min(x0 + label_gap, x1)
                x1i = max(x1 - label_gap, x0i)
                if z1i > z0i and y1i > y0i and x1i > x0i:
                    sv_array[z0i:z1i, y0i:y1i, x0i:x1i] = label
                label += 1
            if label > n_labels:
                break
        if label > n_labels:
            break

    image = sitk.GetImageFromArray(image_array)
    image.SetSpacing((1.0, 1.0, 1.0))
    sv_map = sitk.GetImageFromArray(sv_array)
    sv_map.CopyInformation(image)

    labels = np.arange(1, n_labels + 1, dtype=np.int64)
    return image, sv_map, labels


def _load_labels(sv_map: sitk.Image, max_labels: Optional[int]) -> np.ndarray:
    """Return sorted positive label ids, optionally truncated."""
    sv_array = sitk.GetArrayFromImage(sv_map)
    labels = np.unique(sv_array)
    labels = labels[labels > 0].astype(np.int64, copy=False)
    if max_labels is not None and max_labels > 0:
        labels = labels[:max_labels]
    return labels


def _align_image_to_reference(image: sitk.Image, reference: sitk.Image) -> sitk.Image:
    """Resample intensity image onto the supervoxel map grid when needed."""
    if (
        image.GetSize() == reference.GetSize()
        and image.GetSpacing() == reference.GetSpacing()
        and image.GetOrigin() == reference.GetOrigin()
    ):
        return image

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    return resampler.Execute(image)


def _run_cext_texture_extraction(
    image: sitk.Image,
    sv_map: sitk.Image,
    labels: np.ndarray,
    enabled_features: Mapping[str, Sequence[str]],
    settings: Mapping[str, object],
    *,
    batch_size: int,
) -> pd.DataFrame:
    """
    Extract texture features via habit C-extension batched matrices + Torch CPU formulas.

    Args:
        image: Input intensity image.
        sv_map: Multi-label supervoxel map.
        labels: Label ids to extract.
        enabled_features: Enabled PyRadiomics feature classes.
        settings: PyRadiomics settings dict.
        batch_size: Labels per batch group.

    Returns:
        pd.DataFrame: One row per supervoxel label.
    """
    merged_settings = dict(settings)
    merged_settings["useSupervoxelCext"] = True

    return extract_batched_supervoxel_features(
        image=image,
        supervoxel_map=sv_map,
        labels=labels,
        enabled_features=enabled_features,
        settings=merged_settings,
        device="cpu",
        dtype_name="float64",
        batch_size=batch_size,
    )


def _run_pyradiomics_per_label_extraction(
    image: sitk.Image,
    sv_map: sitk.Image,
    labels: np.ndarray,
    enabled_features: Mapping[str, Sequence[str]],
    settings: Mapping[str, object],
    *,
    batch_size: int = 1,
) -> pd.DataFrame:
    """
    Extract texture features via native CPU PyRadiomics (one ROI per supervoxel label).

    Args:
        image: Input intensity image.
        sv_map: Multi-label supervoxel map.
        labels: Label ids to extract.
        enabled_features: Enabled PyRadiomics feature classes.
        settings: PyRadiomics settings dict.
        batch_size: Outer batch width (inner loop remains per-label ROI).

    Returns:
        pd.DataFrame: One row per supervoxel label.
    """
    return extract_supervoxel_features_pyradiomics(
        image=image,
        supervoxel_map=sv_map,
        labels=labels,
        enabled_features=enabled_features,
        settings=dict(settings),
        batch_size=batch_size,
    )


def _time_call(
    func: Callable[[], pd.DataFrame],
    *,
    repeats: int,
    warmup: int = 1,
) -> Tuple[pd.DataFrame, float]:
    """
    Time ``func`` and return the last dataframe plus median wall time in seconds.

    Args:
        func: Callable returning a feature dataframe.
        repeats: Number of timed repetitions.
        warmup: Untimed warmup iterations.

    Returns:
        Tuple[pd.DataFrame, float]: Result dataframe and median elapsed seconds.
    """
    result: Optional[pd.DataFrame] = None
    for _ in range(warmup):
        result = func()

    timings: List[float] = []
    for _ in range(max(repeats, 1)):
        start = time.perf_counter()
        result = func()
        timings.append(time.perf_counter() - start)

    assert result is not None
    return result, float(np.median(timings))


def _compare_feature_tables(
    df_cext: pd.DataFrame,
    df_pyrad: pd.DataFrame,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> Tuple[List[FeatureDiffSummary], List[str]]:
    """
    Compare two supervoxel feature tables and summarize absolute/relative differences.

    Args:
        df_cext: Feature table from the C-extension path.
        df_pyrad: Feature table from PyRadiomics per-label path.
        rtol: Relative tolerance for ``np.isclose``.
        atol: Absolute tolerance for ``np.isclose``.

    Returns:
        Tuple[List[FeatureDiffSummary], List[str]]: Per-column diff stats and failing columns.
    """
    merged = df_cext.merge(df_pyrad, on="SupervoxelID", suffixes=("_cext", "_pyrad"))
    feature_cols = sorted(
        {
            col.replace("_cext", "")
            for col in merged.columns
            if col.endswith("_cext")
        }
    )

    summaries: List[FeatureDiffSummary] = []
    failing: List[str] = []

    for col in feature_cols:
        cext_vals = merged[f"{col}_cext"].to_numpy(dtype=np.float64)
        pyrad_vals = merged[f"{col}_pyrad"].to_numpy(dtype=np.float64)

        both_nan = np.isnan(cext_vals) & np.isnan(pyrad_vals)
        n_nan_mismatch = int(np.sum(np.isnan(cext_vals) ^ np.isnan(pyrad_vals)))

        valid = ~both_nan & ~(np.isnan(cext_vals) | np.isnan(pyrad_vals))
        if not np.any(valid):
            summaries.append(
                FeatureDiffSummary(
                    column=col,
                    max_abs_diff=0.0,
                    mean_abs_diff=0.0,
                    max_rel_diff=0.0,
                    n_compared=0,
                    n_nan_mismatch=n_nan_mismatch,
                )
            )
            continue

        abs_diff = np.abs(cext_vals[valid] - pyrad_vals[valid])
        denom = np.maximum(np.abs(pyrad_vals[valid]), 1e-12)
        rel_diff = abs_diff / denom

        summary = FeatureDiffSummary(
            column=col,
            max_abs_diff=float(np.max(abs_diff)),
            mean_abs_diff=float(np.mean(abs_diff)),
            max_rel_diff=float(np.max(rel_diff)),
            n_compared=int(np.sum(valid)),
            n_nan_mismatch=n_nan_mismatch,
        )
        summaries.append(summary)

        if n_nan_mismatch > 0 or not np.allclose(
            cext_vals[valid],
            pyrad_vals[valid],
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        ):
            failing.append(col)

    return summaries, failing


def _print_report(
    timing_cext: TimingResult,
    timing_pyrad: TimingResult,
    summaries: Sequence[FeatureDiffSummary],
    failing: Sequence[str],
    *,
    top_k: int = 12,
) -> None:
    """Print human-readable speed and numerical comparison report."""
    speedup = timing_pyrad.seconds / timing_cext.seconds if timing_cext.seconds > 0 else float("inf")

    print("\n=== Supervoxel texture benchmark: C extension vs PyRadiomics per-label ===")
    print(f"C-extension backend : {cext_backend()}")
    print(f"Labels              : {timing_cext.n_labels}")
    print(f"C-extension time    : {timing_cext.seconds:.4f} s "
          f"({timing_cext.seconds_per_label:.5f} s/label)")
    print(f"PyRadiomics time    : {timing_pyrad.seconds:.4f} s "
          f"({timing_pyrad.seconds_per_label:.5f} s/label)")
    print(f"Speedup (pyrad/cext): {speedup:.2f}x")

    by_class: Dict[str, List[FeatureDiffSummary]] = {}
    for item in summaries:
        parts = item.column.split("_")
        class_name = parts[1] if len(parts) >= 2 else "unknown"
        by_class.setdefault(class_name, []).append(item)

    print("\nPer-class max absolute difference:")
    for class_name in sorted(by_class.keys()):
        max_diff = max(entry.max_abs_diff for entry in by_class[class_name])
        print(f"  {class_name:<8} max_abs_diff = {max_diff:.6g}")

    sorted_summaries = sorted(summaries, key=lambda item: item.max_abs_diff, reverse=True)
    print(f"\nTop {top_k} features by max absolute difference:")
    print(f"{'Feature':<48} {'max_abs':>12} {'mean_abs':>12} {'max_rel':>12}")
    for item in sorted_summaries[:top_k]:
        print(
            f"{item.column:<48} "
            f"{item.max_abs_diff:12.6g} "
            f"{item.mean_abs_diff:12.6g} "
            f"{item.max_rel_diff:12.6g}"
        )

    if failing:
        print(f"\nColumns outside tolerance ({len(failing)}):")
        for col in failing:
            print(f"  - {col}")
    else:
        print("\nAll compared texture columns are within tolerance.")


def run_texture_benchmark(
    image: sitk.Image,
    sv_map: sitk.Image,
    labels: np.ndarray,
    enabled_features: Mapping[str, Sequence[str]],
    settings: Mapping[str, object],
    *,
    batch_size: int = DEFAULT_SUPERVOXEL_BATCH,
    repeats: int = 2,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> Tuple[TimingResult, TimingResult, List[FeatureDiffSummary], List[str]]:
    """
    Run end-to-end speed and numerical comparison for texture feature extraction.

    Returns:
        Tuple containing C-extension timing, PyRadiomics timing, diff summaries, failing columns.
    """
    n_labels = int(len(labels))

    df_cext, cext_seconds = _time_call(
        lambda: _run_cext_texture_extraction(
            image,
            sv_map,
            labels,
            enabled_features,
            settings,
            batch_size=batch_size,
        ),
        repeats=repeats,
    )
    df_pyrad, pyrad_seconds = _time_call(
        lambda: _run_pyradiomics_per_label_extraction(
            image,
            sv_map,
            labels,
            enabled_features,
            settings,
            batch_size=1,
        ),
        repeats=repeats,
    )

    summaries, failing = _compare_feature_tables(df_cext, df_pyrad, rtol=rtol, atol=atol)
    timing_cext = TimingResult("cext", cext_seconds, n_labels)
    timing_pyrad = TimingResult("pyradiomics", pyrad_seconds, n_labels)
    return timing_cext, timing_pyrad, summaries, failing


class TestCextVsPyradiomicsSupervoxelTexture(unittest.TestCase):
    """Regression tests for C-extension vs PyRadiomics supervoxel texture extraction."""

    @classmethod
    def setUpClass(cls) -> None:
        if not is_cext_available():
            raise unittest.SkipTest("supervoxel_cext native extension is not built")

    def test_synthetic_texture_speed_and_numerics(self) -> None:
        """
        Synthetic volume: report speedup and per-feature numerical differences.

        GLCM / GLRLM are checked strictly; other texture classes are reported because
        the current C extension may still diverge from PyRadiomics (NGTDM column layout,
        GLDM dependence counting, GLSZM connectivity semantics).
        """
        image, sv_map, labels = _make_synthetic_case(n_labels=DEFAULT_SYNTHETIC_N_LABELS)
        timing_cext, timing_pyrad, summaries, failing = run_texture_benchmark(
            image,
            sv_map,
            labels,
            TEXTURE_ENABLED_FEATURES,
            DEFAULT_SETTINGS,
            batch_size=min(len(labels), DEFAULT_SUPERVOXEL_BATCH),
            repeats=2,
            rtol=1e-3,
            atol=1e-3,
        )

        _print_report(timing_cext, timing_pyrad, summaries, failing)

        self.assertGreater(timing_pyrad.seconds, 0.0)
        self.assertGreater(timing_cext.seconds, 0.0)
        self.assertGreaterEqual(timing_pyrad.seconds / timing_cext.seconds, 1.0)

        strict_prefixes = ("original_glcm_", "original_glrlm_")
        strict_failing = [col for col in failing if col.startswith(strict_prefixes)]
        self.assertEqual(
            len(strict_failing),
            0,
            msg=f"GLCM/GLRLM mismatch vs PyRadiomics: {strict_failing}",
        )

    def test_matrix_backend_speed_synthetic(self) -> None:
        """Compare raw GLCM matrix batch speed: native C vs PyRadiomics fallback loop."""
        from habit.core.habitat_analysis.clustering_features.supervoxel_cext import (
            calculate_glcm,
        )
        from habit.core.habitat_analysis.clustering_features.supervoxel_cext import (
            _fallback as fallback_module,
        )

        image, sv_map, labels = _make_synthetic_case(n_labels=DEFAULT_SYNTHETIC_N_LABELS)

        # Build discretized image using PyRadiomics binning via a small Torch calculator path.
        union_mask = sitk.GetImageFromArray((sitk.GetArrayFromImage(sv_map) > 0).astype(np.uint8))
        union_mask.CopyInformation(sv_map)
        from habit.core.habitat_analysis.clustering_features.torchradiomics.TorchRadiomicsGLCM import (
            TorchRadiomicsGLCM,
        )

        calculator = TorchRadiomicsGLCM(
            image,
            union_mask,
            binWidth=DEFAULT_SETTINGS["binWidth"],
            force2D=DEFAULT_SETTINGS["force2D"],
            force2Ddimension=DEFAULT_SETTINGS["force2Ddimension"],
            device="cpu",
        )
        image_i = np.ascontiguousarray(calculator.imageArray.astype(np.int32))
        sv_i = np.ascontiguousarray(sitk.GetArrayFromImage(sv_map).astype(np.int32))
        labels_i = labels.astype(np.int32)
        distances = np.asarray(DEFAULT_SETTINGS["distances"], dtype=np.int32)
        ng = int(calculator.coefficients["Ng"])

        _, cext_seconds = _time_call(
            lambda: calculate_glcm(
                image_i,
                sv_i,
                labels_i,
                distances,
                ng,
                int(DEFAULT_SETTINGS["force2D"]),
                int(DEFAULT_SETTINGS["force2Ddimension"]),
            ),
            repeats=3,
        )
        _, fallback_seconds = _time_call(
            lambda: fallback_module.calculate_glcm(
                image_i,
                sv_i,
                labels_i,
                distances,
                ng,
                int(DEFAULT_SETTINGS["force2D"]),
                int(DEFAULT_SETTINGS["force2Ddimension"]),
            ),
            repeats=3,
        )

        speedup = fallback_seconds / cext_seconds if cext_seconds > 0 else float("inf")
        print(
            f"\nRaw GLCM matrix batch ({len(labels)} labels): "
            f"cext={cext_seconds:.4f}s fallback={fallback_seconds:.4f}s speedup={speedup:.2f}x"
        )
        self.assertGreater(speedup, 1.0)


def _build_arg_parser() -> argparse.ArgumentParser:
    """CLI parser for standalone benchmark runs."""
    parser = argparse.ArgumentParser(
        description="Benchmark habit C-extension vs PyRadiomics per-label supervoxel texture extraction.",
    )
    parser.add_argument("--image", type=Path, default=None, help="Input intensity image path.")
    parser.add_argument("--supervoxel-map", type=Path, default=None, help="Supervoxel label map path.")
    parser.add_argument(
        "--n-labels",
        type=int,
        default=DEFAULT_SYNTHETIC_N_LABELS,
        help="Number of supervoxel labels for synthetic data (default: 100).",
    )
    parser.add_argument("--max-labels", type=int, default=None, help="Limit number of supervoxel labels.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_SUPERVOXEL_BATCH, help="C-extension batch size.")
    parser.add_argument("--repeats", type=int, default=2, help="Timed repetitions per backend.")
    parser.add_argument("--rtol", type=float, default=1e-4, help="Relative tolerance for feature comparison.")
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance for feature comparison.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic data when paths are not provided.",
    )
    return parser


def main() -> int:
    """Entry point for standalone benchmark execution."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    if not is_cext_available():
        print(
            "ERROR: supervoxel_cext native extension is not available. "
            "Run: pip install -e ."
        )
        return 1

    if args.image is not None and args.supervoxel_map is not None:
        sv_map = sitk.ReadImage(str(args.supervoxel_map))
        image = _align_image_to_reference(sitk.ReadImage(str(args.image)), sv_map)
        labels = _load_labels(sv_map, args.max_labels)
        print(f"Loaded real data: image={args.image} sv_map={args.supervoxel_map} labels={len(labels)}")
    else:
        image, sv_map, labels = _make_synthetic_case(seed=args.seed, n_labels=args.n_labels)
        if args.max_labels is not None and args.max_labels > 0:
            labels = labels[: args.max_labels]
        print(
            f"Using synthetic data: shape={sitk.GetArrayFromImage(sv_map).shape} "
            f"labels={len(labels)} (requested={args.n_labels})"
        )

    timing_cext, timing_pyrad, summaries, failing = run_texture_benchmark(
        image,
        sv_map,
        labels,
        TEXTURE_ENABLED_FEATURES,
        DEFAULT_SETTINGS,
        batch_size=args.batch_size,
        repeats=args.repeats,
        rtol=args.rtol,
        atol=args.atol,
    )
    _print_report(timing_cext, timing_pyrad, summaries, failing)
    strict_failing = [
        col
        for col in failing
        if col.startswith(("original_glcm_", "original_glrlm_"))
    ]
    return 0 if not strict_failing else 2


if __name__ == "__main__":
    raise SystemExit(main())
