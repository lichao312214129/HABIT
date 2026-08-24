# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Native C + vectorized-CPU supervoxel / habitat texture extract.

Hot path for the 0.5 s budget:

1. One numpy crop of the union-mask bounding box (+ ``padDistance``).
2. One discretize: per-label digitize stitched into one volume
   (default ``union_bin=False``, ``execute()`` science) or one union
   ``getBinEdges`` when ``union_bin=True``.
3. One C-extension pass per texture class for **all** labels.
4. Stacked numpy formulas (no TorchRadiomics / SimpleITK calculator).

No per-label ``execute``, ``checkMask``, ``sitk.Hash``, or ``cuda.empty_cache``.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk

from habit.kernels.radiomics.cext import (
    calculate_firstorder,
    calculate_gldm,
    calculate_glcm,
    calculate_glrlm,
    calculate_glszm,
    calculate_ngtdm,
    cext_backend,
    is_cext_available,
)
from habit.kernels.radiomics.cpu_formulas import (
    assign_feature_columns,
    entropy_uniformity_from_disc,
    firstorder_from_cext_stats,
    firstorder_features,
    gldm_features,
    glcm_features,
    glrlm_features,
    glszm_features,
    ngtdm_features,
)
from habit.kernels.radiomics.supervoxel_batch import (
    DEFAULT_FEATURES_BY_CLASS,
    DEFAULT_SUPERVOXEL_PAD_DISTANCE,
    _resolve_enabled_features,
    _resolve_supervoxel_pad_distance,
    _should_crop_union_bbox,
)

logger = logging.getLogger(__name__)


def _profile_enabled() -> bool:
    """Return True when ``HABIT_SV_PROFILE`` requests millisecond timestamps."""
    flag = os.environ.get("HABIT_SV_PROFILE", "")
    return flag not in ("", "0", "false", "False")


def _now() -> float:
    """Monotonic seconds for crop / bin / C / formula breakdowns."""
    return time.perf_counter()


def _numpy_union_crop(
    image: np.ndarray,
    sv_map: np.ndarray,
    pad_distance: int,
) -> Tuple[np.ndarray, np.ndarray, Tuple[slice, ...]]:
    """
    Crop both arrays to the union-mask bounding box plus pad.

    Args:
        image: Intensity volume (z, y, x).
        sv_map: Multi-label map of the same shape.
        pad_distance: Voxels of padding on every side (clipped to bounds).

    Returns:
        Tuple[np.ndarray, np.ndarray, Tuple[slice, ...]]: Cropped image,
        cropped map, and the slices applied (for debugging).
    """
    if image.shape != sv_map.shape:
        raise ValueError(
            f"image shape {image.shape} must match sv_map shape {sv_map.shape}"
        )
    coords = np.where(sv_map > 0)
    if coords[0].size == 0:
        raise ValueError("Supervoxel map has no non-zero labels.")
    slices: List[slice] = []
    for axis, idx in enumerate(coords):
        lo = int(idx.min()) - pad_distance
        hi = int(idx.max()) + 1 + pad_distance
        lo = max(0, lo)
        hi = min(image.shape[axis], hi)
        slices.append(slice(lo, hi))
    crop = tuple(slices)
    return np.ascontiguousarray(image[crop]), np.ascontiguousarray(sv_map[crop]), crop


def _sitk_union_bbox_crop(
    image: sitk.Image,
    supervoxel_map: sitk.Image,
    pad_distance: int,
) -> Tuple[sitk.Image, sitk.Image]:
    """
    Crop SimpleITK images to the union-label bounding box before numpy copy.

    Args:
        image: Intensity volume.
        supervoxel_map: Multi-label map aligned with ``image``.
        pad_distance: Voxels of padding, clipped to the image bounds.

    Returns:
        Tuple[sitk.Image, sitk.Image]: Cropped intensity and label images.
    """
    pad = max(0, int(pad_distance))
    dim = int(image.GetDimension())
    size_xyz = [int(v) for v in image.GetSize()]
    # LabelShapeStatistics walks the volume in C++ and returns the
    # union bounding box. Avoid GetArrayView/nonzero on a 512^3 lattice.
    binary = sitk.BinaryThreshold(supervoxel_map, 1, 2147483647, 1, 0)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(binary)
    if stats.GetNumberOfLabels() < 1:
        raise ValueError("Supervoxel map has no non-zero labels.")
    # SimpleITK: (min_x, min_y[, min_z], size_x, size_y[, size_z]).
    bbox = [int(v) for v in stats.GetBoundingBox(1)]
    if dim == 2:
        x0, y0, sx, sy = bbox
        index = [max(0, x0 - pad), max(0, y0 - pad)]
        hi = [min(size_xyz[0], x0 + sx + pad), min(size_xyz[1], y0 + sy + pad)]
    else:
        x0, y0, z0, sx, sy, sz = bbox
        index = [max(0, x0 - pad), max(0, y0 - pad), max(0, z0 - pad)]
        hi = [
            min(size_xyz[0], x0 + sx + pad),
            min(size_xyz[1], y0 + sy + pad),
            min(size_xyz[2], z0 + sz + pad),
        ]
    roi_size = [hi[i] - index[i] for i in range(dim)]
    cropped_image = sitk.RegionOfInterest(image, roi_size, index)
    cropped_map = sitk.RegionOfInterest(supervoxel_map, roi_size, index)
    return cropped_image, cropped_map


def _bin_edges(values: np.ndarray, settings: Mapping[str, object]) -> np.ndarray:
    """
    PyRadiomics ``getBinEdges`` for the given ROI intensities.

    Args:
        values: 1-D ROI intensities.
        settings: Must carry ``binWidth`` and/or ``binCount``.

    Returns:
        np.ndarray: Bin edges for ``np.digitize``.
    """
    from radiomics.imageoperations import getBinEdges

    return np.asarray(
        getBinEdges(
            np.asarray(values, dtype=np.float64).reshape(-1),
            binWidth=settings.get("binWidth", 25),
            binCount=settings.get("binCount"),
        ),
        dtype=np.float64,
    )


def _discretize_union(
    image: np.ndarray,
    sv_map: np.ndarray,
    settings: Mapping[str, object],
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Discretize every foreground voxel with one shared ``getBinEdges``.

    Args:
        image: Cropped intensity volume.
        sv_map: Cropped multi-label map.
        settings: PyRadiomics bin settings.

    Returns:
        Tuple[np.ndarray, int, np.ndarray]: 1-indexed int32 bins, ``Ng``,
        and a length-1 array of that ``Ng`` (one value for every label).
    """
    roi = sv_map > 0
    edges = _bin_edges(image[roi], settings)
    disc = np.zeros(image.shape, dtype=np.int32)
    disc[roi] = np.digitize(image[roi], edges).astype(np.int32, copy=False)
    ng = int(disc[roi].max()) if np.any(roi) else 1
    return disc, max(ng, 1), np.asarray([max(ng, 1)], dtype=np.int32)


def _discretize_per_label(
    image: np.ndarray,
    sv_map: np.ndarray,
    labels: Sequence[int],
    settings: Mapping[str, object],
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Per-label ``getBinEdges`` (execute() science) stitched into one volume.

    Args:
        image: Cropped intensity volume.
        sv_map: Cropped multi-label map.
        labels: Label ids in extract order.
        settings: PyRadiomics bin settings.

    Returns:
        Tuple[np.ndarray, int, np.ndarray]: Stitched 1-indexed bins, max
        ``Ng`` across labels, and per-label ``Ng``.
    """
    disc = np.zeros(image.shape, dtype=np.int32)
    ngs: List[int] = []
    for label in labels:
        roi = sv_map == int(label)
        if not np.any(roi):
            ngs.append(1)
            continue
        edges = _bin_edges(image[roi], settings)
        bins = np.digitize(image[roi], edges).astype(np.int32, copy=False)
        disc[roi] = bins
        ngs.append(int(bins.max()) if bins.size else 1)
    max_ng = max(ngs) if ngs else 1
    return disc, max(max_ng, 1), np.asarray(ngs, dtype=np.int32)


def _maybe_normalize_full(
    image_sitk: sitk.Image,
    settings: Mapping[str, object],
) -> sitk.Image:
    """
    Apply PyRadiomics whole-image normalize when ``normalize`` is set.

    ``execute()`` normalizes the full volume before crop; doing the same
    here keeps union-crop statistics identical.

    Args:
        image_sitk: Intensity SimpleITK image.
        settings: PyRadiomics settings.

    Returns:
        sitk.Image: Possibly normalised image (same object when disabled).
    """
    if not settings.get("normalize", False):
        return image_sitk
    from radiomics import imageoperations

    return imageoperations.normalizeImage(image_sitk, **dict(settings))


def extract_native_supervoxel_features(
    image: sitk.Image,
    supervoxel_map: sitk.Image,
    labels: np.ndarray,
    *,
    enabled_features: Mapping[str, object],
    settings: Optional[Dict[str, object]] = None,
    image_name: str = "",
    union_bin: bool = False,
    progress_callback: Optional[Callable[[int], None]] = None,
    timings: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Extract per-label texture with one crop, one discretize, one C pass.

    Args:
        image: Intensity SimpleITK image.
        supervoxel_map: Multi-label map aligned with ``image``.
        labels: 1-D label ids to extract.
        enabled_features: PyRadiomics ``enabledFeatures`` mapping.
        settings: PyRadiomics / habit settings.
        image_name: Optional column suffix.
        union_bin: False (default) = per-label ``binWidth``, matching
            PyRadiomics ``execute()`` and ``each_habitat``. True = one
            shared bin on the union mask (optional shared gray scale).
        progress_callback: Optional callback invoked once with ``K``.
        timings: Optional dict filled with millisecond breakdowns
            (``crop_ms``, ``bin_ms``, ``c_<class>_ms``, ``formula_ms``,
            ``assemble_ms``, ``total_ms``).

    Returns:
        pd.DataFrame: One row per label with ``supervoxel_id``.
    """
    settings = dict(settings or {})
    if "union_bin" in settings:
        union_bin = bool(settings["union_bin"])
    resolved = _resolve_enabled_features(enabled_features)
    if not resolved:
        raise ValueError("No enabled supervoxel radiomics feature classes.")

    label_ids = [int(v) for v in np.asarray(labels, dtype=np.int64).reshape(-1)]
    labels_i = np.asarray(label_ids, dtype=np.int32)
    clock: Dict[str, float] = {}
    t_all = _now()

    t0 = _now()
    image = _maybe_normalize_full(image, settings)
    if (
        settings.get("interpolator") is not None
        and settings.get("resampledPixelSpacing") is not None
    ):
        from radiomics import imageoperations

        image, supervoxel_map = imageoperations.resampleImage(
            image, supervoxel_map, **settings
        )

    pad = _resolve_supervoxel_pad_distance(settings) if _should_crop_union_bbox(settings) else 0
    if _should_crop_union_bbox(settings):
        image, supervoxel_map = _sitk_union_bbox_crop(image, supervoxel_map, pad)
    image_np = np.ascontiguousarray(sitk.GetArrayFromImage(image).astype(np.float64))
    sv_np = np.ascontiguousarray(sitk.GetArrayFromImage(supervoxel_map).astype(np.int32))
    if image_np.shape != sv_np.shape:
        raise ValueError(
            f"image array {image_np.shape} must match supervoxel map {sv_np.shape}"
        )
    spacing = tuple(float(v) for v in image.GetSpacing())
    voxel_volume = float(spacing[0] * spacing[1] * spacing[2])
    voxel_shift = float(settings.get("voxelArrayShift", 0))
    clock["crop_ms"] = (_now() - t0) * 1000.0

    t0 = _now()
    if union_bin:
        disc, ng, ng_per_label = _discretize_union(image_np, sv_np, settings)
    else:
        disc, ng, ng_per_label = _discretize_per_label(
            image_np, sv_np, label_ids, settings
        )
    disc_i = np.ascontiguousarray(disc)
    sv_i = np.ascontiguousarray(sv_np.astype(np.int32))
    clock["bin_ms"] = (_now() - t0) * 1000.0

    max_lab = int(labels_i.max()) if labels_i.size else 0
    n_voxels = np.bincount(
        sv_i.ravel(), minlength=max_lab + 1
    )[labels_i].astype(np.float64)
    force2d = int(bool(settings.get("force2D", False)))
    force2d_dim = int(settings.get("force2Ddimension", 0))
    distances = np.asarray(settings.get("distances", [1]), dtype=np.int32)
    alpha = int(settings.get("gldm_a", 0))
    nr = int(max(disc_i.shape)) if disc_i.size else 1
    bin_width = float(settings.get("binWidth", 25))
    gray_levels = np.arange(1, ng + 1, dtype=np.float64)

    logger.info(
        "native supervoxel extract: backend=%s union_bin=%s labels=%d Ng=%d "
        "cropped=%s pad=%d classes=%s",
        cext_backend(),
        union_bin,
        len(label_ids),
        ng,
        disc_i.shape,
        pad,
        sorted(resolved.keys()),
    )
    if not is_cext_available():
        logger.warning(
            "native extract is running on the per-label cMatrices fallback "
            "(cext_backend=%s); rebuild with pip install -e .",
            cext_backend(),
        )

    columns: Dict[str, object] = {"supervoxel_id": np.asarray(label_ids, dtype=np.int64)}

    if "firstorder" in resolved:
        t0 = _now()
        names = list(resolved["firstorder"]) or list(DEFAULT_FEATURES_BY_CLASS["firstorder"])
        hist_names = [name for name in names if name in ("Entropy", "Uniformity")]
        other_names = [name for name in names if name not in ("Entropy", "Uniformity")]
        values: Dict[str, np.ndarray] = {}
        if other_names and cext_backend() == "native":
            t_c = _now()
            stats = calculate_firstorder(
                image_np,
                sv_i,
                labels_i,
                ng,
                bin_width,
                voxel_shift,
                voxel_volume,
            )
            clock["c_firstorder_ms"] = (_now() - t_c) * 1000.0
            values.update(
                firstorder_from_cext_stats(
                    stats,
                    other_names,
                    n_voxels=n_voxels,
                    voxel_array_shift=voxel_shift,
                )
            )
        elif other_names:
            values.update(
                firstorder_features(
                    image_np,
                    sv_i,
                    label_ids,
                    other_names,
                    discretized=disc_i,
                    voxel_array_shift=voxel_shift,
                    voxel_volume=voxel_volume,
                )
            )
        if hist_names:
            values.update(
                entropy_uniformity_from_disc(
                    disc_i, sv_i, label_ids, ng, hist_names
                )
            )
        assign_feature_columns(columns, "firstorder", values, image_name)
        clock["formula_firstorder_ms"] = (_now() - t0) * 1000.0

    if "glcm" in resolved:
        t0 = _now()
        p_glcm, _angles = calculate_glcm(
            disc_i, sv_i, labels_i, distances, ng, force2d, force2d_dim
        )
        clock["c_glcm_ms"] = (_now() - t0) * 1000.0
        t0 = _now()
        names = list(resolved["glcm"]) or list(DEFAULT_FEATURES_BY_CLASS["glcm"])
        values = glcm_features(
            p_glcm,
            names,
            gray_levels=gray_levels,
            ng_full=ng_per_label,
            symmetrical=bool(settings.get("symmetricalGLCM", True)),
        )
        assign_feature_columns(columns, "glcm", values, image_name)
        clock["formula_glcm_ms"] = (_now() - t0) * 1000.0

    if "glrlm" in resolved:
        t0 = _now()
        p_glrlm, _angles = calculate_glrlm(
            disc_i, sv_i, labels_i, ng, nr, force2d, force2d_dim
        )
        clock["c_glrlm_ms"] = (_now() - t0) * 1000.0
        t0 = _now()
        names = list(resolved["glrlm"]) or list(DEFAULT_FEATURES_BY_CLASS["glrlm"])
        values = glrlm_features(p_glrlm, names, gray_levels=gray_levels)
        assign_feature_columns(columns, "glrlm", values, image_name)
        clock["formula_glrlm_ms"] = (_now() - t0) * 1000.0

    if "glszm" in resolved:
        t0 = _now()
        p_glszm = calculate_glszm(disc_i, sv_i, labels_i, ng, force2d, force2d_dim)
        clock["c_glszm_ms"] = (_now() - t0) * 1000.0
        t0 = _now()
        names = list(resolved["glszm"]) or list(DEFAULT_FEATURES_BY_CLASS["glszm"])
        values = glszm_features(
            p_glszm, names, gray_levels=gray_levels, n_voxels=n_voxels
        )
        assign_feature_columns(columns, "glszm", values, image_name)
        clock["formula_glszm_ms"] = (_now() - t0) * 1000.0

    if "gldm" in resolved:
        t0 = _now()
        p_gldm = calculate_gldm(
            disc_i, sv_i, labels_i, distances, ng, alpha, force2d, force2d_dim
        )
        clock["c_gldm_ms"] = (_now() - t0) * 1000.0
        t0 = _now()
        names = list(resolved["gldm"]) or list(DEFAULT_FEATURES_BY_CLASS["gldm"])
        values = gldm_features(p_gldm, names, gray_levels=gray_levels)
        assign_feature_columns(columns, "gldm", values, image_name)
        clock["formula_gldm_ms"] = (_now() - t0) * 1000.0

    if "ngtdm" in resolved:
        t0 = _now()
        p_ngtdm = calculate_ngtdm(
            disc_i, sv_i, labels_i, distances, ng, force2d, force2d_dim
        )
        clock["c_ngtdm_ms"] = (_now() - t0) * 1000.0
        t0 = _now()
        names = list(resolved["ngtdm"]) or list(DEFAULT_FEATURES_BY_CLASS["ngtdm"])
        values = ngtdm_features(p_ngtdm, names)
        assign_feature_columns(columns, "ngtdm", values, image_name)
        clock["formula_ngtdm_ms"] = (_now() - t0) * 1000.0

    t0 = _now()
    frame = pd.DataFrame(columns)
    clock["assemble_ms"] = (_now() - t0) * 1000.0
    clock["total_ms"] = (_now() - t_all) * 1000.0
    clock["formula_ms"] = sum(
        v for k, v in clock.items() if k.startswith("formula_")
    )
    clock["c_ms"] = sum(v for k, v in clock.items() if k.startswith("c_"))
    if timings is not None:
        timings.update(clock)
    if _profile_enabled() or timings is not None:
        logger.info(
            "native extract timings (ms): crop=%.2f bin=%.2f C=%.2f "
            "formula=%.2f assemble=%.2f total=%.2f detail=%s",
            clock.get("crop_ms", 0.0),
            clock.get("bin_ms", 0.0),
            clock.get("c_ms", 0.0),
            clock.get("formula_ms", 0.0),
            clock.get("assemble_ms", 0.0),
            clock.get("total_ms", 0.0),
            {k: round(v, 3) for k, v in clock.items()},
        )
    if progress_callback is not None:
        progress_callback(len(label_ids))
    return frame


# Re-export pad default so tests can import a single module.
DEFAULT_PAD = DEFAULT_SUPERVOXEL_PAD_DISTANCE
