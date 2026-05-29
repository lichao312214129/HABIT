"""
Per-label PyRadiomics ``cMatrices`` fallback when the native C extension is unavailable.

Each function mirrors the ``supervoxel_cext._sv_cmatrices`` API and stacks results on axis 0.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _as_int32_1d(labels: np.ndarray) -> np.ndarray:
    """Ensure labels is a contiguous int32 1D array."""
    arr = np.asarray(labels, dtype=np.int32).reshape(-1)
    return np.ascontiguousarray(arr)


def _as_int32_image(image: np.ndarray) -> np.ndarray:
    """Ensure discretized image is contiguous int32."""
    return np.ascontiguousarray(np.asarray(image, dtype=np.int32))


def _as_int32_map(sv_map: np.ndarray) -> np.ndarray:
    """Ensure supervoxel map is contiguous int32."""
    return np.ascontiguousarray(np.asarray(sv_map, dtype=np.int32))


def _label_mask(sv_map: np.ndarray, label: int) -> np.ndarray:
    """Build boolean ROI mask for one supervoxel label."""
    return np.ascontiguousarray(sv_map == label)


def _ensure_batch_leading(array: np.ndarray) -> np.ndarray:
    """Ensure matrix output has a leading batch dimension."""
    arr = np.asarray(array)
    if arr.ndim == 2:
        return arr[np.newaxis, ...]
    if arr.ndim == 3:
        return arr[np.newaxis, ...]
    return arr


def calculate_glcm(
    image: np.ndarray,
    sv_map: np.ndarray,
    labels: np.ndarray,
    distances: np.ndarray,
    Ng: int,
    force2D: int = 0,
    force2Ddimension: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate GLCM matrices for multiple supervoxel labels via PyRadiomics.

    Args:
        image: Discretized int32 image (1-indexed gray levels).
        sv_map: int32 multi-label supervoxel map.
        labels: int32 1D label ids.
        distances: int32 1D distance offsets.
        Ng: Number of gray levels.
        force2D: PyRadiomics force2D flag (0/1).
        force2Ddimension: Axis to collapse when force2D is enabled.

    Returns:
        Tuple[np.ndarray, np.ndarray]: ``(P_glcm, angles)`` with shapes
        ``(n_labels, Ng, Ng, n_angles)`` and ``(n_angles, 3)``.
    """
    from radiomics import cMatrices

    image_i = _as_int32_image(image)
    sv_map_i = _as_int32_map(sv_map)
    labels_i = _as_int32_1d(labels)
    distances_i = np.ascontiguousarray(np.asarray(distances, dtype=np.int32).reshape(-1))

    p_list = []
    angles_ref = None
    for label in labels_i.tolist():
        mask = _label_mask(sv_map_i, int(label))
        p_glcm, angles = cMatrices.calculate_glcm(
            image_i,
            mask,
            distances_i,
            int(Ng),
            bool(force2D),
            int(force2Ddimension),
        )
        p_arr = _ensure_batch_leading(np.asarray(p_glcm, dtype=np.float64))
        p_list.append(p_arr[0])
        angles_ref = np.asarray(angles, dtype=np.int32)

    if not p_list:
        empty = np.zeros((0, Ng, Ng, 0), dtype=np.float64)
        return empty, np.zeros((0, 3), dtype=np.int32)

    return np.stack(p_list, axis=0), angles_ref


def calculate_glrlm(
    image: np.ndarray,
    sv_map: np.ndarray,
    labels: np.ndarray,
    Ng: int,
    Nr: int,
    force2D: int = 0,
    force2Ddimension: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate GLRLM matrices for multiple supervoxel labels via PyRadiomics.

    Returns:
        Tuple[np.ndarray, np.ndarray]: ``(P_glrlm, angles)`` with shapes
        ``(n_labels, Ng, Nr, n_angles)`` and ``(n_angles, 3)``.
    """
    from radiomics import cMatrices

    image_i = _as_int32_image(image)
    sv_map_i = _as_int32_map(sv_map)
    labels_i = _as_int32_1d(labels)
    nr = int(Nr) if Nr > 0 else int(max(image_i.shape))

    p_list = []
    angles_ref = None
    for label in labels_i.tolist():
        mask = _label_mask(sv_map_i, int(label))
        p_glrlm, angles = cMatrices.calculate_glrlm(
            image_i,
            mask,
            int(Ng),
            nr,
            bool(force2D),
            int(force2Ddimension),
        )
        p_arr = _ensure_batch_leading(np.asarray(p_glrlm, dtype=np.float64))
        p_list.append(p_arr[0])
        angles_ref = np.asarray(angles, dtype=np.int32)

    if not p_list:
        return np.zeros((0, Ng, nr, 0), dtype=np.float64), np.zeros((0, 3), dtype=np.int32)

    max_nr = max(item.shape[2] for item in p_list)
    padded = []
    for item in p_list:
        if item.shape[2] < max_nr:
            pad = np.zeros((item.shape[0], item.shape[1], max_nr - item.shape[2], item.shape[3]))
            item = np.concatenate([item, pad], axis=2)
        padded.append(item)
    return np.stack(padded, axis=0), angles_ref


def calculate_glszm(
    image: np.ndarray,
    sv_map: np.ndarray,
    labels: np.ndarray,
    Ng: int,
    force2D: int = 0,
    force2Ddimension: int = 0,
) -> np.ndarray:
    """
    Calculate GLSZM matrices for multiple supervoxel labels via PyRadiomics.

    Returns:
        np.ndarray: ``P_glszm`` with shape ``(n_labels, Ng, max_zone)``.
    """
    from radiomics import cMatrices

    image_i = _as_int32_image(image)
    sv_map_i = _as_int32_map(sv_map)
    labels_i = _as_int32_1d(labels)

    p_list = []
    for label in labels_i.tolist():
        mask = _label_mask(sv_map_i, int(label))
        ns = int(np.count_nonzero(mask))
        p_glszm = cMatrices.calculate_glszm(
            image_i,
            mask,
            int(Ng),
            ns,
            bool(force2D),
            int(force2Ddimension),
        )
        p_arr = _ensure_batch_leading(np.asarray(p_glszm, dtype=np.float64))
        p_list.append(p_arr[0])

    if not p_list:
        return np.zeros((0, Ng, 0), dtype=np.float64)

    max_zone = max(item.shape[2] for item in p_list)
    padded = []
    for item in p_list:
        if item.shape[2] < max_zone:
            pad = np.zeros((item.shape[0], item.shape[1], max_zone - item.shape[2]))
            item = np.concatenate([item, pad], axis=2)
        padded.append(item)
    return np.stack(padded, axis=0)


def calculate_ngtdm(
    image: np.ndarray,
    sv_map: np.ndarray,
    labels: np.ndarray,
    distances: np.ndarray,
    Ng: int,
    force2D: int = 0,
    force2Ddimension: int = 0,
) -> np.ndarray:
    """
    Calculate NGTDM matrices for multiple supervoxel labels via PyRadiomics.

    Returns:
        np.ndarray: ``P_ngtdm`` with shape ``(n_labels, Ng, 3)``.
    """
    from radiomics import cMatrices

    image_i = _as_int32_image(image)
    sv_map_i = _as_int32_map(sv_map)
    labels_i = _as_int32_1d(labels)
    distances_i = np.ascontiguousarray(np.asarray(distances, dtype=np.int32).reshape(-1))

    p_list = []
    for label in labels_i.tolist():
        mask = _label_mask(sv_map_i, int(label))
        p_ngtdm = cMatrices.calculate_ngtdm(
            image_i,
            mask,
            distances_i,
            int(Ng),
            bool(force2D),
            int(force2Ddimension),
        )
        p_arr = _ensure_batch_leading(np.asarray(p_ngtdm, dtype=np.float64))
        p_list.append(p_arr[0])

    if not p_list:
        return np.zeros((0, Ng, 3), dtype=np.float64)

    return np.stack(p_list, axis=0)


def calculate_gldm(
    image: np.ndarray,
    sv_map: np.ndarray,
    labels: np.ndarray,
    distances: np.ndarray,
    Ng: int,
    alpha: int,
    force2D: int = 0,
    force2Ddimension: int = 0,
) -> np.ndarray:
    """
    Calculate GLDM matrices for multiple supervoxel labels via PyRadiomics.

    Returns:
        np.ndarray: ``P_gldm`` with shape ``(n_labels, Ng, max_dependence)``.
    """
    from radiomics import cMatrices

    image_i = _as_int32_image(image)
    sv_map_i = _as_int32_map(sv_map)
    labels_i = _as_int32_1d(labels)
    distances_i = np.ascontiguousarray(np.asarray(distances, dtype=np.int32).reshape(-1))

    p_list = []
    for label in labels_i.tolist():
        mask = _label_mask(sv_map_i, int(label))
        p_gldm = cMatrices.calculate_gldm(
            image_i,
            mask,
            distances_i,
            int(Ng),
            int(alpha),
            bool(force2D),
            int(force2Ddimension),
        )
        p_arr = _ensure_batch_leading(np.asarray(p_gldm, dtype=np.float64))
        p_list.append(p_arr[0])

    if not p_list:
        return np.zeros((0, Ng, 0), dtype=np.float64)

    max_dep = max(item.shape[2] for item in p_list)
    padded = []
    for item in p_list:
        if item.shape[2] < max_dep:
            pad = np.zeros((item.shape[0], item.shape[1], max_dep - item.shape[2]))
            item = np.concatenate([item, pad], axis=2)
        padded.append(item)
    return np.stack(padded, axis=0)


def calculate_firstorder(
    image: np.ndarray,
    sv_map: np.ndarray,
    labels: np.ndarray,
    Ng: int,
    binWidth: float,
) -> np.ndarray:
    """
    Calculate first-order statistics for multiple labels via PyRadiomics.

    Returns:
        np.ndarray: ``stats`` with shape ``(n_labels, 17)`` in C-extension column order.
    """
    import SimpleITK as sitk
    from radiomics import firstorder

    image_f = np.ascontiguousarray(np.asarray(image, dtype=np.float64))
    sv_map_i = _as_int32_map(sv_map)
    labels_i = _as_int32_1d(labels)

    sitk_image = sitk.GetImageFromArray(image_f)

    # Column order matches ``sv_calculate_firstorder`` in sv_cmatrices.c.
    feature_names = [
        "Energy",
        "TotalEnergy",
        "Entropy",
        "Minimum",
        "10Percentile",
        "90Percentile",
        "Maximum",
        "Mean",
        "Median",
        "InterquartileRange",
        "Range",
        "MeanAbsoluteDeviation",
        "RobustMeanAbsoluteDeviation",
        "RootMeanSquared",
        "Skewness",
        "Kurtosis",
        "Uniformity",
    ]

    rows = []
    for label in labels_i.tolist():
        mask_arr = _label_mask(sv_map_i, int(label)).astype(np.uint8)
        sitk_mask = sitk.GetImageFromArray(mask_arr)
        calculator = firstorder.RadiomicsFirstOrder(
            sitk_image,
            sitk_mask,
            binWidth=float(binWidth),
        )
        calculator._initCalculation(None)
        row = []
        for name in feature_names:
            method = getattr(calculator, f"get{name}FeatureValue")
            value = np.asarray(method(), dtype=np.float64).reshape(-1)
            row.append(float(value[0]) if value.size else float("nan"))
        rows.append(row)

    if not rows:
        return np.zeros((0, len(feature_names)), dtype=np.float64)

    return np.asarray(rows, dtype=np.float64)
