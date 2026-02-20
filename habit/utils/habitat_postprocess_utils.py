"""
Post-processing utilities for habitat/supervoxel label maps.

This module removes tiny connected components inside ROI and reassigns them to
nearby dominant labels to reduce fragmented patches.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import logging
import time
import numpy as np
import SimpleITK as sitk


logger = logging.getLogger(__name__)


def _connected_component_keep_large_sitk(
    binary_mask: np.ndarray,
    min_component_size: int,
    connectivity: int,
) -> np.ndarray:
    """
    Keep only large connected components in a binary mask with SimpleITK.

    Args:
        binary_mask: 3D binary array where True means candidate voxels.
        min_component_size: Minimum voxels required to keep a component.
        connectivity: Connectivity level in [1, 2, 3].

    Returns:
        np.ndarray: 3D boolean mask with only large components preserved.
    """
    # SimpleITK uses a boolean "fullyConnected" switch:
    # False ~= 6-neighborhood, True ~= full neighborhood (edge+corner connected).
    # We map connectivity=1 to False, and 2/3 to True for a practical 3D behavior.
    fully_connected = int(np.clip(connectivity, 1, 3)) > 1

    image = sitk.GetImageFromArray(binary_mask.astype(np.uint8))
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(bool(fully_connected))
    cc = cc_filter.Execute(image)

    relabel_filter = sitk.RelabelComponentImageFilter()
    relabel_filter.SetMinimumObjectSize(int(min_component_size))
    relabeled = relabel_filter.Execute(cc)
    return sitk.GetArrayFromImage(relabeled) > 0


def _distance_to_seed_sitk(seed_mask: np.ndarray) -> np.ndarray:
    """
    Compute distance-to-seed map using SimpleITK signed Maurer distance.

    Args:
        seed_mask: 3D binary seed mask where True marks seed voxels.

    Returns:
        np.ndarray: 3D float32 distance map (0 on seed voxels).
    """
    seed_img = sitk.GetImageFromArray(seed_mask.astype(np.uint8))
    dist_img = sitk.SignedMaurerDistanceMap(
        seed_img,
        insideIsPositive=False,
        squaredDistance=False,
        useImageSpacing=False,
    )
    dist_arr = sitk.GetArrayFromImage(sitk.Abs(dist_img))
    return dist_arr.astype(np.float32, copy=False)


def _remove_and_refill_by_nearest_seed(
    label_map: np.ndarray,
    roi_mask: np.ndarray,
    min_component_size: int,
    connectivity: int,
    debug_postprocess: bool = False,
) -> np.ndarray:
    """
    Fast ROI-safe cleanup:
    1) Temporarily remove tiny components per label.
    2) Refill removed ROI voxels by nearest large-component seed label.

    This guarantees all ROI voxels remain labeled (>0) as long as at least one
    large seed component exists after filtering.

    Args:
        label_map: 3D integer label map where 0 means background.
        roi_mask: 3D boolean mask indicating valid ROI.
        min_component_size: Minimum voxels for each connected component.
        connectivity: Connectivity level in [1, 2, 3].
        debug_postprocess: Whether to emit verbose per-label logs.

    Returns:
        np.ndarray: 3D integer label map after cleanup and refill.
    """
    start_time_s = time.perf_counter()
    roi_voxels = int(np.count_nonzero(roi_mask))
    cleaned_seed = np.zeros_like(label_map, dtype=np.int32)
    labels = np.unique(label_map[roi_mask])
    labels = labels[labels > 0]

    logger.info(
        "Postprocess start: roi_voxels=%d, labels=%d, min_component_size=%d, connectivity=%d",
        roi_voxels,
        int(labels.size),
        int(min_component_size),
        int(connectivity),
    )

    if labels.size == 0:
        logger.info("Postprocess skipped: no positive labels inside ROI.")
        return cleaned_seed

    # Step A: keep only large components for each class as trusted seeds.
    step_a_start_s = time.perf_counter()
    for label_id in labels:
        class_mask = (label_map == label_id) & roi_mask
        if not np.any(class_mask):
            continue
        kept_mask = _connected_component_keep_large_sitk(
            binary_mask=class_mask,
            min_component_size=min_component_size,
            connectivity=connectivity,
        )
        if debug_postprocess:
            class_voxels = int(np.count_nonzero(class_mask))
            kept_voxels = int(np.count_nonzero(kept_mask & roi_mask))
            removed_voxels = class_voxels - kept_voxels
            logger.info(
                "Postprocess label=%d: class_voxels=%d, kept_voxels=%d, removed_voxels=%d",
                int(label_id),
                class_voxels,
                kept_voxels,
                removed_voxels,
            )
        cleaned_seed[kept_mask & roi_mask] = int(label_id)

    removed_mask = roi_mask & (cleaned_seed == 0)
    kept_total = int(np.count_nonzero(cleaned_seed > 0))
    removed_total = int(np.count_nonzero(removed_mask))
    logger.info(
        "Postprocess seed filtering done: kept_voxels=%d, removed_voxels=%d, elapsed=%.2fs",
        kept_total,
        removed_total,
        time.perf_counter() - step_a_start_s,
    )

    if not np.any(removed_mask):
        cleaned_seed[~roi_mask] = 0
        logger.info(
            "Postprocess finished: total_elapsed=%.2fs, roi_zero_after=0",
            time.perf_counter() - start_time_s,
        )
        return cleaned_seed

    seed_labels = np.unique(cleaned_seed[roi_mask])
    seed_labels = seed_labels[seed_labels > 0]
    if seed_labels.size == 0:
        # Extreme case: all components are below threshold.
        # Return original labels to avoid producing unlabeled ROI voxels.
        logger.warning(
            "Postprocess fallback: all components were removed by size filtering; "
            "returning original labels for this subject."
        )
        fallback = label_map.astype(np.int32, copy=True)
        fallback[~roi_mask] = 0
        fallback_roi_zero = int(np.count_nonzero((fallback == 0) & roi_mask))
        logger.info(
            "Postprocess finished (fallback): total_elapsed=%.2fs, roi_zero_after=%d",
            time.perf_counter() - start_time_s,
            fallback_roi_zero,
        )
        return fallback

    # Step B: refill removed voxels by nearest seed label.
    step_b_start_s = time.perf_counter()
    best_distance = np.full(label_map.shape, np.inf, dtype=np.float32)
    best_label = np.zeros(label_map.shape, dtype=np.int32)

    for label_id in seed_labels:
        seed_mask = (cleaned_seed == label_id) & roi_mask
        if not np.any(seed_mask):
            continue
        distance_map = _distance_to_seed_sitk(seed_mask=seed_mask)
        update_mask = removed_mask & (distance_map < best_distance)
        if debug_postprocess:
            update_voxels = int(np.count_nonzero(update_mask))
            logger.info(
                "Postprocess refill candidate label=%d: updated_voxels=%d",
                int(label_id),
                update_voxels,
            )
        best_distance[update_mask] = distance_map[update_mask]
        best_label[update_mask] = int(label_id)

    output = cleaned_seed.copy()
    output[removed_mask] = best_label[removed_mask]

    unresolved_before_guard = int(np.count_nonzero(roi_mask & (output == 0)))
    logger.info(
        "Postprocess distance refill done: unresolved_before_guard=%d, elapsed=%.2fs",
        unresolved_before_guard,
        time.perf_counter() - step_b_start_s,
    )

    # Final ROI consistency guard: ROI voxels must remain labeled.
    unresolved = roi_mask & (output == 0)
    if np.any(unresolved):
        logger.warning(
            "Postprocess found unresolved ROI voxels (%d). Restoring original labels "
            "for these voxels.",
            int(np.count_nonzero(unresolved)),
        )
        output[unresolved] = label_map[unresolved]

    output[~roi_mask] = 0
    roi_zero_after = int(np.count_nonzero((output == 0) & roi_mask))
    logger.info(
        "Postprocess finished: total_elapsed=%.2fs, roi_zero_after=%d",
        time.perf_counter() - start_time_s,
        roi_zero_after,
    )
    return output


def remove_small_connected_components(
    label_map: np.ndarray,
    roi_mask: np.ndarray,
    settings: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Remove tiny connected components by label-wise reassignment in ROI.

    Args:
        label_map: 3D integer label map where 0 means background.
        roi_mask: 3D boolean mask indicating valid ROI.
        settings: Post-process settings dictionary with keys:
            - enabled (bool)
            - min_component_size (int)
            - connectivity (int)

    Returns:
        np.ndarray: Cleaned label map with reduced tiny fragments.
    """
    if settings is None or not bool(settings.get("enabled", False)):
        return label_map

    cleaned = label_map.astype(np.int32, copy=True)
    min_component_size = int(max(1, settings.get("min_component_size", 30)))
    connectivity = int(settings.get("connectivity", 1))
    debug_postprocess = bool(settings.get("debug_postprocess", False))
    # Single fast path: SimpleITK-based remove+refill.
    # This preserves all ROI voxels while reducing fragmented tiny islands.
    return _remove_and_refill_by_nearest_seed(
        label_map=cleaned,
        roi_mask=roi_mask,
        min_component_size=min_component_size,
        connectivity=connectivity,
        debug_postprocess=debug_postprocess,
    )

