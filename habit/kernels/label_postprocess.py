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
"""Connected-component cleanup for integer label maps (L0).

Removes tiny per-label islands inside an ROI and reassigns those voxels to the
nearest surviving seed label. Callers decide whether cleanup is enabled; this
module always runs the numerical path when invoked.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

__all__ = ["remove_small_connected_components"]


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
        connectivity: Connectivity level in ``[1, 2, 3]``.

    Returns:
        3D boolean mask with only large components preserved.
    """
    import SimpleITK as sitk

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
        3D float32 distance map (0 on seed voxels).
    """
    import SimpleITK as sitk

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
) -> np.ndarray:
    """
    Fast ROI-safe cleanup:

    1. Temporarily remove tiny components per label.
    2. Refill removed ROI voxels by nearest large-component seed label.

    This keeps every ROI voxel labeled (>0) whenever at least one large seed
    component survives filtering.

    Args:
        label_map: 3D integer label map where 0 means background.
        roi_mask: 3D boolean mask indicating valid ROI.
        min_component_size: Minimum voxels for each connected component.
        connectivity: Connectivity level in ``[1, 2, 3]``.

    Returns:
        3D integer label map after cleanup and refill.
    """
    cleaned_seed = np.zeros_like(label_map, dtype=np.int32)
    labels = np.unique(label_map[roi_mask])
    labels = labels[labels > 0]

    if labels.size == 0:
        return cleaned_seed

    for label_id in labels:
        class_mask = (label_map == label_id) & roi_mask
        if not np.any(class_mask):
            continue
        kept_mask = _connected_component_keep_large_sitk(
            binary_mask=class_mask,
            min_component_size=min_component_size,
            connectivity=connectivity,
        )
        cleaned_seed[kept_mask & roi_mask] = int(label_id)

    removed_mask = roi_mask & (cleaned_seed == 0)
    if not np.any(removed_mask):
        cleaned_seed[~roi_mask] = 0
        return cleaned_seed

    seed_labels = np.unique(cleaned_seed[roi_mask])
    seed_labels = seed_labels[seed_labels > 0]
    if seed_labels.size == 0:
        # Extreme case: all components are below threshold.
        # Return original labels to avoid producing unlabeled ROI voxels.
        fallback = label_map.astype(np.int32, copy=True)
        fallback[~roi_mask] = 0
        return fallback

    best_distance = np.full(label_map.shape, np.inf, dtype=np.float32)
    best_label = np.zeros(label_map.shape, dtype=np.int32)

    for label_id in seed_labels:
        seed_mask = (cleaned_seed == label_id) & roi_mask
        if not np.any(seed_mask):
            continue
        distance_map = _distance_to_seed_sitk(seed_mask=seed_mask)
        update_mask = removed_mask & (distance_map < best_distance)
        best_distance[update_mask] = distance_map[update_mask]
        best_label[update_mask] = int(label_id)

    output = cleaned_seed.copy()
    output[removed_mask] = best_label[removed_mask]

    unresolved = roi_mask & (output == 0)
    if np.any(unresolved):
        output[unresolved] = label_map[unresolved]

    output[~roi_mask] = 0
    return output


def remove_small_connected_components(
    label_map: np.ndarray,
    roi_mask: np.ndarray,
    *,
    min_component_size: int = 30,
    connectivity: int = 1,
    settings: Optional[Mapping[str, Any]] = None,
) -> np.ndarray:
    """
    Remove tiny connected components by label-wise reassignment in ROI.

    Args:
        label_map: 3D integer label map where 0 means background.
        roi_mask: 3D boolean mask indicating valid ROI.
        min_component_size: Minimum voxels required to keep a component.
        connectivity: Neighborhood connectivity in ``{1, 2, 3}``.
        settings: Optional legacy mapping. When provided, ``min_component_size``
            and ``connectivity`` may be overridden from it. The ``enabled``
            key is ignored here; callers that honour ``enabled`` must gate
            the call themselves (or use the utils facade).

    Returns:
        Cleaned label map with reduced tiny fragments.
    """
    size = int(min_component_size)
    conn = int(connectivity)
    if settings is not None:
        size = int(max(1, settings.get("min_component_size", size)))
        conn = int(settings.get("connectivity", conn))
    else:
        size = int(max(1, size))

    cleaned = np.asarray(label_map, dtype=np.int32)
    mask = np.asarray(roi_mask, dtype=bool)
    return _remove_and_refill_by_nearest_seed(
        label_map=cleaned,
        roi_mask=mask,
        min_component_size=size,
        connectivity=conn,
    )
