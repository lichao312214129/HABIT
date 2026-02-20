"""
Post-processing utilities for habitat/supervoxel label maps.

This module removes tiny connected components inside ROI and reassigns them to
nearby dominant labels to reduce fragmented patches.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy import ndimage


def _get_connectivity_structure(connectivity: int) -> np.ndarray:
    """
    Build 3D neighborhood structure for connected-component analysis.

    Args:
        connectivity: Connectivity level in [1, 2, 3] for 6/18/26-neighborhood.

    Returns:
        np.ndarray: Boolean structure kernel for ndimage labeling.
    """
    conn = int(np.clip(connectivity, 1, 3))
    return ndimage.generate_binary_structure(rank=3, connectivity=conn)


def _neighbor_vote_label(
    label_map: np.ndarray,
    component_mask: np.ndarray,
    roi_mask: np.ndarray,
    structure: np.ndarray,
) -> int:
    """
    Select target label for a tiny component using neighbor majority vote.

    Args:
        label_map: Integer label map (0 for background).
        component_mask: Boolean mask of the tiny component.
        roi_mask: Boolean ROI mask.
        structure: Connectivity structure.

    Returns:
        int: Target label ID (>0). Returns 0 if no valid neighbor exists.
    """
    dilated = ndimage.binary_dilation(component_mask, structure=structure)
    boundary = dilated & (~component_mask) & roi_mask
    candidate_labels = label_map[boundary]
    candidate_labels = candidate_labels[candidate_labels > 0]
    if candidate_labels.size == 0:
        return 0
    labels, counts = np.unique(candidate_labels, return_counts=True)
    return int(labels[np.argmax(counts)])


def remove_small_connected_components(
    label_map: np.ndarray,
    roi_mask: np.ndarray,
    settings: Optional[Dict] = None,
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
            - reassign_method (str, currently supports "neighbor_vote")
            - max_iterations (int)

    Returns:
        np.ndarray: Cleaned label map with reduced tiny fragments.
    """
    if settings is None or not bool(settings.get("enabled", False)):
        return label_map

    cleaned = label_map.astype(np.int32, copy=True)
    min_component_size = int(max(1, settings.get("min_component_size", 30)))
    connectivity = int(settings.get("connectivity", 1))
    max_iterations = int(max(1, settings.get("max_iterations", 3)))
    reassign_method = str(settings.get("reassign_method", "neighbor_vote")).lower()
    structure = _get_connectivity_structure(connectivity)

    for _ in range(max_iterations):
        changed = False
        labels = np.unique(cleaned[roi_mask])
        labels = labels[labels > 0]
        if labels.size == 0:
            break

        for label_id in labels:
            class_mask = (cleaned == label_id) & roi_mask
            cc_map, cc_count = ndimage.label(class_mask, structure=structure)
            if cc_count <= 1:
                continue

            sizes = np.bincount(cc_map.ravel())
            for cc_id in range(1, cc_count + 1):
                comp_size = int(sizes[cc_id]) if cc_id < sizes.size else 0
                if comp_size >= min_component_size:
                    continue

                comp_mask = cc_map == cc_id
                if reassign_method == "neighbor_vote":
                    target_label = _neighbor_vote_label(
                        cleaned, comp_mask, roi_mask, structure
                    )
                else:
                    target_label = _neighbor_vote_label(
                        cleaned, comp_mask, roi_mask, structure
                    )

                if target_label > 0 and target_label != label_id:
                    cleaned[comp_mask] = target_label
                    changed = True

        if not changed:
            break

    # Ensure non-ROI stays background.
    cleaned[~roi_mask] = 0
    return cleaned

