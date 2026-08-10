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
"""Connected-region node extraction for habitat graph features."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage as ndi

from habit.kernels.habitat_graph.models import (
    HabitatGraphNode,
    HabitatNodeExtractionResult,
)

__all__ = ["extract_habitat_nodes"]


def _connectivity_structure(ndim: int, connectivity: str) -> np.ndarray:
    """
    Build an n-dimensional binary structure for connected-component labeling.

    Args:
        ndim: Number of array dimensions.
        connectivity: ``"face"`` for 4-neighborhood in 2D / 6-neighborhood in
            3D, or ``"full"`` for diagonal-inclusive connectivity.

    Returns:
        np.ndarray: Binary structure accepted by ``scipy.ndimage.label``.
    """
    if connectivity == "face":
        return ndi.generate_binary_structure(rank=ndim, connectivity=1)
    if connectivity == "full":
        return ndi.generate_binary_structure(rank=ndim, connectivity=ndim)
    raise ValueError("connectivity must be 'face' or 'full'.")


def _component_bbox(coords: np.ndarray) -> Tuple[int, ...]:
    """
    Return a half-open bounding box for component coordinates.

    Args:
        coords: Array of component voxel coordinates with shape ``(n, ndim)``.

    Returns:
        Tuple[int, ...]: ``(min_dim0, ..., min_dimN, max_dim0, ..., max_dimN)``.
    """
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    return tuple(int(v) for v in np.concatenate([mins, maxs]))


def _subdivide_component(
    coords: np.ndarray,
    block_size: int,
    ndim: int,
    min_coverage: float,
) -> List[np.ndarray]:
    """
    Split a large connected component into fixed-size grid blocks.

    Each voxel is assigned to an n-dimensional block of edge length
    ``block_size``. A block is kept only when the fraction of its volume covered
    by the component exceeds ``min_coverage``, which mirrors the source PathPrism
    behavior of dropping sparsely covered boundary blocks. Splitting prevents a
    single large region from collapsing into one graph node, which would erase
    the internal spatial structure that graph features are meant to capture.

    Args:
        coords: Component voxel coordinates with shape ``(n, ndim)``.
        block_size: Edge length of each grid block in voxels.
        ndim: Number of array dimensions.
        min_coverage: Minimum covered fraction of a block volume to keep it.

    Returns:
        List[np.ndarray]: One coordinate array per kept block. Empty when no
        block reaches the coverage threshold.
    """
    mins = coords.min(axis=0)
    # Integer block index per voxel along every dimension.
    block_indices = (coords - mins) // block_size
    unique_blocks, inverse = np.unique(block_indices, axis=0, return_inverse=True)
    block_volume = float(block_size**ndim)

    kept_blocks: List[np.ndarray] = []
    for block_id in range(unique_blocks.shape[0]):
        block_coords = coords[inverse == block_id]
        coverage = block_coords.shape[0] / block_volume
        if coverage > min_coverage:
            kept_blocks.append(block_coords)
    return kept_blocks


def extract_habitat_nodes(
    label_array: np.ndarray,
    connectivity: str = "face",
    min_region_voxels: int = 1,
    erosion_radius: int = 0,
    subdivide_region_voxels: int = 0,
    block_size: int = 30,
    block_min_coverage: float = 0.5,
) -> HabitatNodeExtractionResult:
    """
    Convert each connected habitat region into one or more graph nodes.

    Args:
        label_array: Integer habitat label map. Background must be encoded as 0.
        connectivity: Connected-component neighborhood rule.
        min_region_voxels: Components smaller than this voxel count are ignored.
        erosion_radius: Optional binary erosion iterations applied per habitat
            before connected-component labeling. The default is 0 because
            medical image habitats may contain small but meaningful regions.
        subdivide_region_voxels: Connected components whose voxel count exceeds
            this value are split into fixed-size grid blocks, one node per block.
            The default is 0, which disables splitting. Use a positive value when
            a single large habitat blob would otherwise dominate graph features.
        block_size: Edge length of each grid block in voxels when splitting.
        block_min_coverage: Minimum covered fraction of a block volume required
            to keep that block as a node when splitting.

    Returns:
        HabitatNodeExtractionResult: Nodes grouped by habitat label plus
        component maps used by contact-based edge builders.
    """
    if label_array.ndim not in (2, 3):
        raise ValueError("label_array must be 2D or 3D.")
    if min_region_voxels < 1:
        raise ValueError("min_region_voxels must be >= 1.")
    if erosion_radius < 0:
        raise ValueError("erosion_radius must be >= 0.")
    if subdivide_region_voxels < 0:
        raise ValueError("subdivide_region_voxels must be >= 0.")
    if block_size < 1:
        raise ValueError("block_size must be >= 1.")
    if not 0.0 <= block_min_coverage <= 1.0:
        raise ValueError("block_min_coverage must be in [0, 1].")

    labels = [int(v) for v in np.unique(label_array) if int(v) > 0]
    structure = _connectivity_structure(label_array.ndim, connectivity)
    nodes_by_habitat: Dict[int, List[HabitatGraphNode]] = {}
    component_maps: Dict[int, np.ndarray] = {}

    for habitat_label in labels:
        mask = label_array == habitat_label
        if erosion_radius > 0:
            mask = ndi.binary_erosion(
                mask,
                structure=structure,
                iterations=erosion_radius,
                border_value=0,
            )

        labeled_components, component_count = ndi.label(mask, structure=structure)
        kept_component_map = np.zeros_like(labeled_components, dtype=np.int32)
        habitat_nodes: List[HabitatGraphNode] = []
        # Block nodes need unique component ids that never collide with the
        # original connected-component ids painted into the component map.
        next_block_component_id = int(component_count) + 1

        for component_id in range(1, int(component_count) + 1):
            coords = np.argwhere(labeled_components == component_id)
            voxel_count = int(coords.shape[0])
            if voxel_count < min_region_voxels:
                continue

            should_subdivide = (
                subdivide_region_voxels > 0
                and voxel_count > subdivide_region_voxels
            )
            block_groups: List[np.ndarray] = []
            if should_subdivide:
                block_groups = _subdivide_component(
                    coords=coords,
                    block_size=block_size,
                    ndim=label_array.ndim,
                    min_coverage=block_min_coverage,
                )

            if block_groups:
                # One graph node per kept block; each block gets a unique id.
                for block_coords in block_groups:
                    block_component_id = next_block_component_id
                    next_block_component_id += 1
                    kept_component_map[tuple(block_coords.T)] = block_component_id
                    habitat_nodes.append(
                        HabitatGraphNode(
                            node_id=f"h{habitat_label}_c{block_component_id}",
                            habitat_label=habitat_label,
                            component_id=block_component_id,
                            centroid=block_coords.mean(axis=0).astype(float),
                            voxel_count=int(block_coords.shape[0]),
                            bbox=_component_bbox(block_coords),
                        )
                    )
            else:
                # Keep the whole component as a single node (default behavior).
                kept_component_map[tuple(coords.T)] = component_id
                habitat_nodes.append(
                    HabitatGraphNode(
                        node_id=f"h{habitat_label}_c{component_id}",
                        habitat_label=habitat_label,
                        component_id=component_id,
                        centroid=coords.mean(axis=0).astype(float),
                        voxel_count=voxel_count,
                        bbox=_component_bbox(coords),
                    )
                )

        nodes_by_habitat[habitat_label] = habitat_nodes
        component_maps[habitat_label] = kept_component_map

    return HabitatNodeExtractionResult(
        label_array=label_array,
        nodes_by_habitat=nodes_by_habitat,
        component_maps=component_maps,
    )
