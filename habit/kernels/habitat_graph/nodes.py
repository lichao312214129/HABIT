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

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi

from habit.kernels.habitat_graph.models import (
    BACKGROUND_SHELL_LABEL,
    HabitatGraphNode,
    HabitatNodeExtractionResult,
    NodeMethod,
)

__all__ = ["BACKGROUND_SHELL_LABEL", "extract_habitat_nodes"]


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
    origin: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """
    Split a connected component into fixed-size grid blocks.

    Each voxel is assigned to an n-dimensional block of edge length
    ``block_size``. A block is kept only when the fraction of its volume covered
    by the component exceeds ``min_coverage``, which mirrors the source PathPrism
    behavior of dropping sparsely covered boundary blocks.

    Args:
        coords: Component voxel coordinates with shape ``(n, ndim)``.
        block_size: Edge length of each grid block in voxels.
        ndim: Number of array dimensions.
        min_coverage: Minimum covered fraction of a block volume to keep it.
        origin: Lattice origin in voxel indices. ``None`` uses this
            component's own bounding-box minimum (legacy per-component
            grid). Pass the tumour-VOI minimum for a global lattice.

    Returns:
        List[np.ndarray]: One coordinate array per kept block. Empty when no
        block reaches the coverage threshold.
    """
    mins = coords.min(axis=0) if origin is None else np.asarray(origin)
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


def _voi_grid_origin(label_array: np.ndarray) -> Optional[np.ndarray]:
    """
    Return the inclusive voxel-index origin of the non-background VOI.

    Args:
        label_array: Integer habitat label map (background encoded as 0).

    Returns:
        np.ndarray | None: One integer per axis, or ``None`` when empty.
    """
    coords = np.argwhere(label_array > 0)
    if coords.size == 0:
        return None
    return coords.min(axis=0).astype(int, copy=False)


def _eroded_label_array(
    label_array: np.ndarray,
    labels: List[int],
    structure: np.ndarray,
    erosion_radius: int,
) -> np.ndarray:
    """
    Optionally erode each habitat and rebuild the integer label map.

    Args:
        label_array: Integer habitat label map.
        labels: Positive habitat ids to process.
        structure: Binary structure for ``scipy.ndimage.binary_erosion``.
        erosion_radius: Erosion iterations; ``0`` returns ``label_array``.

    Returns:
        np.ndarray: Label map after per-habitat erosion (copy if eroded).
    """
    if erosion_radius <= 0:
        return label_array
    eroded = np.zeros_like(label_array)
    for habitat_label in labels:
        mask = ndi.binary_erosion(
            label_array == habitat_label,
            structure=structure,
            iterations=erosion_radius,
            border_value=0,
        )
        eroded[mask] = habitat_label
    return eroded


def _node_from_coords(
    habitat_label: int,
    component_id: int,
    coords: np.ndarray,
) -> HabitatGraphNode:
    """Build one graph node from a voxel-coordinate set."""
    return HabitatGraphNode(
        node_id=f"h{habitat_label}_c{component_id}",
        habitat_label=habitat_label,
        component_id=component_id,
        centroid=coords.mean(axis=0).astype(float),
        voxel_count=int(coords.shape[0]),
        bbox=_component_bbox(coords),
    )


def _background_shell_mask(
    label_array: np.ndarray,
    width: int,
    connectivity: str,
) -> np.ndarray:
    """
    Return the peritumoral shell (dilated VOI minus the original VOI).

    Dilation uses the same neighbourhood as graph ``connectivity``
    (``face`` or ``full``) and stays inside the array bounds.

    Args:
        label_array: Integer habitat map (background encoded as 0).
        width: Dilation iterations (>= 1). One iteration is a 1-voxel ring.
        connectivity: ``"face"`` or ``"full"``.

    Returns:
        np.ndarray: Boolean mask of shell voxels. Empty when the VOI is empty.
    """
    if width < 1:
        raise ValueError("background_shell_width must be >= 1.")
    structure = _connectivity_structure(label_array.ndim, connectivity)
    voi = label_array > 0
    if not np.any(voi):
        return np.zeros(label_array.shape, dtype=bool)
    dilated = ndi.binary_dilation(
        voi, structure=structure, iterations=int(width)
    )
    return dilated & ~voi


def _remap_nodes_to_background(
    nodes: List[HabitatGraphNode],
) -> List[HabitatGraphNode]:
    """Rewrite painted shell nodes to the reserved background class."""
    remapped: List[HabitatGraphNode] = []
    for node in nodes:
        remapped.append(
            HabitatGraphNode(
                node_id=f"bg_c{int(node.component_id)}",
                habitat_label=BACKGROUND_SHELL_LABEL,
                component_id=int(node.component_id),
                centroid=node.centroid,
                voxel_count=int(node.voxel_count),
                bbox=node.bbox,
            )
        )
    return remapped


def _extract_uniform_grid_nodes(
    label_array: np.ndarray,
    labels: List[int],
    structure: np.ndarray,
    min_region_voxels: int,
    erosion_radius: int,
    block_size: int,
    block_min_coverage: float,
    grid_origin: Optional[np.ndarray] = None,
) -> HabitatNodeExtractionResult:
    """
    Tessellate the tumour VOI and emit one node per cell subregion.

    Every non-background voxel is assigned to an axis-aligned cube of edge
    ``block_size`` whose origin is the VOI bounding-box minimum. A cube is
    kept when its occupied fraction is strictly greater than
    ``block_min_coverage`` (cell-level filter for nearly-empty cubes).
    Inside each kept cube, every connected component of every habitat
    becomes its own node at that subregion's voxel-index centroid, so a
    mixed cube can contribute several nodes. Subregions smaller than
    ``min_region_voxels`` are dropped (fragment filter). Habitats that
    occupy no kept subregion still emit one residual node so a present
    label is never silently dropped.

    Args:
        label_array: Integer habitat label map (background encoded as 0).
        labels: Positive habitat ids present before erosion.
        structure: Neighbourhood for optional erosion and in-cell
            connected-component labeling.
        min_region_voxels: Drop in-cell subregions (and residual nodes)
            smaller than this voxel count.
        erosion_radius: Optional per-habitat erosion iterations.
        block_size: Cube edge length in voxels.
        block_min_coverage: Minimum occupied fraction of a cube to keep
            the cell (strictly greater than this value).
        grid_origin: Optional lattice origin in voxel indices. ``None``
            uses the non-background bounding-box minimum of ``working``.
            Pass the tumour-VOI origin when extracting the background
            shell so habitat cubes stay on the same lattice.

    Returns:
        HabitatNodeExtractionResult: Nodes, component maps, and lattice.
    """
    working = _eroded_label_array(label_array, labels, structure, erosion_radius)
    origin = (
        np.asarray(grid_origin, dtype=int)
        if grid_origin is not None
        else _voi_grid_origin(working)
    )
    nodes_by_habitat: Dict[int, List[HabitatGraphNode]] = {}
    component_maps: Dict[int, np.ndarray] = {
        int(label): np.zeros(working.shape, dtype=np.int32) for label in labels
    }
    if origin is None:
        return HabitatNodeExtractionResult(
            label_array=label_array,
            nodes_by_habitat=nodes_by_habitat,
            component_maps=component_maps,
            grid_origin=None,
            grid_block_size=int(block_size),
        )

    coords = np.argwhere(working > 0)
    voxel_labels = working[tuple(coords.T)]
    block_indices = (coords - origin) // block_size
    unique_blocks, inverse = np.unique(block_indices, axis=0, return_inverse=True)
    block_volume = float(block_size ** working.ndim)
    cube_shape = tuple(int(block_size) for _ in range(working.ndim))
    next_component_id = 1
    kept_by_habitat: Dict[int, List[HabitatGraphNode]] = {
        int(label): [] for label in labels
    }

    for block_id in range(unique_blocks.shape[0]):
        in_block = inverse == block_id
        block_coords = coords[in_block]
        block_lab = voxel_labels[in_block]
        coverage = block_coords.shape[0] / block_volume
        if coverage <= block_min_coverage:
            continue
        # Local cube so in-cell connected components stay spatially correct.
        local = block_coords - origin - unique_blocks[block_id] * block_size
        cube = np.zeros(cube_shape, dtype=np.int32)
        cube[tuple(local.T)] = block_lab
        for habitat_label in (int(v) for v in np.unique(block_lab) if int(v) > 0):
            labeled, n_cc = ndi.label(cube == habitat_label, structure=structure)
            for cc_id in range(1, int(n_cc) + 1):
                cc_local = np.argwhere(labeled == cc_id)
                if cc_local.shape[0] < min_region_voxels:
                    continue
                global_coords = (
                    cc_local + origin + unique_blocks[block_id] * block_size
                )
                component_id = next_component_id
                next_component_id += 1
                component_maps[habitat_label][tuple(global_coords.T)] = component_id
                kept_by_habitat[habitat_label].append(
                    _node_from_coords(habitat_label, component_id, global_coords)
                )

    present_after = {int(v) for v in np.unique(working) if int(v) > 0}
    for habitat_label in labels:
        habitat_nodes = kept_by_habitat.get(habitat_label, [])
        if habitat_nodes:
            nodes_by_habitat[habitat_label] = habitat_nodes
            continue
        if habitat_label not in present_after:
            continue
        leftover = np.argwhere(working == habitat_label)
        if leftover.shape[0] < min_region_voxels:
            continue
        # Residual node: the habitat is present but no subregion was kept.
        component_id = next_component_id
        next_component_id += 1
        component_maps[habitat_label][tuple(leftover.T)] = component_id
        nodes_by_habitat[habitat_label] = [
            _node_from_coords(habitat_label, component_id, leftover)
        ]

    return HabitatNodeExtractionResult(
        label_array=label_array,
        nodes_by_habitat=nodes_by_habitat,
        component_maps=component_maps,
        grid_origin=tuple(int(v) for v in origin),
        grid_block_size=int(block_size),
    )


def _extract_component_nodes(
    label_array: np.ndarray,
    labels: List[int],
    structure: np.ndarray,
    min_region_voxels: int,
    erosion_radius: int,
    subdivide_region_voxels: int,
    block_size: int,
    block_min_coverage: float,
) -> HabitatNodeExtractionResult:
    """Connected-component nodes (optional size split). Habitat labels only."""
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


def _merge_background_shell_nodes(
    habitat_result: HabitatNodeExtractionResult,
    *,
    label_array: np.ndarray,
    connectivity: str,
    min_region_voxels: int,
    block_size: int,
    block_min_coverage: float,
    node_method: NodeMethod,
    background_shell_width: int,
) -> HabitatNodeExtractionResult:
    """
    Append reserved-class shell nodes without moving the habitat lattice.

    Habitat nodes stay exactly as extracted from the original VOI. The
    shell is a second extract on ``dilated VOI AND NOT VOI``, using the
    same ``block_size`` / coverage (and the same grid origin in
    ``uniform_grid`` mode).
    """
    shell = _background_shell_mask(
        label_array, background_shell_width, connectivity
    )
    paint_id = 1
    shell_labels = np.zeros(label_array.shape, dtype=np.int32)
    shell_labels[shell] = paint_id
    structure = _connectivity_structure(label_array.ndim, connectivity)
    if node_method == "uniform_grid":
        origin = habitat_result.grid_origin
        origin_arr = (
            np.asarray(origin, dtype=int) if origin is not None else None
        )
        shell_result = _extract_uniform_grid_nodes(
            label_array=shell_labels,
            labels=[paint_id],
            structure=structure,
            min_region_voxels=min_region_voxels,
            erosion_radius=0,
            block_size=block_size,
            block_min_coverage=block_min_coverage,
            grid_origin=origin_arr,
        )
    else:
        shell_result = _extract_component_nodes(
            label_array=shell_labels,
            labels=[paint_id],
            structure=structure,
            min_region_voxels=min_region_voxels,
            erosion_radius=0,
            subdivide_region_voxels=0,
            block_size=block_size,
            block_min_coverage=block_min_coverage,
        )
    bg_nodes = _remap_nodes_to_background(
        list(shell_result.nodes_by_habitat.get(paint_id, []))
    )
    bg_map = shell_result.component_maps.get(
        paint_id, np.zeros(label_array.shape, dtype=np.int32)
    )
    nodes_by_habitat = dict(habitat_result.nodes_by_habitat)
    nodes_by_habitat[BACKGROUND_SHELL_LABEL] = bg_nodes
    component_maps = dict(habitat_result.component_maps)
    component_maps[BACKGROUND_SHELL_LABEL] = bg_map
    return HabitatNodeExtractionResult(
        label_array=habitat_result.label_array,
        nodes_by_habitat=nodes_by_habitat,
        component_maps=component_maps,
        grid_origin=habitat_result.grid_origin,
        grid_block_size=habitat_result.grid_block_size,
    )


def extract_habitat_nodes(
    label_array: np.ndarray,
    connectivity: str = "full",
    min_region_voxels: int = 1,
    erosion_radius: int = 0,
    subdivide_region_voxels: int = 0,
    block_size: int = 8,
    block_min_coverage: float = 0.2,
    node_method: NodeMethod = "uniform_grid",
    include_background_shell: bool = True,
    background_shell_width: int = 1,
) -> HabitatNodeExtractionResult:
    """
    Convert a habitat label map into graph nodes.

    Default ``node_method='uniform_grid'`` paints a global axis-aligned
    lattice of cubes with edge ``block_size`` (default 8 **voxels**, not
    millimetres) over the tumour VOI. Cubes whose occupied fraction
    exceeds ``block_min_coverage`` (default 0.2) are kept; **each
    connected subregion inside a kept cube becomes its own node** at
    that subregion's voxel-index centroid (several habitats and/or
    several components of one habitat can share a cube). Pass
    ``node_method='component'`` for the older connected-component nodes,
    optionally split when a component exceeds
    ``subdivide_region_voxels``.

    Args:
        label_array: Integer habitat label map. Background must be encoded as 0.
        connectivity: Connected-component neighborhood rule. Default
            ``"full"`` is 8-connected in 2D / 26-connected in 3D. Pass
            ``"face"`` for 4-connected / 6-connected neighborhoods. Used by
            ``component`` mode and by optional erosion in both modes.
        min_region_voxels: Components / residual nodes smaller than this
            voxel count are ignored.
        erosion_radius: Optional binary erosion iterations applied per habitat
            before node extraction. The default is 0.
        subdivide_region_voxels: In ``component`` mode, split components
            larger than this into grid blocks. ``0`` disables splitting.
            Ignored by ``uniform_grid``.
        block_size: Cube edge length in voxels (default 8), not millimetres.
            With the default ``distance_threshold=5``, face-adjacent
            8-cubes connect (closest voxels are one hop apart). One empty
            lattice cell between cubes is closest-voxel distance about 8
            and stays disconnected.
        block_min_coverage: Minimum occupied fraction of a cube to keep
            the cell (default 0.2; a cube is kept when coverage is
            strictly greater than this value). Applied per cell, not
            per subregion. Tiny in-cell fragments are dropped by
            ``min_region_voxels``.
        node_method: ``"uniform_grid"`` (default) or ``"component"``.
        include_background_shell: If True (default), add a peritumoral
            background shell as a reserved class (not a clustered
            habitat). Habitat nodes stay on the original VOI lattice.
        background_shell_width: Dilation width in voxels (>= 1). Default
            is a 1-voxel ring outside the ROI, clipped to the array.

    Returns:
        HabitatNodeExtractionResult: Nodes grouped by habitat label plus
        component maps used by contact-based edge builders. ``uniform_grid``
        also fills ``grid_origin`` / ``grid_block_size`` for dashed overlays.
        When the shell is on, ``nodes_by_habitat`` also has
        :data:`~habit.kernels.habitat_graph.models.BACKGROUND_SHELL_LABEL`.
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
    if node_method not in ("uniform_grid", "component"):
        raise ValueError("node_method must be 'uniform_grid' or 'component'.")
    if background_shell_width < 1:
        raise ValueError("background_shell_width must be >= 1.")

    labels = [int(v) for v in np.unique(label_array) if int(v) > 0]
    structure = _connectivity_structure(label_array.ndim, connectivity)
    if node_method == "uniform_grid":
        habitat_result = _extract_uniform_grid_nodes(
            label_array=label_array,
            labels=labels,
            structure=structure,
            min_region_voxels=min_region_voxels,
            erosion_radius=erosion_radius,
            block_size=block_size,
            block_min_coverage=block_min_coverage,
        )
    else:
        habitat_result = _extract_component_nodes(
            label_array=label_array,
            labels=labels,
            structure=structure,
            min_region_voxels=min_region_voxels,
            erosion_radius=erosion_radius,
            subdivide_region_voxels=subdivide_region_voxels,
            block_size=block_size,
            block_min_coverage=block_min_coverage,
        )
    if not include_background_shell:
        return habitat_result
    return _merge_background_shell_nodes(
        habitat_result,
        label_array=label_array,
        connectivity=connectivity,
        min_region_voxels=min_region_voxels,
        block_size=block_size,
        block_min_coverage=block_min_coverage,
        node_method=node_method,
        background_shell_width=background_shell_width,
    )
