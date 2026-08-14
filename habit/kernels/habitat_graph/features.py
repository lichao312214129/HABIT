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
"""Subject-level graph feature extraction from a habitat label map.

This is the L0 entry point of the habitat-graph kernel family: an integer
label array in, a flat feature dictionary out. It performs no IO, no logging,
and holds no state, so the exact feature definitions stay independently
reviewable and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import re
from typing import Dict, List, Optional, Sequence

import numpy as np

from habit.kernels.habitat_graph.edges import (
    build_adjacency_graph,
    build_centroid_distance_graph,
    iter_label_pairs,
)
from habit.kernels.habitat_graph.metrics import (
    calculate_pairwise_graph_metrics,
    calculate_single_graph_metrics,
)
from habit.kernels.habitat_graph.models import (
    EdgeMethod,
    EdgeWeightMode,
    HabitatGraphNode,
)
from habit.kernels.habitat_graph.nodes import extract_habitat_nodes

__all__ = [
    "HabitatGraphFeatureOptions",
    "extract_graph_features",
    "extract_graph_features_for_labels",
    "pair_count",
]


@dataclass(frozen=True)
class HabitatGraphFeatureOptions:
    """Runtime options for graph-based habitat feature extraction.

    Default edge rule: ``edge_method='adjacency'`` with
    ``adjacency_connectivity='face'`` and ``adjacency_min_voxels=10``. An
    edge exists when two regions share a boundary and the contact voxel
    count is at least 10.
    """

    include_single_habitat_graph: bool = True
    include_pairwise_habitat_graph: bool = True
    # Default edge rule: connect face-adjacent regions whose shared-boundary
    # (contact) voxel-pair count is at least ``adjacency_min_voxels``.
    edge_method: EdgeMethod = "adjacency"
    distance_threshold: float = 5.0
    # Adjacency edge parameters (used when edge_method == "adjacency").
    adjacency_connectivity: str = "face"
    adjacency_min_voxels: int = 10
    edge_weight: EdgeWeightMode = "none"
    min_region_voxels: int = 1
    connectivity: str = "face"
    # Default erosion (1 iteration) removes a one-voxel boundary shell to
    # suppress segmentation edge noise before connected-component labeling.
    erosion_radius: int = 1
    # Subdivision is on by default: contiguous habitats would otherwise collapse
    # into a single graph node and erase internal spatial structure. Components
    # larger than ``subdivide_region_voxels`` are split into ``block_size`` grid
    # blocks. ``block_size`` matches the default ``distance_threshold`` so that
    # adjacent blocks connect into a usable lattice graph.
    subdivide_region_voxels: int = 1000
    block_size: int = 5
    block_min_coverage: float = 0.5
    pairwise_include_intra_edges: bool = True
    include_extended_metrics: bool = True
    extended_min_nodes: int = 10


# Feature key suffixes grouped by physical dimension, used to attach VOI-size
# normalized companions. Each matching "<key>" gets an extra "<key>_norm".
# Distance-like values are scaled by the tumor bounding-box diagonal. Contact
# counts approximate an interface area, so they scale with V**((ndim-1)/ndim).
# Voxel sums and counts scale with V (fraction / density).
_LENGTH_NORM_SUFFIXES = (
    "_avg_edge_distance",
    "_std_edge_distance",
    "_spatial_dispersion",
)
_CONTACT_NORM_SUFFIXES = (
    "_contact_voxels_sum",
    "_contact_voxels_mean",
    "_contact_voxels_max",
)
_VOLUME_FRACTION_SUFFIXES = (
    "_avg_node_voxels",
    "_std_node_voxels",
)
_COUNT_NORM_SUFFIXES = (
    "_n_nodes",
    "_n_nodes_1",
    "_n_nodes_2",
    "_n_edges",
)


def _tumor_bbox_diagonal(label_array: np.ndarray) -> float:
    """
    Return the non-background tumor bounding-box diagonal in voxel units.

    Args:
        label_array: Integer habitat label map; background is encoded as 0.

    Returns:
        float: Euclidean diagonal length of the occupied half-open bounding box.
    """
    coords = np.argwhere(label_array > 0)
    if coords.size == 0:
        return 0.0
    bbox_lengths = coords.max(axis=0) - coords.min(axis=0) + 1
    return float(np.linalg.norm(bbox_lengths.astype(float)))


def _augment_with_normalized_features(
    features: Dict[str, float],
    label_array: np.ndarray,
) -> None:
    """
    Add VOI-size-normalized companions for size-dependent graph features.

    The tumor VOI measure ``V`` is the number of non-background voxels.
    Distance-like features are divided by the tumor bounding-box diagonal,
    contact features by ``V**((d-1)/d)``, and count/voxel features by ``V``.
    Original keys are preserved and each normalized value is stored under
    ``"<feature>_norm"``. Additional habitat-specific fractions are added with
    explicit suffixes when a per-label denominator is available.

    Args:
        features: Feature dictionary mutated in place with ``*_norm`` entries.
        label_array: Integer habitat label map; background (0) is excluded.
    """
    voi_voxels = float(np.count_nonzero(label_array))
    ndim = int(label_array.ndim)
    if voi_voxels <= 0 or ndim <= 0:
        return

    length_scale = _tumor_bbox_diagonal(label_array)
    contact_scale = voi_voxels ** ((ndim - 1.0) / ndim)
    volume_scale = voi_voxels
    habitat_volumes: Dict[int, float] = {
        int(label): float(np.count_nonzero(label_array == label))
        for label in np.unique(label_array)
        if int(label) > 0
    }

    def _scaled(value: float, scale: float) -> float:
        return float(value / scale) if scale > 0 else 0.0

    for key, value in list(features.items()):
        if key.endswith(_LENGTH_NORM_SUFFIXES):
            features[f"{key}_norm"] = _scaled(value, length_scale)
        elif key.endswith(_CONTACT_NORM_SUFFIXES):
            features[f"{key}_norm"] = _scaled(value, contact_scale)
        elif key.endswith(_VOLUME_FRACTION_SUFFIXES):
            features[f"{key}_norm"] = _scaled(value, volume_scale)
        elif key.endswith(_COUNT_NORM_SUFFIXES) or key == "graph_num_nodes_total":
            features[f"{key}_norm"] = _scaled(value, volume_scale)

        single_match = re.match(r"^single_h(?P<label>\d+)_(?P<suffix>.+)$", key)
        if single_match:
            label = int(single_match.group("label"))
            suffix = single_match.group("suffix")
            habitat_volume = habitat_volumes.get(label, 0.0)
            if suffix == "n_nodes":
                features[f"single_h{label}_n_nodes_per_habitat_volume"] = _scaled(
                    value,
                    habitat_volume,
                )
            elif suffix == "avg_node_voxels":
                features[f"single_h{label}_avg_node_voxels_fraction"] = _scaled(
                    value,
                    habitat_volume,
                )
            elif suffix == "std_node_voxels":
                features[f"single_h{label}_std_node_voxels_fraction"] = _scaled(
                    value,
                    habitat_volume,
                )
            continue

        pair_match = re.match(
            r"^pair_h(?P<label_a>\d+)_h(?P<label_b>\d+)_(?P<suffix>.+)$",
            key,
        )
        if pair_match:
            label_a = int(pair_match.group("label_a"))
            label_b = int(pair_match.group("label_b"))
            suffix = pair_match.group("suffix")
            if suffix == "n_nodes_1":
                features[
                    f"pair_h{label_a}_h{label_b}_n_nodes_1_per_habitat_volume"
                ] = _scaled(value, habitat_volumes.get(label_a, 0.0))
            elif suffix == "n_nodes_2":
                features[
                    f"pair_h{label_a}_h{label_b}_n_nodes_2_per_habitat_volume"
                ] = _scaled(value, habitat_volumes.get(label_b, 0.0))


def _flatten_nodes(
    nodes: Sequence[Sequence[HabitatGraphNode]],
) -> List[HabitatGraphNode]:
    """Flatten grouped node sequences while preserving deterministic order."""
    flattened: List[HabitatGraphNode] = []
    for group in nodes:
        flattened.extend(group)
    return flattened


def extract_graph_features(
    label_array: np.ndarray,
    *,
    options: HabitatGraphFeatureOptions = HabitatGraphFeatureOptions(),
    expected_labels: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    """
    Extract subject-level graph features from a habitat label map.

    Default ``options`` use ``edge_method='adjacency'`` with
    ``adjacency_min_voxels=10``: an edge exists when two regions are
    adjacent and the contact (shared-boundary) voxel count is >= 10.

    Args:
        label_array: Already segmented habitat map. Label 0 is treated as
            background and excluded from graph construction.
        options: Graph construction and metric options.
        expected_labels: Optional canonical habitat ids to report. When given,
            every listed label produces its ``single_h*`` columns and every
            unordered pair its ``pair_h*_h*`` columns even when the label is
            absent from this subject (empty graphs yield zero-valued metrics),
            so cohort-level tables have stable columns. When ``None``, only
            labels actually present in ``label_array`` are reported (the
            historical v0.1 behaviour).

    Returns:
        Dict[str, float]: Flat feature dictionary ready for table assembly.
    """
    labels_array = np.asarray(label_array).astype(np.int32, copy=False)
    node_result = extract_habitat_nodes(
        label_array=labels_array,
        connectivity=options.connectivity,
        min_region_voxels=options.min_region_voxels,
        erosion_radius=options.erosion_radius,
        subdivide_region_voxels=options.subdivide_region_voxels,
        block_size=options.block_size,
        block_min_coverage=options.block_min_coverage,
    )

    present_labels = sorted(node_result.nodes_by_habitat.keys())
    if expected_labels is not None:
        habitat_labels = sorted(int(label) for label in expected_labels)
    else:
        habitat_labels = present_labels
    features: Dict[str, float] = {
        # Count of habitats actually present in this subject, independent of
        # the canonical id set used for column coverage.
        "graph_num_habitats": float(len(present_labels)),
        "graph_num_nodes_total": float(
            sum(len(nodes) for nodes in node_result.nodes_by_habitat.values())
        ),
    }

    if options.include_single_habitat_graph:
        for habitat_label in habitat_labels:
            # Labels listed in ``expected_labels`` but absent from this map
            # yield an empty node list, i.e. a zero-valued empty graph.
            nodes = node_result.nodes_by_habitat.get(habitat_label, [])
            if options.edge_method == "adjacency":
                graph = build_adjacency_graph(
                    node_result=node_result,
                    labels=(habitat_label,),
                    graph_kind="single",
                    adjacency_connectivity=options.adjacency_connectivity,
                    adjacency_min_voxels=options.adjacency_min_voxels,
                    edge_weight=options.edge_weight,
                )
            else:
                graph = build_centroid_distance_graph(
                    nodes=nodes,
                    labels=(habitat_label,),
                    graph_kind="single",
                    distance_threshold=options.distance_threshold,
                    edge_weight=options.edge_weight,
                )
            features.update(
                calculate_single_graph_metrics(
                    graph,
                    include_extended_metrics=options.include_extended_metrics,
                    extended_min_nodes=options.extended_min_nodes,
                )
            )

    if options.include_pairwise_habitat_graph:
        for label_a, label_b in iter_label_pairs(habitat_labels):
            pair_nodes = _flatten_nodes(
                [
                    node_result.nodes_by_habitat.get(label_a, []),
                    node_result.nodes_by_habitat.get(label_b, []),
                ]
            )
            if options.edge_method == "adjacency":
                graph = build_adjacency_graph(
                    node_result=node_result,
                    labels=(label_a, label_b),
                    graph_kind="pairwise",
                    adjacency_connectivity=options.adjacency_connectivity,
                    adjacency_min_voxels=options.adjacency_min_voxels,
                    edge_weight=options.edge_weight,
                    include_intra_edges=options.pairwise_include_intra_edges,
                )
            else:
                graph = build_centroid_distance_graph(
                    nodes=pair_nodes,
                    labels=(label_a, label_b),
                    graph_kind="pairwise",
                    distance_threshold=options.distance_threshold,
                    edge_weight=options.edge_weight,
                    include_intra_edges=options.pairwise_include_intra_edges,
                )
            features.update(
                calculate_pairwise_graph_metrics(
                    graph,
                    include_extended_metrics=options.include_extended_metrics,
                    extended_min_nodes=options.extended_min_nodes,
                )
            )

    _augment_with_normalized_features(features, labels_array)
    return features


def extract_graph_features_for_labels(
    label_array: np.ndarray,
    labels: Sequence[int],
    *,
    options: HabitatGraphFeatureOptions = HabitatGraphFeatureOptions(),
) -> Dict[str, float]:
    """
    Extract graph features after restricting the habitat map to selected labels.

    Args:
        label_array: Already segmented habitat map.
        labels: Habitat labels to keep. Other labels are set to background.
        options: Graph construction and metric options.

    Returns:
        Dict[str, float]: Flat feature dictionary for the selected labels.
    """
    labels_array = np.asarray(label_array).astype(np.int32, copy=False)
    selected_labels = np.asarray([int(label) for label in labels], dtype=np.int32)
    keep_mask = np.isin(labels_array, selected_labels)
    restricted = np.where(keep_mask, labels_array, 0).astype(np.int32, copy=False)
    return extract_graph_features(restricted, options=options)


def pair_count(label_count: int) -> int:
    """
    Return the number of pairwise graphs for a given habitat-label count.

    Args:
        label_count: Number of non-background habitat labels.

    Returns:
        int: Number of unique unordered label pairs.
    """
    if label_count < 2:
        return 0
    return sum(1 for _ in combinations(range(label_count), 2))
