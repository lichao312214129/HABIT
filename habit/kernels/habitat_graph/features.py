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
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from habit.kernels.habitat_graph.array_graph import graph_arrays_from_table
from habit.kernels.habitat_graph.edges import (
    as_intra_edge,
    build_adjacency_graph,
    build_centroid_distance_graph,
    build_centroid_inter_edges,
    build_min_distance_edge_table,
    build_min_distance_edges,
    build_min_distance_graph,
    build_min_distance_inter_edges,
    compose_pairwise_graph,
    iter_label_pairs,
)
from habit.kernels.habitat_graph.metrics import (
    _extended_pairwise_from_arrays,
    _extended_single_from_arrays,
    _pairwise_features_from_arrays,
    _single_features_from_arrays,
    calculate_pairwise_graph_metrics,
    calculate_single_graph_metrics,
)
from habit.kernels.habitat_graph.models import (
    EdgeMethod,
    EdgeWeightMode,
    HabitatGraph,
    HabitatGraphEdge,
    HabitatGraphNode,
    HabitatNodeExtractionResult,
    NodeMethod,
    pair_feature_prefix,
    single_feature_prefix,
)
from habit.kernels.habitat_graph.nodes import (
    _nonzero_bbox_slices,
    extract_habitat_nodes,
)

__all__ = [
    "HabitatGraphFeatureOptions",
    "extract_graph_features",
    "extract_graph_features_for_labels",
    "pair_count",
]


@dataclass(frozen=True)
class HabitatGraphFeatureOptions:
    """Runtime options for graph-based habitat feature extraction.

    Default node rule: ``node_method='uniform_grid'`` with ``block_size=8``
    (global VOI lattice; each kept cube emits one node per connected
    habitat subregion at that subregion's voxel centroid; edge length in
    **voxels**, not millimetres). Default edge rule: ``edge_method='min_distance'``
    with ``distance_threshold=5.0``. Face-adjacent 8-cubes connect
    (closest voxels are one hop apart). One empty lattice cell between
    cubes is closest-voxel distance about 8, which is greater than 5, so
    those stay disconnected. ``adjacency`` and ``centroid_distance``
    remain available.
    ``erosion_radius=0`` measures the habitat labels as drawn.
    """

    include_single_habitat_graph: bool = True
    include_pairwise_habitat_graph: bool = True
    # Default: connect regions whose closest voxels are within
    # ``distance_threshold``. Pass ``"adjacency"`` for contact-voxel edges
    # or ``"centroid_distance"`` for centroid proximity.
    edge_method: EdgeMethod = "min_distance"
    distance_threshold: float = 5.0
    # Adjacency edge parameters (used when edge_method == "adjacency").
    adjacency_connectivity: str = "corner"
    adjacency_min_voxels: int = 10
    edge_weight: EdgeWeightMode = "none"
    min_region_voxels: int = 1
    connectivity: str = "full"
    # Default is off: distances / contact are measured on the habitat labels
    # as drawn. Pass a positive value to shrink each habitat (binary erosion
    # iterations) before labeling and edge construction.
    erosion_radius: int = 0
    # Default node construction: a global cube lattice; each kept cell
    # emits one node per connected habitat subregion (centroid).
    # ``component`` restores connected-component nodes; those larger than
    # ``subdivide_region_voxels`` are then split (``0`` disables that split).
    node_method: NodeMethod = "uniform_grid"
    subdivide_region_voxels: int = 1000
    # 8^3 = 512 voxels per full cube. Units are voxels, not millimetres.
    # Paired with distance_threshold=5: face-adjacent 8-cubes connect
    # (closest-voxel hop 1); one empty lattice cell is about 8 > 5.
    block_size: int = 8
    block_min_coverage: float = 0.2
    pairwise_include_intra_edges: bool = True
    # Default True: efficiency / small-world / rich-club / node
    # distributions. On typical habitat maps this is cheap (analytic
    # Humphries S, not a rewire ensemble). Pass False to omit.
    include_extended_metrics: bool = True
    extended_min_nodes: int = 10
    # Default small-world is Humphries analytic ER *S* (one column).
    # ``config`` / ``rewire`` replace that column with a degree-preserving
    # ensemble; they do not add a second sigma.
    small_world_nrand: int = 100
    small_world_niter: int = 100
    rich_club_q: int = 100
    graph_null_sampler: str = "analytic"
    graph_null_device: str = "auto"


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
# Only a contact *sum* has the dimensional behavior of a whole interface.
# Mean and maximum contact are local edge summaries and are normalized against
# the local node scale in ``calculate_pairwise_graph_metrics`` instead.
_CONTACT_NORM_SUFFIXES = ("_contact_voxels_sum",)
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


def _crop_to_tumor_voi(label_array: np.ndarray) -> np.ndarray:
    """
    Restrict a habitat map to the tumour VOI plus one voxel of pad.

    One-step / two-step habitat maps are often stored on the full CT
    lattice. Graph construction and VOI-normalized companions only
    depend on non-background voxels; scanning the empty field is
    wasted work and must not change any reported number (bbox
    diagonals are translation-invariant).

    Args:
        label_array: Integer habitat map; ``0`` is background.

    Returns:
        np.ndarray: Contiguous crop of the occupied bounding box, or
        the original array when the map is empty.
    """
    boxed = _nonzero_bbox_slices(np.asarray(label_array), pad=1)
    if boxed is None:
        return np.asarray(label_array)
    slices, _offset = boxed
    return np.ascontiguousarray(label_array[slices])


def _habitat_voxel_counts(label_array: np.ndarray) -> Dict[int, float]:
    """Count non-background voxels per habitat on a (usually cropped) map."""
    flat = np.asarray(label_array).ravel()
    if flat.size == 0:
        return {}
    max_label = int(flat.max()) if flat.size else 0
    if max_label <= 0:
        return {}
    counts = np.bincount(np.clip(flat, 0, max_label), minlength=max_label + 1)
    return {
        int(label): float(counts[label])
        for label in range(1, max_label + 1)
        if int(counts[label]) > 0
    }


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


def _label_bbox_diagonal(label_array: np.ndarray, labels: Sequence[int]) -> float:
    """
    Return the voxel-space bounding-box diagonal for selected habitat labels.

    Background and the optional synthetic shell are not present in
    ``label_array``. A zero result therefore means that none of ``labels`` is
    an observed positive habitat label.

    Args:
        label_array: Integer habitat label map; background is encoded as 0.
        labels: Positive habitat labels whose union defines the bounding box.

    Returns:
        float: Euclidean diagonal of the occupied bounding box in voxel units,
        or ``0.0`` when no requested positive label is present.
    """
    positive_labels = [int(label) for label in labels if int(label) > 0]
    if not positive_labels:
        return 0.0
    coords = np.argwhere(np.isin(label_array, positive_labels))
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
    total interface contact by ``V**((d-1)/d)``, and count/voxel features by
    ``V``. Original keys are preserved and each normalized value is stored
    under ``"<feature>_norm"``. Additional habitat-specific densities and
    spatial scales are added with explicit suffixes when their denominator is
    available.

    Args:
        features: Feature dictionary mutated in place with ``*_norm`` entries.
        label_array: Integer habitat label map; background (0) is excluded.
            A full-CT lattice is cropped to the tumour VOI first.
    """
    voi = _crop_to_tumor_voi(label_array)
    voi_voxels = float(np.count_nonzero(voi))
    ndim = int(voi.ndim)
    if voi_voxels <= 0 or ndim <= 0:
        return

    length_scale = _tumor_bbox_diagonal(voi)
    contact_scale = voi_voxels ** ((ndim - 1.0) / ndim)
    volume_scale = voi_voxels
    habitat_volumes = _habitat_voxel_counts(voi)
    # One bbox walk per habitat / pair, never per feature column.
    single_bbox: Dict[int, float] = {
        hid: _label_bbox_diagonal(voi, (hid,)) for hid in habitat_volumes
    }
    pair_bbox: Dict[Tuple[int, int], float] = {}

    def _pair_bbox(label_a: int, label_b: int) -> float:
        key = (min(int(label_a), int(label_b)), max(int(label_a), int(label_b)))
        cached = pair_bbox.get(key)
        if cached is None:
            cached = _label_bbox_diagonal(voi, key)
            pair_bbox[key] = cached
        return cached

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
            if suffix in {"n_nodes", "n_edges", "connected_components"}:
                features[f"single_h{label}_{suffix}_per_habitat_volume"] = _scaled(
                    value, habitat_volume
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
            elif suffix in {
                "avg_edge_distance",
                "std_edge_distance",
                "spatial_dispersion",
            }:
                features[
                    f"single_h{label}_{suffix}_per_habitat_bbox_diagonal"
                ] = _scaled(value, single_bbox.get(label, 0.0))
            continue

        pair_match = re.match(
            r"^pair_h(?P<label_a>\d+)_h(?P<label_b>\d+)_(?P<suffix>.+)$",
            key,
        )
        if pair_match:
            label_a = int(pair_match.group("label_a"))
            label_b = int(pair_match.group("label_b"))
            suffix = pair_match.group("suffix")
            pair_volume = (
                habitat_volumes.get(label_a, 0.0) + habitat_volumes.get(label_b, 0.0)
            )
            pair_area_scale = (
                pair_volume ** ((ndim - 1.0) / ndim) if pair_volume > 0 else 0.0
            )
            if suffix == "n_nodes_1":
                features[
                    f"pair_h{label_a}_h{label_b}_n_nodes_1_per_habitat_volume"
                ] = _scaled(value, habitat_volumes.get(label_a, 0.0))
            elif suffix == "n_nodes_2":
                features[
                    f"pair_h{label_a}_h{label_b}_n_nodes_2_per_habitat_volume"
                ] = _scaled(value, habitat_volumes.get(label_b, 0.0))
            elif suffix == "contact_voxels_sum":
                features[
                    f"pair_h{label_a}_h{label_b}_"
                    "contact_voxels_sum_per_pair_area_scale"
                ] = _scaled(value, pair_area_scale)
            elif suffix in {"avg_edge_distance", "std_edge_distance"}:
                features[
                    f"pair_h{label_a}_h{label_b}_"
                    f"{suffix}_per_pair_bbox_diagonal"
                ] = _scaled(value, _pair_bbox(label_a, label_b))
            continue


def _metric_kwargs(options: HabitatGraphFeatureOptions) -> Dict[str, object]:
    """Shared keyword arguments for single / pairwise metric kernels."""
    return {
        "include_extended_metrics": options.include_extended_metrics,
        "extended_min_nodes": options.extended_min_nodes,
        "small_world_nrand": options.small_world_nrand,
        "small_world_niter": options.small_world_niter,
        "rich_club_q": options.rich_club_q,
        "graph_null_sampler": options.graph_null_sampler,
        "graph_null_device": options.graph_null_device,
    }


def _compute_graph_metrics_job(
    kind: str,
    graph: HabitatGraph,
    kwargs: Dict[str, object],
) -> Dict[str, float]:
    """Module-level worker so process pools can pickle the metric call."""
    if kind == "single":
        return calculate_single_graph_metrics(graph, **kwargs)  # type: ignore[arg-type]
    return calculate_pairwise_graph_metrics(graph, **kwargs)  # type: ignore[arg-type]


def _run_metric_jobs(
    jobs: Sequence[Tuple[str, HabitatGraph]],
    options: HabitatGraphFeatureOptions,
) -> List[Dict[str, float]]:
    """
    Evaluate independent graph-metric jobs.

    Each job is a different graph, so process-parallel Brandes / Louvain
    is safe. One or two graphs stay in-process (spawn overhead dominates).
    """
    kwargs = _metric_kwargs(options)
    largest = max((len(graph.nodes) for _kind, graph in jobs), default=0)
    # Spawn is more expensive than Brandes on the tiny graphs used in tests.
    if len(jobs) <= 2 or largest < 80:
        return [_compute_graph_metrics_job(kind, graph, kwargs) for kind, graph in jobs]
    try:
        from joblib import Parallel, delayed
    except Exception:
        return [_compute_graph_metrics_job(kind, graph, kwargs) for kind, graph in jobs]
    return list(
        Parallel(n_jobs=-1, prefer="threads")(
            delayed(_compute_graph_metrics_job)(kind, graph, kwargs)
            for kind, graph in jobs
        )
    )


def _true_habitat_labels(labels: Sequence[int]) -> List[int]:
    """Return the sorted positive habitat labels present in ``labels``."""
    return sorted(int(label) for label in labels if int(label) > 0)


def _slice_min_distance_edges(
    edges: Sequence[HabitatGraphEdge],
    nodes: Sequence[HabitatGraphNode],
    *,
    as_intra: bool,
) -> List[HabitatGraphEdge]:
    """Keep table edges whose endpoints both sit in ``nodes``."""
    allowed = {node.node_id for node in nodes}
    labels = {node.node_id: int(node.habitat_label) for node in nodes}
    sliced: List[HabitatGraphEdge] = []
    for edge in edges:
        if edge.source not in allowed or edge.target not in allowed:
            continue
        if as_intra and labels[edge.source] == labels[edge.target]:
            sliced.append(as_intra_edge(edge))
        elif (not as_intra) or labels[edge.source] != labels[edge.target]:
            sliced.append(edge)
    return sliced


def _build_single_graph(
    node_result: HabitatNodeExtractionResult,
    nodes: Sequence[HabitatGraphNode],
    habitat_label: int,
    options: HabitatGraphFeatureOptions,
    min_distance_edges: Optional[Sequence[HabitatGraphEdge]] = None,
) -> HabitatGraph:
    """Build one single-habitat graph with the requested edge rule."""
    if options.edge_method == "adjacency":
        return build_adjacency_graph(
            node_result=node_result,
            labels=(habitat_label,),
            graph_kind="single",
            adjacency_connectivity=options.adjacency_connectivity,
            adjacency_min_voxels=options.adjacency_min_voxels,
            edge_weight=options.edge_weight,
        )
    if options.edge_method == "min_distance":
        if min_distance_edges is not None:
            return HabitatGraph(
                graph_kind="single",
                labels=(int(habitat_label),),
                nodes={node.node_id: node for node in nodes},
                edges=_slice_min_distance_edges(
                    min_distance_edges, nodes, as_intra=False
                ),
            )
        return build_min_distance_graph(
            node_result=node_result,
            labels=(habitat_label,),
            graph_kind="single",
            distance_threshold=options.distance_threshold,
            edge_weight=options.edge_weight,
        )
    return build_centroid_distance_graph(
        nodes=nodes,
        labels=(habitat_label,),
        graph_kind="single",
        distance_threshold=options.distance_threshold,
        edge_weight=options.edge_weight,
    )


def _reused_intra_edges(
    single_graphs: Mapping[int, HabitatGraph],
    label: int,
) -> List[HabitatGraphEdge]:
    """Intra edges already measured on the single-habitat graph, retagged."""
    graph = single_graphs.get(int(label))
    if graph is None:
        return []
    return [as_intra_edge(edge) for edge in graph.edges]


def _build_pairwise_graph(
    node_result: HabitatNodeExtractionResult,
    pair_nodes: Sequence[HabitatGraphNode],
    label_a: int,
    label_b: int,
    options: HabitatGraphFeatureOptions,
    single_graphs: Mapping[int, HabitatGraph],
    min_distance_edges: Optional[Sequence[HabitatGraphEdge]] = None,
) -> HabitatGraph:
    """Build a pairwise graph, reusing intra edges when singles already exist.

    ``min_distance`` and ``centroid_distance`` only measure cross-habitat
    pairs when both single-habitat graphs are cached. ``adjacency`` still
    paints the pair once (contact counting is one voxel pass).
    """
    reuse_intra = (
        options.pairwise_include_intra_edges
        and int(label_a) in single_graphs
        and int(label_b) in single_graphs
    )
    if options.edge_method == "adjacency":
        return build_adjacency_graph(
            node_result=node_result,
            labels=(label_a, label_b),
            graph_kind="pairwise",
            adjacency_connectivity=options.adjacency_connectivity,
            adjacency_min_voxels=options.adjacency_min_voxels,
            edge_weight=options.edge_weight,
            include_intra_edges=options.pairwise_include_intra_edges,
        )
    if options.edge_method == "min_distance":
        if min_distance_edges is not None:
            sliced = _slice_min_distance_edges(
                min_distance_edges, pair_nodes, as_intra=True
            )
            if not options.pairwise_include_intra_edges:
                sliced = [edge for edge in sliced if edge.edge_type != "intra"]
            inter_edges = [edge for edge in sliced if edge.edge_type != "intra"]
            intra_edges = [edge for edge in sliced if edge.edge_type == "intra"]
            return compose_pairwise_graph(
                pair_nodes, (label_a, label_b), inter_edges, intra_edges
            )
        if reuse_intra:
            inter_edges = build_min_distance_inter_edges(
                node_result,
                (label_a, label_b),
                options.distance_threshold,
                options.edge_weight,
            )
            intra_edges = _reused_intra_edges(single_graphs, label_a)
            intra_edges.extend(_reused_intra_edges(single_graphs, label_b))
            return compose_pairwise_graph(
                pair_nodes, (label_a, label_b), inter_edges, intra_edges
            )
        return build_min_distance_graph(
            node_result=node_result,
            labels=(label_a, label_b),
            graph_kind="pairwise",
            distance_threshold=options.distance_threshold,
            edge_weight=options.edge_weight,
            include_intra_edges=options.pairwise_include_intra_edges,
        )
    if reuse_intra:
        nodes_a = node_result.nodes_by_habitat.get(label_a, [])
        nodes_b = node_result.nodes_by_habitat.get(label_b, [])
        inter_edges = build_centroid_inter_edges(
            nodes_a,
            nodes_b,
            options.distance_threshold,
            options.edge_weight,
        )
        intra_edges = _reused_intra_edges(single_graphs, label_a)
        intra_edges.extend(_reused_intra_edges(single_graphs, label_b))
        return compose_pairwise_graph(
            pair_nodes, (label_a, label_b), inter_edges, intra_edges
        )
    return build_centroid_distance_graph(
        nodes=pair_nodes,
        labels=(label_a, label_b),
        graph_kind="pairwise",
        distance_threshold=options.distance_threshold,
        edge_weight=options.edge_weight,
        include_intra_edges=options.pairwise_include_intra_edges,
    )


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

    Default ``options`` use ``node_method='uniform_grid'`` (8-voxel cubes,
    one node per in-cell subregion centroid) and
    ``edge_method='min_distance'`` with ``distance_threshold=5.0``.
    Pass ``node_method='component'`` / ``edge_method='adjacency'`` for the
    older connected-component contact graph.

    Args:
        label_array: Already segmented habitat map. Label 0 is treated as
            background and excluded from graph construction. A full-CT
            lattice is cropped to the tumour VOI before nodes, edges,
            metrics, and size-normalized companions are computed.
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
    labels_array = _crop_to_tumor_voi(
        np.asarray(label_array).astype(np.int32, copy=False)
    )
    node_result = extract_habitat_nodes(
        label_array=labels_array,
        connectivity=options.connectivity,
        min_region_voxels=options.min_region_voxels,
        erosion_radius=options.erosion_radius,
        subdivide_region_voxels=options.subdivide_region_voxels,
        block_size=options.block_size,
        block_min_coverage=options.block_min_coverage,
        node_method=options.node_method,
    )

    present_labels = _true_habitat_labels(node_result.nodes_by_habitat.keys())
    if expected_labels is not None:
        habitat_labels = _true_habitat_labels(expected_labels)
    else:
        habitat_labels = present_labels
    features: Dict[str, float] = {
        "graph_num_habitats": float(len(present_labels)),
        "graph_num_nodes_total": float(
            sum(len(nodes) for nodes in node_result.nodes_by_habitat.values())
        ),
    }

    single_labels = list(habitat_labels)
    # min_distance stays on the CSR array path even when extended
    # metrics are on (library default). Extended columns are attached
    # from the same arrays; the scientific definitions do not change.
    use_array_path = options.edge_method == "min_distance"
    if use_array_path and (
        options.include_single_habitat_graph or options.include_pairwise_habitat_graph
    ):
        all_nodes = _flatten_nodes(
            [node_result.nodes_by_habitat.get(label, []) for label in single_labels]
        )
        table = build_min_distance_edge_table(
            node_result,
            all_nodes,
            options.distance_threshold,
        )
        metric_jobs_arrays: List[Tuple[str, object]] = []
        if options.include_single_habitat_graph:
            for habitat_label in single_labels:
                arrays = graph_arrays_from_table(
                    all_nodes,
                    table,
                    (int(habitat_label),),
                    "single",
                    include_intra=True,
                    edge_weight=options.edge_weight,
                )
                nodes = node_result.nodes_by_habitat.get(habitat_label, [])
                metric_jobs_arrays.append(("single", (arrays, nodes)))
        if options.include_pairwise_habitat_graph:
            for label_a, label_b in iter_label_pairs(habitat_labels):
                arrays = graph_arrays_from_table(
                    all_nodes,
                    table,
                    (int(label_a), int(label_b)),
                    "pairwise",
                    include_intra=options.pairwise_include_intra_edges,
                    edge_weight=options.edge_weight,
                )
                metric_jobs_arrays.append(("pairwise", (arrays, ())))
        largest = max(
            (
                len(payload[0].node_ids)  # type: ignore[index]
                for _kind, payload in metric_jobs_arrays
            ),
            default=0,
        )

        def _run_array_job(kind: str, payload: object) -> Dict[str, float]:
            if kind == "single":
                arrays, nodes = payload  # type: ignore[misc]
                features_job = _single_features_from_arrays(arrays, nodes)
                if options.include_extended_metrics:
                    features_job.update(
                        _extended_single_from_arrays(
                            arrays,
                            single_feature_prefix(arrays.labels[0]),
                            extended_min_nodes=options.extended_min_nodes,
                            small_world_nrand=options.small_world_nrand,
                            small_world_niter=options.small_world_niter,
                            rich_club_q=options.rich_club_q,
                            graph_null_sampler=options.graph_null_sampler,
                            graph_null_device=options.graph_null_device,
                        )
                    )
                return features_job
            arrays, _unused = payload  # type: ignore[misc]
            features_job = _pairwise_features_from_arrays(arrays)
            if options.include_extended_metrics:
                features_job.update(
                    _extended_pairwise_from_arrays(
                        arrays,
                        pair_feature_prefix(arrays.labels[0], arrays.labels[1]),
                        extended_min_nodes=options.extended_min_nodes,
                        small_world_nrand=options.small_world_nrand,
                        small_world_niter=options.small_world_niter,
                        rich_club_q=options.rich_club_q,
                        graph_null_sampler=options.graph_null_sampler,
                        graph_null_device=options.graph_null_device,
                    )
                )
            return features_job

        if len(metric_jobs_arrays) <= 2 or largest < 80:
            payloads = [
                _run_array_job(kind, payload)
                for kind, payload in metric_jobs_arrays
            ]
        else:
            try:
                from joblib import Parallel, delayed

                payloads = list(
                    Parallel(n_jobs=-1, prefer="threads")(
                        delayed(_run_array_job)(kind, payload)
                        for kind, payload in metric_jobs_arrays
                    )
                )
            except Exception:
                payloads = [
                    _run_array_job(kind, payload)
                    for kind, payload in metric_jobs_arrays
                ]
        for payload in payloads:
            features.update(payload)
        _augment_with_normalized_features(features, labels_array)
        return features

    min_distance_edges: List[HabitatGraphEdge] = []
    if options.edge_method == "min_distance" and (
        options.include_single_habitat_graph or options.include_pairwise_habitat_graph
    ):
        all_nodes = _flatten_nodes(
            [node_result.nodes_by_habitat.get(label, []) for label in single_labels]
        )
        min_distance_edges = build_min_distance_edges(
            node_result,
            all_nodes,
            options.distance_threshold,
            options.edge_weight,
        )
    single_graphs: Dict[int, HabitatGraph] = {}
    metric_jobs: List[Tuple[str, HabitatGraph]] = []

    if options.include_single_habitat_graph:
        for habitat_label in single_labels:
            nodes = node_result.nodes_by_habitat.get(habitat_label, [])
            graph = _build_single_graph(
                node_result,
                nodes,
                habitat_label,
                options,
                min_distance_edges=min_distance_edges,
            )
            single_graphs[int(habitat_label)] = graph
            metric_jobs.append(("single", graph))

    if options.include_pairwise_habitat_graph:
        pair_labels = list(iter_label_pairs(habitat_labels))
        for label_a, label_b in pair_labels:  # noqa: B007
            pair_nodes = _flatten_nodes(
                [
                    node_result.nodes_by_habitat.get(label_a, []),
                    node_result.nodes_by_habitat.get(label_b, []),
                ]
            )
            graph = _build_pairwise_graph(
                node_result,
                pair_nodes,
                int(label_a),
                int(label_b),
                options,
                single_graphs,
                min_distance_edges=min_distance_edges,
            )
            metric_jobs.append(("pairwise", graph))

    for payload in _run_metric_jobs(metric_jobs, options):
        features.update(payload)

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
    labels_array = _crop_to_tumor_voi(
        np.asarray(label_array).astype(np.int32, copy=False)
    )
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
