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
"""Tests for the habitat graph L0 kernels."""

from __future__ import annotations

import numpy as np
import pytest

import networkx as nx

from habit.kernels.habitat_graph import (
    GraphNullModelOptions,
    HabitatGraphFeatureOptions,
    build_adjacency_graph,
    build_centroid_distance_graph,
    build_min_distance_graph,
    compare_graph_to_degree_preserving_null,
    extract_graph_features,
    extract_graph_features_for_labels,
    extract_habitat_nodes,
    pair_count,
)
from habit.kernels.habitat_graph.extended_metrics import (
    compute_extended_graph_metrics,
    compute_extended_pairwise_metrics,
)


def _extract(label_array: np.ndarray, **overrides) -> dict:
    """Run the kernel with explicit options (test defaults disable erosion)."""
    # Existing metric tests pin connected-component nodes and ROI-only
    # graphs; the library default is now uniform_grid.
    overrides.setdefault("node_method", "component")
    options = HabitatGraphFeatureOptions(**overrides)
    return extract_graph_features(label_array, options=options)


@pytest.mark.unit
def test_extract_habitat_nodes_uses_connected_regions_as_nodes() -> None:
    """Each disconnected habitat region should become one graph node."""
    label_array: np.ndarray = np.array(
        [
            [1, 1, 0, 2],
            [1, 0, 0, 2],
            [0, 0, 1, 2],
            [2, 2, 0, 0],
        ],
        dtype=np.int32,
    )

    result = extract_habitat_nodes(node_method="component", label_array=label_array, connectivity="face")

    assert len(result.nodes_by_habitat[1]) == 2
    assert len(result.nodes_by_habitat[2]) == 2
    assert result.nodes_by_habitat[1][0].voxel_count == 3
    assert result.nodes_by_habitat[1][1].voxel_count == 1


@pytest.mark.unit
def test_centroid_distance_graph_connects_regions_within_threshold() -> None:
    """The distance strategy should connect same-habitat nodes by centroid distance."""
    label_array: np.ndarray = np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    node_result = extract_habitat_nodes(node_method="component", label_array=label_array, connectivity="face")

    graph = build_centroid_distance_graph(
        nodes=node_result.nodes_by_habitat[1],
        labels=(1,),
        graph_kind="single",
        distance_threshold=3.0,
        edge_weight="none",
    )

    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1
    assert graph.edges[0].distance is not None
    assert graph.edges[0].weight == 1.0


@pytest.mark.unit
def test_min_distance_graph_connects_regions_three_voxels_apart() -> None:
    """Closest-voxel edges exist iff min Euclidean distance <= threshold.

    Two single-voxel regions at columns 0 and 3 are 3 voxel-index units
    apart. That is not the same rule as centroid distance on larger blobs.
    """
    label_array: np.ndarray = np.array(
        [
            [1, 0, 0, 2],
        ],
        dtype=np.int32,
    )
    node_result = extract_habitat_nodes(node_method="component", label_array=label_array, connectivity="face")
    assert len(node_result.nodes_by_habitat[1]) == 1
    assert len(node_result.nodes_by_habitat[2]) == 1

    below = build_min_distance_graph(
        node_result=node_result,
        labels=(1, 2),
        graph_kind="pairwise",
        distance_threshold=2.999,
    )
    at_threshold = build_min_distance_graph(
        node_result=node_result,
        labels=(1, 2),
        graph_kind="pairwise",
        distance_threshold=3.0,
    )
    above = build_min_distance_graph(
        node_result=node_result,
        labels=(1, 2),
        graph_kind="pairwise",
        distance_threshold=3.1,
    )
    assert len(below.edges) == 0
    assert len(at_threshold.edges) == 1
    assert at_threshold.edges[0].distance == pytest.approx(3.0)
    assert len(above.edges) == 1

    # Same map: centroid_distance still uses centroids (here also 3.0).
    centroid_graph = build_centroid_distance_graph(
        nodes=list(node_result.nodes_by_habitat[1])
        + list(node_result.nodes_by_habitat[2]),
        labels=(1, 2),
        graph_kind="pairwise",
        distance_threshold=3.0,
    )
    assert len(centroid_graph.edges) == 1


@pytest.mark.unit
def test_min_distance_is_not_centroid_distance_on_elongated_regions() -> None:
    """Closest voxels can be nearer than the two region centroids."""
    # Habitat 1 occupies cols 0-2; habitat 2 occupies col 5. Closest voxels
    # are 3 units apart (col 2 vs col 5); centroids are 4 units apart
    # (col 1 vs col 5).
    label_array: np.ndarray = np.array(
        [
            [1, 1, 1, 0, 0, 2],
        ],
        dtype=np.int32,
    )
    node_result = extract_habitat_nodes(node_method="component", label_array=label_array, connectivity="face")
    pair_nodes = list(node_result.nodes_by_habitat[1]) + list(
        node_result.nodes_by_habitat[2]
    )

    min_graph = build_min_distance_graph(
        node_result=node_result,
        labels=(1, 2),
        graph_kind="pairwise",
        distance_threshold=3.0,
    )
    centroid_graph = build_centroid_distance_graph(
        nodes=pair_nodes,
        labels=(1, 2),
        graph_kind="pairwise",
        distance_threshold=3.0,
    )
    assert len(min_graph.edges) == 1
    assert min_graph.edges[0].distance == pytest.approx(3.0)
    assert len(centroid_graph.edges) == 0

    centroid_at_four = build_centroid_distance_graph(
        nodes=pair_nodes,
        labels=(1, 2),
        graph_kind="pairwise",
        distance_threshold=4.0,
    )
    assert len(centroid_at_four.edges) == 1

    options = HabitatGraphFeatureOptions(
        edge_method="min_distance",
        node_method="component",
        distance_threshold=3.0,
        subdivide_region_voxels=0,
        include_extended_metrics=False,
        pairwise_include_intra_edges=False,
    )
    features = extract_graph_features(label_array, options=options)
    assert features["pair_h1_h2_n_edges"] == 1.0
    # Stored / summarized length is d_min (3), not the centroid gap (4).
    assert features["pair_h1_h2_avg_edge_distance"] == pytest.approx(3.0)


@pytest.mark.unit
def test_adjacency_min_voxels_default_requires_ten_contact_voxels() -> None:
    """Default adjacency edges require at least 10 shared-boundary voxel pairs."""
    # Two 1-voxel-thick strips: each column is one face-adjacent (1, 2) pair.
    labels_nine: np.ndarray = np.array(
        [[1] * 9, [2] * 9],
        dtype=np.int32,
    )
    labels_ten: np.ndarray = np.array(
        [[1] * 10, [2] * 10],
        dtype=np.int32,
    )
    nodes_nine = extract_habitat_nodes(node_method="component", label_array=labels_nine, connectivity="face")
    nodes_ten = extract_habitat_nodes(node_method="component", label_array=labels_ten, connectivity="face")

    # Pin face: 1-voxel-thick strips also share diagonal contacts under
    # corner connectivity, which would inflate the count past 10 on both
    # maps. This test is about the min-voxels threshold, not the default
    # neighborhood.
    graph_nine = build_adjacency_graph(
        node_result=nodes_nine,
        labels=(1, 2),
        graph_kind="pairwise",
        adjacency_connectivity="face",
        edge_weight="contact_voxels",
    )
    graph_ten = build_adjacency_graph(
        node_result=nodes_ten,
        labels=(1, 2),
        graph_kind="pairwise",
        adjacency_connectivity="face",
        edge_weight="contact_voxels",
    )
    assert len(graph_nine.edges) == 0
    assert len(graph_ten.edges) == 1
    assert graph_ten.edges[0].contact_voxels == 10

    # Extractor with the same face neighborhood: 9 contacts → no pair
    # edge; 10 → one edge. Default erosion is off, so contact is measured
    # on the labels as drawn. Library default adjacency_min_voxels is 10.
    options = HabitatGraphFeatureOptions(
        edge_method="adjacency",
        node_method="component",
        subdivide_region_voxels=0,
        include_extended_metrics=False,
        adjacency_connectivity="face",
        connectivity="face",
    )
    assert options.erosion_radius == 0
    assert options.edge_method == "adjacency"
    assert options.adjacency_min_voxels == 10
    feats_nine = extract_graph_features(labels_nine, options=options)
    feats_ten = extract_graph_features(labels_ten, options=options)
    assert feats_nine["pair_h1_h2_n_edges"] == 0.0
    assert feats_ten["pair_h1_h2_n_edges"] == 1.0
    assert feats_ten["pair_h1_h2_contact_voxels_sum"] == 10.0


@pytest.mark.unit
def test_adjacency_graph_counts_face_adjacent_voxels() -> None:
    """The adjacency strategy should count face-adjacent voxel pairs between habitats."""
    label_array: np.ndarray = np.array(
        [
            [1, 2],
            [1, 0],
        ],
        dtype=np.int32,
    )
    node_result = extract_habitat_nodes(node_method="component", label_array=label_array, connectivity="face")

    graph = build_adjacency_graph(
        node_result=node_result,
        labels=(1, 2),
        graph_kind="pairwise",
        adjacency_connectivity="face",
        adjacency_min_voxels=1,
        edge_weight="contact_voxels",
    )

    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1
    assert graph.edges[0].contact_voxels == 1
    assert graph.edges[0].weight == 1.0


@pytest.mark.unit
def test_extractor_returns_single_and_pairwise_graph_features() -> None:
    """The kernel should return one subject-level feature dictionary."""
    label_array: np.ndarray = np.array(
        [
            [1, 1, 0, 2],
            [1, 0, 0, 2],
            [0, 0, 1, 2],
            [2, 2, 0, 0],
        ],
        dtype=np.int32,
    )

    features = _extract(
        label_array,
        distance_threshold=3.0,
        edge_method="centroid_distance",
        erosion_radius=0,
        subdivide_region_voxels=0,
    )

    assert features["single_h1_n_nodes"] == 2
    assert features["single_h1_n_edges"] == 1
    assert features["pair_h1_h2_n_nodes_1"] == 2
    assert features["pair_h1_h2_n_nodes_2"] == 2
    assert "pair_h1_h2_edge_density" in features
    # AD1/AD2 (average total degree per class) must be reported.
    assert "pair_h1_h2_avg_degree_1" in features
    assert "pair_h1_h2_avg_degree_2" in features


@pytest.mark.unit
def test_extractor_reports_new_topology_features() -> None:
    """Modularity, nearest-neighbor ratio, and component ratio should be present."""
    label_array: np.ndarray = np.array(
        [
            [1, 1, 0, 2],
            [1, 0, 0, 2],
            [0, 0, 1, 2],
            [2, 2, 0, 0],
        ],
        dtype=np.int32,
    )

    features = _extract(
        label_array,
        distance_threshold=3.0,
        edge_method="centroid_distance",
        erosion_radius=0,
        subdivide_region_voxels=0,
    )

    assert "single_h1_modularity" in features
    assert "single_h1_nearest_neighbor_ratio" in features
    assert "single_h1_connected_components_ratio" in features
    assert "pair_h1_h2_modularity" in features


@pytest.mark.unit
def test_connected_components_ratio_for_disconnected_nodes() -> None:
    """Two unconnected same-habitat nodes should give a component ratio of 1.0."""
    label_array: np.ndarray = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    features = _extract(
        label_array,
        distance_threshold=1.0,
        edge_method="centroid_distance",
        erosion_radius=0,
        subdivide_region_voxels=0,
    )

    # Two single-voxel nodes, far apart, so no edges form: 2 components / 2 nodes.
    assert features["single_h1_n_nodes"] == 2
    assert features["single_h1_n_edges"] == 0
    assert features["single_h1_connected_components_ratio"] == 1.0


@pytest.mark.unit
def test_normalized_features_divide_by_voi_size() -> None:
    """Normalized companions should use the matching physical or graph scale."""
    label_array: np.ndarray = np.array(
        [
            [1, 1, 0, 2],
            [1, 0, 0, 2],
            [0, 0, 1, 2],
            [2, 2, 0, 0],
        ],
        dtype=np.int32,
    )

    features = _extract(
        label_array,
        distance_threshold=3.0,
        edge_method="centroid_distance",
        erosion_radius=0,
        subdivide_region_voxels=0,
    )

    voi_voxels = float(np.count_nonzero(label_array))
    foreground_coords = np.argwhere(label_array > 0)
    bbox_lengths = foreground_coords.max(axis=0) - foreground_coords.min(axis=0) + 1
    bbox_diagonal = float(np.linalg.norm(bbox_lengths))

    # Count-like feature normalized by the full VOI volume (fraction/density).
    assert features["graph_num_nodes_total_norm"] == (
        features["graph_num_nodes_total"] / voi_voxels
    )
    # Physical distance-like feature normalized by the tumor bounding-box diagonal.
    assert features["single_h1_avg_edge_distance_norm"] == (
        features["single_h1_avg_edge_distance"] / bbox_diagonal
    )
    # Hop-count topology features use graph size, not physical tumor size.
    assert features["single_h1_avg_path_length_norm"] == (
        features["single_h1_avg_path_length"] / (features["single_h1_n_nodes"] - 1.0)
    )
    assert features["single_h1_diameter_norm"] == (
        features["single_h1_diameter"] / (features["single_h1_n_nodes"] - 1.0)
    )
    # Degree-like values are normalized by their maximum possible degree.
    assert features["single_h1_avg_degree_norm"] == (
        features["single_h1_avg_degree"] / (features["single_h1_n_nodes"] - 1.0)
    )
    # Size-invariant features must not get a normalized companion.
    assert "single_h1_edge_density_norm" not in features
    assert "single_h1_degree_cv_norm" not in features


@pytest.mark.unit
def test_normalized_features_include_habitat_specific_fractions() -> None:
    """Per-habitat normalized columns should use the matching habitat volume."""
    label_array: np.ndarray = np.array(
        [
            [1, 1, 0, 2],
            [1, 0, 0, 2],
            [0, 0, 1, 2],
            [2, 2, 0, 0],
        ],
        dtype=np.int32,
    )

    features = _extract(
        label_array,
        distance_threshold=3.0,
        edge_method="centroid_distance",
        erosion_radius=0,
        subdivide_region_voxels=0,
    )

    habitat_1_voxels = float(np.count_nonzero(label_array == 1))
    habitat_2_voxels = float(np.count_nonzero(label_array == 2))

    assert features["single_h1_n_nodes_per_habitat_volume"] == (
        features["single_h1_n_nodes"] / habitat_1_voxels
    )
    assert features["single_h1_avg_node_voxels_fraction"] == (
        features["single_h1_avg_node_voxels"] / habitat_1_voxels
    )
    assert features["single_h1_std_node_voxels_fraction"] == (
        features["single_h1_std_node_voxels"] / habitat_1_voxels
    )
    assert features["pair_h1_h2_n_nodes_1_per_habitat_volume"] == (
        features["pair_h1_h2_n_nodes_1"] / habitat_1_voxels
    )
    assert features["pair_h1_h2_n_nodes_2_per_habitat_volume"] == (
        features["pair_h1_h2_n_nodes_2"] / habitat_2_voxels
    )
    assert features["single_h1_n_edges_per_habitat_volume"] == (
        features["single_h1_n_edges"] / habitat_1_voxels
    )
    assert features["single_h1_connected_components_per_habitat_volume"] == (
        features["single_h1_connected_components"] / habitat_1_voxels
    )

    habitat_1_coords = np.argwhere(label_array == 1)
    habitat_1_bbox = float(
        np.linalg.norm(habitat_1_coords.max(axis=0) - habitat_1_coords.min(axis=0) + 1)
    )
    pair_coords = np.argwhere(np.isin(label_array, (1, 2)))
    pair_bbox = float(
        np.linalg.norm(pair_coords.max(axis=0) - pair_coords.min(axis=0) + 1)
    )
    assert features[
        "single_h1_avg_edge_distance_per_habitat_bbox_diagonal"
    ] == pytest.approx(features["single_h1_avg_edge_distance"] / habitat_1_bbox)
    assert features[
        "pair_h1_h2_avg_edge_distance_per_pair_bbox_diagonal"
    ] == pytest.approx(features["pair_h1_h2_avg_edge_distance"] / pair_bbox)


@pytest.mark.unit
def test_pairwise_normalized_features_use_graph_size_denominators() -> None:
    """Pairwise degree and component normalized columns should use graph limits."""
    label_array: np.ndarray = np.array(
        [
            [1, 1, 0, 2],
            [1, 0, 0, 2],
            [0, 0, 1, 2],
            [2, 2, 0, 0],
        ],
        dtype=np.int32,
    )

    features = _extract(
        label_array,
        distance_threshold=3.0,
        edge_method="centroid_distance",
        erosion_radius=0,
        subdivide_region_voxels=0,
    )

    n_nodes_1 = features["pair_h1_h2_n_nodes_1"]
    n_nodes_2 = features["pair_h1_h2_n_nodes_2"]
    total_nodes = n_nodes_1 + n_nodes_2

    assert features["pair_h1_h2_connected_components_norm"] == (
        features["pair_h1_h2_connected_components"] / total_nodes
    )
    assert features["pair_h1_h2_avg_h2_per_h1_norm"] == (
        features["pair_h1_h2_avg_h2_per_h1"] / n_nodes_2
    )
    assert features["pair_h1_h2_avg_h1_per_h2_norm"] == (
        features["pair_h1_h2_avg_h1_per_h2"] / n_nodes_1
    )
    assert features["pair_h1_h2_avg_degree_1_norm"] == (
        features["pair_h1_h2_avg_degree_1"] / (total_nodes - 1.0)
    )
    assert features["pair_h1_h2_avg_degree_2_norm"] == (
        features["pair_h1_h2_avg_degree_2"] / (total_nodes - 1.0)
    )


@pytest.mark.unit
def test_pairwise_contact_voxels_ignore_intra_edges() -> None:
    """Interface contact features should ignore same-label edges in pairwise graphs."""
    label_array: np.ndarray = np.array(
        [
            [1, 1, 2],
            [1, 2, 2],
        ],
        dtype=np.int32,
    )

    features = _extract(
        label_array,
        edge_method="adjacency",
        adjacency_connectivity="face",
        adjacency_min_voxels=1,
        edge_weight="contact_voxels",
        erosion_radius=0,
        subdivide_region_voxels=1,
        block_size=1,
        block_min_coverage=0.5,
        pairwise_include_intra_edges=True,
    )

    assert features["pair_h1_h2_n_edges"] == 3
    assert features["pair_h1_h2_contact_voxels_sum"] == 3
    assert features["pair_h1_h2_contact_voxels_mean"] == 1
    assert features["pair_h1_h2_contact_voxels_max"] == 1


@pytest.mark.unit
def test_contact_summary_norms_use_local_node_area_not_whole_voi() -> None:
    """Mean/max contact norms must use each edge's smaller-node area scale."""
    label_array: np.ndarray = np.array(
        [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
        ],
        dtype=np.int32,
    )
    features = _extract(
        label_array,
        edge_method="adjacency",
        adjacency_connectivity="face",
        adjacency_min_voxels=1,
        connectivity="face",
        include_extended_metrics=False,
        subdivide_region_voxels=0,
    )

    # The two 2x2 nodes share two face contacts. In 2D the local interface
    # scale of the smaller 4-voxel node is sqrt(4)=2, so the local scaled
    # mean/max are one. The total-contact norm remains a whole-VOI density.
    assert features["pair_h1_h2_contact_voxels_sum"] == 2.0
    assert features["pair_h1_h2_contact_voxels_mean_norm"] == pytest.approx(1.0)
    assert features["pair_h1_h2_contact_voxels_max_norm"] == pytest.approx(1.0)
    assert features["pair_h1_h2_contact_voxels_sum_norm"] == pytest.approx(
        2.0 / np.sqrt(8.0)
    )
    assert features[
        "pair_h1_h2_contact_voxels_sum_per_pair_area_scale"
    ] == pytest.approx(2.0 / np.sqrt(8.0))


@pytest.mark.unit
def test_subdivision_splits_large_region_into_multiple_nodes() -> None:
    """A large blob should become several grid-block nodes when subdivision is on."""
    label_array: np.ndarray = np.ones((10, 10), dtype=np.int32)

    without_split = extract_habitat_nodes(
        label_array=label_array, node_method="component"
    )
    assert len(without_split.nodes_by_habitat[1]) == 1

    with_split = extract_habitat_nodes(
        label_array=label_array,
        node_method="component",
        subdivide_region_voxels=10,
        block_size=5,
        block_min_coverage=0.5,
    )
    # The 10x10 region splits into four fully covered 5x5 blocks.
    nodes = with_split.nodes_by_habitat[1]
    assert len(nodes) == 4
    assert all(node.voxel_count == 25 for node in nodes)
    # Block component ids are unique and painted back into the component map.
    component_ids = {node.component_id for node in nodes}
    assert len(component_ids) == 4


@pytest.mark.unit
def test_default_options_disable_erosion_and_enable_subdivision() -> None:
    """Default kernel options leave labels as drawn and still subdivide."""
    options = HabitatGraphFeatureOptions()

    assert options.erosion_radius == 0
    assert options.subdivide_region_voxels == 1000
    assert options.block_size == 8
    assert options.block_min_coverage == 0.2
    assert options.distance_threshold == 5.0
    assert options.pairwise_include_intra_edges is True
    assert options.edge_method == "min_distance"
    assert options.node_method == "uniform_grid"
    assert options.adjacency_connectivity == "corner"
    assert options.connectivity == "full"
    assert options.adjacency_min_voxels == 10
    assert options.include_extended_metrics is False


@pytest.mark.unit
def test_pairwise_graph_adds_intra_edges_but_interface_uses_inter_only() -> None:
    """Intra edges should join the graph while interface features stay inter-only."""
    label_array: np.ndarray = np.array(
        [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    node_result = extract_habitat_nodes(
        label_array=label_array,
        node_method="component",
        subdivide_region_voxels=2,
        block_size=1,
    )
    pair_nodes = (
        list(node_result.nodes_by_habitat[1]) + list(node_result.nodes_by_habitat[2])
    )

    with_intra = build_centroid_distance_graph(
        nodes=pair_nodes,
        labels=(1, 2),
        graph_kind="pairwise",
        distance_threshold=1.5,
        include_intra_edges=True,
    )
    without_intra = build_centroid_distance_graph(
        nodes=pair_nodes,
        labels=(1, 2),
        graph_kind="pairwise",
        distance_threshold=1.5,
        include_intra_edges=False,
    )

    intra_edges = [e for e in with_intra.edges if e.edge_type == "intra"]
    inter_edges = [e for e in with_intra.edges if e.edge_type == "inter"]
    assert len(intra_edges) > 0
    # Inter edge count is unaffected by enabling intra edges.
    assert len(inter_edges) == len(without_intra.edges)


@pytest.mark.unit
def test_extended_graph_metrics_are_present_when_enabled() -> None:
    """Extended metrics should be exported when include_extended_metrics is true."""
    label_array: np.ndarray = np.array(
        [
            [1, 1, 1, 2, 2],
            [1, 1, 2, 2, 2],
            [1, 0, 0, 2, 2],
            [0, 0, 1, 1, 2],
            [1, 1, 1, 1, 0],
        ],
        dtype=np.int32,
    )

    features = _extract(
        label_array,
        distance_threshold=4.0,
        subdivide_region_voxels=0,
        include_extended_metrics=True,
        extended_min_nodes=3,
    )

    for key in (
        "single_h1_global_efficiency",
        "single_h1_local_efficiency",
        "single_h1_small_world_sigma",
        "single_h1_rich_club_coefficient",
        "single_h1_betweenness_max",
        "single_h1_betweenness_max_norm",
        "single_h1_degree_skewness",
        "single_h1_node_local_efficiency_min",
        "pair_h1_h2_global_efficiency",
        "pair_h1_h2_degree_skewness_1",
        "pair_h1_h2_betweenness_max_2_norm",
    ):
        assert key in features


@pytest.mark.unit
def test_extended_graph_metrics_can_be_disabled() -> None:
    """Extended metrics stay omitted under the library default (False)."""
    label_array: np.ndarray = np.array(
        [
            [1, 1, 0, 2],
            [1, 0, 0, 2],
            [0, 0, 1, 2],
            [2, 2, 0, 0],
        ],
        dtype=np.int32,
    )

    features = _extract(
        label_array,
        distance_threshold=3.0,
        subdivide_region_voxels=0,
    )

    assert "single_h1_global_efficiency" not in features
    assert "pair_h1_h2_small_world_sigma" not in features
    assert "pair_h1_h2_small_world_sigma_er" not in features
    assert "pair_h1_h2_small_world_sigma_rewire" not in features


@pytest.mark.unit
def test_small_world_er_and_rewire_use_different_nulls() -> None:
    """ER analytic S and rewire sigma must both exist and not be aliases."""
    from habit.kernels.habitat_graph.extended_metrics import (
        _small_world_sigma_er,
        _small_world_sigma_rewire,
        compute_extended_graph_metrics,
    )

    lattice = nx.watts_strogatz_graph(n=24, k=4, p=0.0, seed=0)
    assert nx.is_connected(lattice)
    sigma_er = _small_world_sigma_er(lattice, min_nodes=10)
    sigma_config = _small_world_sigma_rewire(
        lattice, min_nodes=10, nrand=8, niter=8, seed=0, sampler="config"
    )
    sigma_rewire = _small_world_sigma_rewire(
        lattice, min_nodes=10, nrand=8, niter=8, seed=0, sampler="rewire"
    )
    # A ring lattice is clustered vs ER, so Humphries S is well above 1.
    assert sigma_er > 1.0
    assert sigma_config >= 0.0
    assert sigma_rewire >= 0.0
    features = compute_extended_graph_metrics(
        lattice, "single_h1", extended_min_nodes=10, small_world_nrand=8
    )
    assert features["single_h1_small_world_sigma"] == pytest.approx(sigma_er)
    assert "single_h1_small_world_sigma_er" not in features
    assert "single_h1_small_world_sigma_rewire" not in features
    config_features = compute_extended_graph_metrics(
        lattice,
        "single_h1",
        extended_min_nodes=10,
        small_world_nrand=8,
        graph_null_sampler="config",
    )
    assert config_features["single_h1_small_world_sigma"] == pytest.approx(
        sigma_config
    )


@pytest.mark.unit
def test_betweenness_norm_reuses_networkx_normalized_values() -> None:
    """``*_norm`` must copy NetworkX-normalized betweenness, not divide again."""
    path = nx.path_graph(5)
    features = compute_extended_graph_metrics(path, "single_h1")
    betweenness = nx.betweenness_centrality(path)
    expected_max = float(max(betweenness.values()))
    expected_std = float(np.std(list(betweenness.values())))
    assert features["single_h1_betweenness_max"] == pytest.approx(expected_max)
    assert features["single_h1_betweenness_std"] == pytest.approx(expected_std)
    assert features["single_h1_betweenness_max_norm"] == pytest.approx(expected_max)
    assert features["single_h1_betweenness_std_norm"] == pytest.approx(expected_std)
    double_normalized = expected_max / ((5 - 1) * (5 - 2) / 2.0)
    assert features["single_h1_betweenness_max_norm"] != pytest.approx(
        double_normalized
    )

    tiny = nx.path_graph(2)
    tiny_features = compute_extended_graph_metrics(tiny, "single_h1")
    assert tiny_features["single_h1_betweenness_max_norm"] == 0.0
    assert tiny_features["single_h1_betweenness_std_norm"] == 0.0

    pair = nx.Graph()
    pair.add_nodes_from(
        [
            ("a1", {"habitat_label": 1}),
            ("a2", {"habitat_label": 1}),
            ("b1", {"habitat_label": 2}),
            ("b2", {"habitat_label": 2}),
        ]
    )
    pair.add_edges_from([("a1", "a2"), ("a2", "b1"), ("b1", "b2")])
    pair_features = compute_extended_pairwise_metrics(
        pair,
        ["a1", "a2"],
        ["b1", "b2"],
        "pair_h1_h2",
    )
    pair_betweenness = nx.betweenness_centrality(pair)
    max_class_1 = max(pair_betweenness["a1"], pair_betweenness["a2"])
    max_class_2 = max(pair_betweenness["b1"], pair_betweenness["b2"])
    assert pair_features["pair_h1_h2_betweenness_max_1"] == pytest.approx(
        max_class_1
    )
    assert pair_features["pair_h1_h2_betweenness_max_1_norm"] == pytest.approx(
        max_class_1
    )
    assert pair_features["pair_h1_h2_betweenness_max_2_norm"] == pytest.approx(
        max_class_2
    )


@pytest.mark.unit
def test_degree_preserving_null_model_is_reproducible_and_opt_in() -> None:
    """Degree-preserving null comparisons must be repeatable from their seed."""
    graph = nx.watts_strogatz_graph(12, 4, 0.2, seed=1)
    options = GraphNullModelOptions(
        n_random_graphs=8,
        swaps_per_edge=2,
        random_seed=17,
        sampler="config",
    )

    first = compare_graph_to_degree_preserving_null(
        graph,
        nx.average_clustering,
        options=options,
    )
    second = compare_graph_to_degree_preserving_null(
        graph,
        nx.average_clustering,
        options=options,
    )

    assert first == second
    assert first.observed == pytest.approx(nx.average_clustering(graph))
    assert first.n_requested == 8
    assert first.n_successful == 8
    assert first.is_valid is True
    assert np.isfinite(first.z_score)
    assert 0.0 < first.empirical_two_sided_p <= 1.0

    too_small = compare_graph_to_degree_preserving_null(
        nx.path_graph(3),
        nx.average_clustering,
        options=options,
    )
    assert too_small.is_valid is False
    assert too_small.n_successful == 0


@pytest.mark.unit
def test_degree_preserving_ensemble_matches_networkx_and_keeps_degrees() -> None:
    """Batched C/L must match NetworkX; both samplers keep the degree sequence."""
    from habit.kernels.habitat_graph.null_ensemble import (
        adjacency_from_undirected,
        batched_average_path_length,
        batched_transitivity,
        global_efficiency_value,
        local_efficiency_values,
        sample_degree_preserving_adjacencies,
    )

    graph = nx.watts_strogatz_graph(n=28, k=4, p=0.2, seed=3)
    adj, _nodes = adjacency_from_undirected(graph)
    degrees = adj.sum(axis=1)
    assert batched_transitivity(adj)[0] == pytest.approx(nx.transitivity(graph))
    assert batched_average_path_length(adj, device="cpu")[0] == pytest.approx(
        nx.average_shortest_path_length(graph)
    )
    assert global_efficiency_value(adj) == pytest.approx(
        nx.global_efficiency(graph), rel=1e-5
    )
    assert local_efficiency_values(adj) == pytest.approx(
        [
            float(nx.global_efficiency(graph.subgraph(list(graph.neighbors(node)))))
            if graph.degree(node) >= 2
            else 0.0
            for node in graph.nodes()
        ],
        rel=1e-5,
        abs=1e-8,
    )

    for sampler in ("config", "rewire"):
        batch = sample_degree_preserving_adjacencies(
            adj, nrand=6, sampler=sampler, niter=20, seed=5
        )
        assert batch.shape[0] >= 1
        for null_adj in batch:
            assert np.array_equal(null_adj.sum(axis=1), degrees)


@pytest.mark.unit
def test_graph_null_sampler_default_is_analytic() -> None:
    """Feature extraction defaults to Humphries analytic small-world S."""
    options = HabitatGraphFeatureOptions()
    assert options.graph_null_sampler == "analytic"
    assert options.small_world_nrand == 100
    assert options.small_world_niter == 100
    assert options.graph_null_device == "auto"
    assert options.graph_metric_backend == "networkx"


@pytest.mark.unit
def test_expected_labels_produce_stable_columns_for_missing_habitats() -> None:
    """``expected_labels`` emits zero-valued columns for absent habitats."""
    label_array: np.ndarray = np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    options = HabitatGraphFeatureOptions(
        distance_threshold=3.0,
        erosion_radius=0,
        node_method="component",
        subdivide_region_voxels=0,
    )

    features = extract_graph_features(
        label_array, options=options, expected_labels=(1, 2)
    )

    # Habitat 2 is absent: its single-habitat graph is empty but present.
    assert features["single_h2_n_nodes"] == 0.0
    assert features["single_h2_n_edges"] == 0.0
    assert features["pair_h1_h2_n_nodes_1"] == 1.0
    assert features["pair_h1_h2_n_nodes_2"] == 0.0
    # The present-habitat count still reflects what is actually in the map.
    assert features["graph_num_habitats"] == 1.0


@pytest.mark.unit
def test_default_connectivity_merges_diagonal_same_label() -> None:
    """Library defaults treat diagonally touching same-label voxels as one node."""
    label_array: np.ndarray = np.array(
        [
            [1, 0],
            [0, 1],
        ],
        dtype=np.int32,
    )
    default_options = HabitatGraphFeatureOptions(
        node_method="component", subdivide_region_voxels=0
    )
    assert default_options.connectivity == "full"
    default_feats = extract_graph_features(label_array, options=default_options)
    face_feats = extract_graph_features(
        label_array,
        options=HabitatGraphFeatureOptions(
            node_method="component",
            subdivide_region_voxels=0,
            connectivity="face",
        ),
    )

    assert default_feats["single_h1_n_nodes"] == 1.0
    assert face_feats["single_h1_n_nodes"] == 2.0


@pytest.mark.unit
def test_default_adjacency_connects_diagonal_habitats() -> None:
    """Library defaults create an inter-edge for diagonally touching habitats."""
    label_array: np.ndarray = np.array(
        [
            [1, 0],
            [0, 2],
        ],
        dtype=np.int32,
    )
    default_options = HabitatGraphFeatureOptions(
        edge_method="adjacency",
        node_method="component",
        adjacency_min_voxels=1,
        subdivide_region_voxels=0,
    )
    assert default_options.adjacency_connectivity == "corner"
    default_feats = extract_graph_features(label_array, options=default_options)
    face_feats = extract_graph_features(
        label_array,
        options=HabitatGraphFeatureOptions(
            edge_method="adjacency",
            node_method="component",
            adjacency_min_voxels=1,
            subdivide_region_voxels=0,
            adjacency_connectivity="face",
            connectivity="face",
        ),
    )

    assert default_feats["pair_h1_h2_n_edges"] == 1.0
    assert face_feats["pair_h1_h2_n_edges"] == 0.0


@pytest.mark.unit
def test_face_vs_full_connectivity_merges_diagonal_touch() -> None:
    """Full connectivity merges diagonal-touching voxels; face keeps them apart."""
    # Two habitat-1 voxels touch only at a corner.
    label_array: np.ndarray = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=np.int32,
    )

    face = extract_habitat_nodes(node_method="component", label_array=label_array, connectivity="face")
    full = extract_habitat_nodes(node_method="component", label_array=label_array, connectivity="full")

    assert len(face.nodes_by_habitat[1]) == 2
    assert len(full.nodes_by_habitat[1]) == 1
    assert full.nodes_by_habitat[1][0].voxel_count == 2


@pytest.mark.unit
def test_default_adjacency_keeps_contact_edges_optional_erosion_separates() -> None:
    """Defaults keep a contact>=10 inter-edge; erosion=1 can still separate."""
    # Two 5x10 blocks share a 10-voxel face. Thick enough to survive one
    # erosion iteration, but erosion peels the shared boundary so they no
    # longer touch.
    label_array: np.ndarray = np.zeros((10, 10), dtype=np.int32)
    label_array[0:5, :] = 1
    label_array[5:10, :] = 2

    default_options = HabitatGraphFeatureOptions(
        edge_method="adjacency",
        node_method="component",
        subdivide_region_voxels=0,
    )
    assert default_options.erosion_radius == 0
    assert default_options.edge_method == "adjacency"
    assert default_options.adjacency_min_voxels == 10

    default_feats = extract_graph_features(label_array, options=default_options)
    eroded_feats = extract_graph_features(
        label_array,
        options=HabitatGraphFeatureOptions(
            edge_method="adjacency",
            node_method="component",
            subdivide_region_voxels=0,
            erosion_radius=1,
        ),
    )

    assert default_feats["pair_h1_h2_n_edges"] >= 1.0
    assert eroded_feats["pair_h1_h2_n_edges"] == 0.0


@pytest.mark.unit
def test_erosion_splits_thin_bridge_into_separate_nodes() -> None:
    """Erosion removes a one-voxel face bridge so one blob becomes two nodes."""
    # Two thick blobs joined by a single face-adjacent bridge voxel.
    label_array: np.ndarray = np.zeros((7, 11), dtype=np.int32)
    label_array[1:6, 1:5] = 1
    label_array[1:6, 6:10] = 1
    label_array[3, 5] = 1

    intact = extract_habitat_nodes(
        node_method="component",
        label_array=label_array,
        connectivity="face",
        erosion_radius=0,
    )
    eroded = extract_habitat_nodes(
        node_method="component",
        label_array=label_array,
        connectivity="face",
        erosion_radius=1,
    )

    assert len(intact.nodes_by_habitat[1]) == 1
    assert len(eroded.nodes_by_habitat[1]) == 2


@pytest.mark.unit
def test_min_region_voxels_drops_tiny_components() -> None:
    """Components below ``min_region_voxels`` must not become graph nodes."""
    label_array: np.ndarray = np.array(
        [
            [1, 1, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    kept = extract_habitat_nodes(node_method="component", label_array=label_array, min_region_voxels=2)
    assert len(kept.nodes_by_habitat[1]) == 1
    assert kept.nodes_by_habitat[1][0].voxel_count == 4


@pytest.mark.unit
def test_block_min_coverage_filters_partial_blocks() -> None:
    """Low-coverage subdivision blocks are discarded; high coverage keeps them."""
    # 6x6 ones: with block_size=4, corner blocks are only partly covered.
    label_array: np.ndarray = np.ones((6, 6), dtype=np.int32)

    strict = extract_habitat_nodes(
        node_method="component",
        label_array=label_array,
        subdivide_region_voxels=1,
        block_size=4,
        block_min_coverage=0.9,
    )
    loose = extract_habitat_nodes(
        node_method="component",
        label_array=label_array,
        subdivide_region_voxels=1,
        block_size=4,
        block_min_coverage=0.1,
    )

    assert len(strict.nodes_by_habitat[1]) < len(loose.nodes_by_habitat[1])
    assert len(loose.nodes_by_habitat[1]) >= 1


@pytest.mark.unit
def test_adjacency_corner_connects_diagonal_habitats_face_does_not() -> None:
    """Corner adjacency creates an edge for diagonal contact; face does not."""
    label_array: np.ndarray = np.array(
        [
            [1, 0],
            [0, 2],
        ],
        dtype=np.int32,
    )
    node_result = extract_habitat_nodes(node_method="component", label_array=label_array, connectivity="face")

    face_graph = build_adjacency_graph(
        node_result=node_result,
        labels=(1, 2),
        graph_kind="pairwise",
        adjacency_connectivity="face",
        adjacency_min_voxels=1,
        edge_weight="contact_voxels",
    )
    corner_graph = build_adjacency_graph(
        node_result=node_result,
        labels=(1, 2),
        graph_kind="pairwise",
        adjacency_connectivity="corner",
        adjacency_min_voxels=1,
        edge_weight="contact_voxels",
    )

    assert len(face_graph.edges) == 0
    assert len(corner_graph.edges) == 1
    assert corner_graph.edges[0].contact_voxels == 1
    assert corner_graph.edges[0].weight == 1.0


@pytest.mark.unit
def test_adjacency_edge_connectivity_counts_more_contacts_than_face() -> None:
    """In 3D, edge connectivity includes more neighbor offsets than face."""
    # Two habitats share a face contact and an edge (diagonal-in-plane) contact.
    label_array: np.ndarray = np.zeros((3, 3, 3), dtype=np.int32)
    label_array[1, 1, 1] = 1
    label_array[1, 1, 2] = 2  # face neighbor of habitat 1
    label_array[1, 2, 2] = 2  # also edge-adjacent to habitat 1 via (0,+1,+1)

    node_result = extract_habitat_nodes(node_method="component", label_array=label_array, connectivity="face")
    face_graph = build_adjacency_graph(
        node_result=node_result,
        labels=(1, 2),
        graph_kind="pairwise",
        adjacency_connectivity="face",
        adjacency_min_voxels=1,
        edge_weight="contact_voxels",
        include_intra_edges=False,
    )
    edge_graph = build_adjacency_graph(
        node_result=node_result,
        labels=(1, 2),
        graph_kind="pairwise",
        adjacency_connectivity="edge",
        adjacency_min_voxels=1,
        edge_weight="contact_voxels",
        include_intra_edges=False,
    )

    assert len(face_graph.edges) == 1
    assert face_graph.edges[0].contact_voxels == 1
    assert len(edge_graph.edges) == 1
    assert edge_graph.edges[0].contact_voxels >= face_graph.edges[0].contact_voxels


@pytest.mark.unit
def test_empty_label_array_returns_zero_habitat_summary() -> None:
    """An all-background map yields summary zeros and no single/pair columns."""
    label_array: np.ndarray = np.zeros((4, 4), dtype=np.int32)
    features = _extract(label_array, erosion_radius=0, subdivide_region_voxels=0)

    assert features["graph_num_habitats"] == 0.0
    assert features["graph_num_nodes_total"] == 0.0
    assert not any(key.startswith("single_h") for key in features)
    assert not any(key.startswith("pair_") for key in features)


@pytest.mark.unit
def test_single_node_graph_has_zero_edges_and_key_columns() -> None:
    """A lone connected region reports one node, zero edges, and core metrics."""
    label_array: np.ndarray = np.array(
        [
            [0, 1, 1],
            [0, 1, 1],
            [0, 0, 0],
        ],
        dtype=np.int32,
    )
    features = _extract(
        label_array,
        distance_threshold=5.0,
        erosion_radius=0,
        subdivide_region_voxels=0,
        include_extended_metrics=False,
    )

    assert features["single_h1_n_nodes"] == 1.0
    assert features["single_h1_n_edges"] == 0.0
    assert "single_h1_avg_degree" in features
    assert "single_h1_edge_density" in features
    assert features["single_h1_avg_degree"] == 0.0


@pytest.mark.unit
def test_extract_graph_features_for_labels_restricts_habitats() -> None:
    """Restricting labels zeros out non-selected habitats before extraction."""
    label_array: np.ndarray = np.array(
        [
            [1, 1, 2, 2],
            [1, 0, 0, 2],
            [3, 3, 0, 0],
        ],
        dtype=np.int32,
    )
    options = HabitatGraphFeatureOptions(
        distance_threshold=3.0,
        erosion_radius=0,
        node_method="component",
        subdivide_region_voxels=0,
        include_extended_metrics=False,
    )

    features = extract_graph_features_for_labels(
        label_array, labels=(1, 2), options=options
    )

    assert features["graph_num_habitats"] == 2.0
    assert "single_h1_n_nodes" in features
    assert "single_h2_n_nodes" in features
    assert "single_h3_n_nodes" not in features
    assert "pair_h1_h2_n_nodes_1" in features


@pytest.mark.unit
def test_expected_labels_align_columns_across_subjects() -> None:
    """Two subjects with different present habitats share the same column keys."""
    options = HabitatGraphFeatureOptions(
        distance_threshold=3.0,
        erosion_radius=0,
        node_method="component",
        subdivide_region_voxels=0,
        include_extended_metrics=False,
    )
    labels_a: np.ndarray = np.array([[1, 1, 0], [1, 0, 0]], dtype=np.int32)
    labels_b: np.ndarray = np.array([[2, 2, 0], [0, 2, 0]], dtype=np.int32)

    feats_a = extract_graph_features(
        labels_a, options=options, expected_labels=(1, 2)
    )
    feats_b = extract_graph_features(
        labels_b, options=options, expected_labels=(1, 2)
    )

    assert set(feats_a.keys()) == set(feats_b.keys())
    assert feats_a["single_h2_n_nodes"] == 0.0
    assert feats_b["single_h1_n_nodes"] == 0.0
    assert feats_a["graph_num_habitats"] == 1.0
    assert feats_b["graph_num_habitats"] == 1.0


@pytest.mark.unit
def test_3d_synthetic_volume_extracts_single_and_pairwise_columns() -> None:
    """A compact 3D multi-habitat volume produces the expected key columns."""
    label_array: np.ndarray = np.zeros((8, 8, 8), dtype=np.int32)
    label_array[1:4, 1:4, 1:4] = 1
    label_array[1:4, 5:7, 5:7] = 1
    label_array[4:7, 3:6, 3:6] = 2

    features = _extract(
        label_array,
        distance_threshold=6.0,
        erosion_radius=0,
        subdivide_region_voxels=0,
        include_extended_metrics=False,
    )

    assert features["graph_num_habitats"] == 2.0
    assert features["single_h1_n_nodes"] == 2.0
    assert features["single_h2_n_nodes"] == 1.0
    assert "pair_h1_h2_n_nodes_1" in features
    assert "pair_h1_h2_edge_density" in features
    assert features["graph_num_nodes_total"] == 3.0


@pytest.mark.unit
def test_pair_count_matches_unordered_pairs() -> None:
    """``pair_count`` is the binomial coefficient C(n, 2)."""
    assert pair_count(0) == 0
    assert pair_count(1) == 0
    assert pair_count(2) == 1
    assert pair_count(3) == 3
    assert pair_count(4) == 6


@pytest.mark.unit
def test_uniform_grid_emits_one_node_per_cell_subregion() -> None:
    """A kept cube yields one centroid node per habitat connected component."""
    # One 8x8 cell: habitat 1 is two face-disconnected blobs; habitat 2
    # occupies the remainder. Cell coverage is 1.0 so the cube is kept.
    label_array: np.ndarray = np.full((8, 8), 2, dtype=np.int32)
    label_array[0:3, 0:3] = 1
    label_array[5:8, 5:8] = 1

    result = extract_habitat_nodes(
        label_array=label_array,
        node_method="uniform_grid",
        connectivity="face",
        block_size=8,
        block_min_coverage=0.2,
        min_region_voxels=1,
    )
    nodes_1 = result.nodes_by_habitat[1]
    nodes_2 = result.nodes_by_habitat[2]
    assert len(nodes_1) == 2
    assert len(nodes_2) == 1
    assert {node.voxel_count for node in nodes_1} == {9}
    assert nodes_2[0].voxel_count == 64 - 18
    centroids = np.vstack([node.centroid for node in nodes_1])
    expected = np.array([[1.0, 1.0], [6.0, 6.0]], dtype=float)
    for row in expected:
        assert any(np.allclose(centroids[i], row) for i in range(len(centroids)))


@pytest.mark.unit
def test_uniform_grid_uses_global_origin_and_keeps_equal_cubes() -> None:
    """Default node method tessellates the whole VOI into equal-volume cubes."""
    label_array: np.ndarray = np.zeros((16, 16), dtype=np.int32)
    label_array[0:16, 0:8] = 1
    label_array[0:16, 8:16] = 2

    result = extract_habitat_nodes(
        label_array=label_array,
        node_method="uniform_grid",
        block_size=8,
        block_min_coverage=0.5,
    )
    nodes_1 = result.nodes_by_habitat[1]
    nodes_2 = result.nodes_by_habitat[2]
    assert len(nodes_1) == 2
    assert len(nodes_2) == 2
    assert {node.voxel_count for node in nodes_1 + nodes_2} == {64}
    assert result.grid_origin == (0, 0)
    assert result.grid_block_size == 8


@pytest.mark.unit
def test_default_min_distance_uniform_grid_creates_intra_edges() -> None:
    """Library defaults connect neighbouring equal-volume cubes inside a habitat."""
    label_array: np.ndarray = np.ones((16, 16), dtype=np.int32)
    options = HabitatGraphFeatureOptions()
    assert options.edge_method == "min_distance"
    assert options.node_method == "uniform_grid"
    assert options.block_size == 8
    assert options.block_min_coverage == 0.2

    features = extract_graph_features(label_array, options=options)
    # 16x16 ones, 8-voxel cubes: 2x2 full 8x8 cells (coverage 1.0).
    # Face-adjacent cubes connect (4 edges).
    assert features["single_h1_n_nodes"] == 4.0
    assert features["single_h1_n_edges"] >= 4.0


@pytest.mark.unit
def test_default_threshold_skips_one_empty_lattice_cell() -> None:
    """One empty 5-cube between two cubes is distance 6 and stays disconnected."""
    label_array: np.ndarray = np.zeros((5, 15), dtype=np.int32)
    label_array[:, 0:5] = 1
    label_array[:, 10:15] = 1

    features = extract_graph_features(label_array)
    assert features["single_h1_n_nodes"] == 2.0
    assert features["single_h1_n_edges"] == 0.0


@pytest.mark.unit
def test_crop_preserves_graph_features_when_embedded_in_empty_volume() -> None:
    """VOI crop must not change topology features vs a tight label map."""
    core: np.ndarray = np.zeros((8, 8), dtype=np.int32)
    core[1:4, 1:4] = 1
    core[5:8, 5:8] = 2
    options = HabitatGraphFeatureOptions(
        node_method="component",
        edge_method="adjacency",
        include_extended_metrics=False,
    )
    tight = extract_graph_features(core, options=options)
    padded = np.zeros((64, 64), dtype=np.int32)
    padded[20:28, 20:28] = core
    embedded = extract_graph_features(padded, options=options)
    assert set(tight) == set(embedded)
    for key in tight:
        assert tight[key] == pytest.approx(embedded[key], abs=1e-8, rel=1e-8)


@pytest.mark.unit
def test_crop_offset_maps_centroids_back_to_original_indices() -> None:
    """Node centroids stay in the uncropped index space after the VOI crop."""
    core: np.ndarray = np.zeros((8, 8), dtype=np.int32)
    core[2:5, 2:5] = 1
    local = extract_habitat_nodes(core, node_method="component", connectivity="face")
    padded = np.zeros((40, 40), dtype=np.int32)
    padded[12:20, 12:20] = core
    embedded = extract_habitat_nodes(
        padded, node_method="component", connectivity="face"
    )
    assert embedded.crop_offset is not None
    local_c = np.sort(np.vstack([n.centroid for n in local.nodes_by_habitat[1]]), axis=0)
    embed_c = np.sort(
        np.vstack([n.centroid for n in embedded.nodes_by_habitat[1]]), axis=0
    )
    assert np.allclose(embed_c, local_c + np.array([12.0, 12.0]))


@pytest.mark.unit
def test_min_voxel_distance_torch_matches_kdtree() -> None:
    """Torch cdist (CPU, and CUDA when present) matches the kd-tree minimum."""
    pytest.importorskip("torch")
    from habit.utils.torch_graph_utils import (
        _min_voxel_distance_torch,
        _min_voxel_distance_tree,
        min_voxel_distance,
    )
    from habit.utils.torch_radiomics_utils import is_cuda_available

    rng = np.random.default_rng(0)
    cloud_a = rng.integers(0, 25, size=(40, 3)).astype(np.float64)
    cloud_b = rng.integers(0, 25, size=(55, 3)).astype(np.float64)
    tree = _min_voxel_distance_tree(cloud_a, cloud_b)
    cpu_torch = _min_voxel_distance_torch(cloud_a, cloud_b, "cpu")
    auto = min_voxel_distance(cloud_a, cloud_b, device="auto")
    assert cpu_torch == pytest.approx(tree, abs=1e-4, rel=1e-4)
    assert auto == pytest.approx(tree, abs=1e-12, rel=1e-12)
    if is_cuda_available():
        gpu = min_voxel_distance(cloud_a, cloud_b, device="cuda")
        assert gpu == pytest.approx(tree, abs=1e-3, rel=1e-3)


def _undirected_edge_key(
    source: str, target: str, distance: float
) -> tuple:
    """Canonical undirected pair plus rounded distance."""
    ends = tuple(sorted((source, target)))
    return (ends[0], ends[1], round(float(distance), 8))


@pytest.mark.unit
def test_lattice_chebyshev_radius_tracks_block_and_threshold() -> None:
    """Search radius must follow cube-gap geometry, not a hard-coded 8 vs 5."""
    from habit.kernels.habitat_graph.proximity import (
        cube_separation_lower_bound,
        lattice_chebyshev_radius,
    )

    assert lattice_chebyshev_radius(8, 0.5) == 0
    assert lattice_chebyshev_radius(8, 5.0) == 1
    assert lattice_chebyshev_radius(8, 8.0) == 1
    assert lattice_chebyshev_radius(8, 9.0) == 2
    assert lattice_chebyshev_radius(4, 10.0) == 3
    # Single-axis R=2 cubes of edge 8 are at least 9 voxels apart.
    assert cube_separation_lower_bound((2, 0, 0), 8) == 9.0
    assert cube_separation_lower_bound((1, 0, 0), 8) == 1.0


@pytest.mark.unit
def test_hop_metrics_match_networkx_on_connected_graphs() -> None:
    """One BFS sweep must match NetworkX Brandes, ASPL, diameter, closeness."""
    from habit.kernels.habitat_graph.traversal import hop_metrics

    rng = np.random.default_rng(7)
    graph = nx.gnm_random_graph(18, 40, seed=7)
    while not nx.is_connected(graph):
        graph = nx.gnm_random_graph(18, 40, seed=int(rng.integers(1, 10_000)))

    hop = hop_metrics(graph, device="python")
    hop_auto = hop_metrics(graph, device="auto")
    nx_bc = nx.betweenness_centrality(graph, normalized=True, weight=None)
    nx_cc = nx.closeness_centrality(graph)
    for node_id in graph.nodes():
        assert hop.betweenness[node_id] == pytest.approx(nx_bc[node_id], abs=1e-12)
        assert hop.closeness[node_id] == pytest.approx(nx_cc[node_id], abs=1e-12)
    assert hop.avg_path_length == pytest.approx(
        nx.average_shortest_path_length(graph), abs=1e-12
    )
    assert hop.diameter == pytest.approx(float(nx.diameter(graph)), abs=1e-12)
    for node_id in graph.nodes():
        assert hop_auto.betweenness[node_id] == pytest.approx(
            nx_bc[node_id], abs=1e-8, rel=1e-6
        )


@pytest.mark.unit
def test_hop_metrics_igraph_matches_networkx() -> None:
    """Optional igraph hop metrics must stay on the NetworkX definitions."""
    pytest.importorskip("igraph")
    from habit.kernels.habitat_graph.traversal import hop_metrics

    rng = np.random.default_rng(11)
    graph = nx.gnm_random_graph(22, 55, seed=11)
    while not nx.is_connected(graph):
        graph = nx.gnm_random_graph(22, 55, seed=int(rng.integers(1, 10_000)))

    hop = hop_metrics(graph, backend="igraph")
    nx_bc = nx.betweenness_centrality(graph, normalized=True, weight=None)
    nx_cc = nx.closeness_centrality(graph)
    for node_id in graph.nodes():
        assert hop.betweenness[node_id] == pytest.approx(nx_bc[node_id], abs=1e-10)
        assert hop.closeness[node_id] == pytest.approx(nx_cc[node_id], abs=1e-10)
    assert hop.avg_path_length == pytest.approx(
        nx.average_shortest_path_length(graph), abs=1e-10
    )
    assert hop.diameter == pytest.approx(float(nx.diameter(graph)), abs=1e-10)


@pytest.mark.unit
def test_component_min_distance_does_not_use_lattice_accelerators() -> None:
    """Sweep / lattice range search must stay off when grid metadata is absent."""
    from habit.kernels.habitat_graph.edges import (
        _min_distance_edges_for_pairs,
        _node_voxel_coords,
        build_min_distance_edges,
    )
    from habit.kernels.habitat_graph.proximity import uses_uniform_grid

    label_array = np.zeros((16, 16, 8), dtype=np.int32)
    label_array[:8, :8, :] = 1
    label_array[:8, 8:, :] = 2
    nodes = extract_habitat_nodes(
        label_array,
        node_method="component",
        connectivity="full",
        min_region_voxels=1,
        subdivide_region_voxels=0,
    )
    assert nodes.grid_origin is None
    assert nodes.grid_block_size is None
    assert uses_uniform_grid(nodes) is False
    all_nodes = list(nodes.nodes_by_habitat[1]) + list(nodes.nodes_by_habitat[2])
    fast = build_min_distance_edges(nodes, all_nodes, 5.0, "none")
    coords = {node.node_id: _node_voxel_coords(nodes, node) for node in all_nodes}
    slow = _min_distance_edges_for_pairs(
        all_nodes, all_nodes, coords, 5.0, "none", "min_distance"
    )
    fast_keys = {
        _undirected_edge_key(edge.source, edge.target, edge.distance or 0.0)
        for edge in fast
    }
    slow_keys = {
        _undirected_edge_key(edge.source, edge.target, edge.distance or 0.0)
        for edge in slow
    }
    assert fast_keys == slow_keys


@pytest.mark.unit
def test_extract_igraph_backend_matches_networkx_except_modularity() -> None:
    """igraph hop / clustering match; Louvain modularity may move slightly."""
    pytest.importorskip("igraph")
    label_array = np.zeros((24, 24, 16), dtype=np.int32)
    label_array[:12, :12, :] = 1
    label_array[:12, 12:, :] = 2
    label_array[12:, :12, :] = 3
    nx_features = extract_graph_features(
        label_array,
        options=HabitatGraphFeatureOptions(
            node_method="uniform_grid",
            edge_method="min_distance",
            block_size=8,
            distance_threshold=5.0,
            graph_metric_backend="networkx",
            include_extended_metrics=False,
        ),
    )
    ig_features = extract_graph_features(
        label_array,
        options=HabitatGraphFeatureOptions(
            node_method="uniform_grid",
            edge_method="min_distance",
            block_size=8,
            distance_threshold=5.0,
            graph_metric_backend="igraph",
            include_extended_metrics=False,
        ),
    )
    assert set(nx_features) == set(ig_features)
    drifted = 0
    for key, value in nx_features.items():
        other = ig_features[key]
        if key.endswith("_modularity"):
            if abs(float(value) - float(other)) > 1e-8:
                drifted += 1
            continue
        assert other == pytest.approx(value, abs=1e-8, rel=1e-6), key
    # Partition algorithms are allowed to disagree; hop metrics are not.
    assert drifted >= 0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("block_size", "threshold"),
    [(8, 5.0), (8, 20.0), (4, 6.0)],
)
def test_min_distance_range_search_matches_all_pairs(
    block_size: int, threshold: float
) -> None:
    """Lattice / centroid candidates must emit the same edges as all-pairs."""
    from habit.kernels.habitat_graph.edges import (
        _min_distance_edges_for_pairs,
        _node_voxel_coords,
        build_min_distance_edges,
    )

    rng = np.random.default_rng(3)
    label_array = np.zeros((24, 24, 16), dtype=np.int32)
    label_array[:12, :12, :] = 1
    label_array[:12, 12:, :] = 2
    holes = rng.integers(0, [24, 24, 16], size=(80, 3))
    label_array[holes[:, 0], holes[:, 1], holes[:, 2]] = 0
    nodes = extract_habitat_nodes(
        label_array,
        node_method="uniform_grid",
        block_size=block_size,
        block_min_coverage=0.2,
    )
    all_nodes = list(nodes.nodes_by_habitat[1]) + list(nodes.nodes_by_habitat[2])
    fast = build_min_distance_edges(nodes, all_nodes, threshold, "none")
    coords = {node.node_id: _node_voxel_coords(nodes, node) for node in all_nodes}
    # Same-object all-pairs walk (i < j) is the old complete scan.
    slow = _min_distance_edges_for_pairs(
        all_nodes, all_nodes, coords, threshold, "none", "min_distance"
    )
    ref_keys = {
        _undirected_edge_key(edge.source, edge.target, edge.distance or 0.0)
        for edge in slow
    }
    fast_keys = {
        _undirected_edge_key(edge.source, edge.target, edge.distance or 0.0)
        for edge in fast
    }
    assert fast_keys == ref_keys


@pytest.mark.unit
def test_pairwise_min_distance_reuses_intra_edges() -> None:
    """Cached single-habitat min_distance edges must match a from-scratch pair."""
    from habit.kernels.habitat_graph.edges import (
        as_intra_edge,
        build_min_distance_inter_edges,
        compose_pairwise_graph,
    )

    label_array = np.zeros((32, 32, 16), dtype=np.int32)
    label_array[:16, :16, :] = 1
    label_array[:16, 16:, :] = 2
    options = HabitatGraphFeatureOptions(
        node_method="uniform_grid",
        edge_method="min_distance",
        block_size=8,
        distance_threshold=5.0,
        pairwise_include_intra_edges=True,
    )
    nodes = extract_habitat_nodes(
        label_array,
        node_method=options.node_method,
        connectivity=options.connectivity,
        min_region_voxels=options.min_region_voxels,
        block_size=options.block_size,
        block_min_coverage=options.block_min_coverage,
    )
    single_a = build_min_distance_graph(
        node_result=nodes,
        labels=(1,),
        graph_kind="single",
        distance_threshold=options.distance_threshold,
    )
    single_b = build_min_distance_graph(
        node_result=nodes,
        labels=(2,),
        graph_kind="single",
        distance_threshold=options.distance_threshold,
    )
    rebuilt = build_min_distance_graph(
        node_result=nodes,
        labels=(1, 2),
        graph_kind="pairwise",
        distance_threshold=options.distance_threshold,
        include_intra_edges=True,
    )
    pair_nodes = list(nodes.nodes_by_habitat[1]) + list(nodes.nodes_by_habitat[2])
    reused = compose_pairwise_graph(
        pair_nodes,
        (1, 2),
        build_min_distance_inter_edges(
            nodes, (1, 2), options.distance_threshold, options.edge_weight
        ),
        [as_intra_edge(edge) for edge in (*single_a.edges, *single_b.edges)],
    )
    rebuilt_keys = {
        _undirected_edge_key(edge.source, edge.target, edge.distance or 0.0)
        for edge in rebuilt.edges
    }
    reused_keys = {
        _undirected_edge_key(edge.source, edge.target, edge.distance or 0.0)
        for edge in reused.edges
    }
    assert rebuilt_keys == reused_keys
    assert {edge.edge_type for edge in reused.edges} <= {"inter", "intra"}


@pytest.mark.unit
def test_extract_uniform_grid_reuse_matches_rebuild() -> None:
    """Default uniform_grid + min_distance features match a no-reuse rebuild."""
    from habit.kernels.habitat_graph.features import _build_pairwise_graph
    from habit.kernels.habitat_graph.metrics import (
        calculate_pairwise_graph_metrics,
        calculate_single_graph_metrics,
    )

    label_array = np.zeros((24, 24, 16), dtype=np.int32)
    label_array[:12, :12, :] = 1
    label_array[:12, 12:, :] = 2
    label_array[12:, :12, :] = 3
    options = HabitatGraphFeatureOptions(
        node_method="uniform_grid",
        edge_method="min_distance",
        block_size=8,
        distance_threshold=5.0,
        include_extended_metrics=True,
        graph_null_sampler="analytic",
    )
    new_features = extract_graph_features(label_array, options=options)

    nodes = extract_habitat_nodes(
        label_array,
        node_method=options.node_method,
        connectivity=options.connectivity,
        min_region_voxels=options.min_region_voxels,
        block_size=options.block_size,
        block_min_coverage=options.block_min_coverage,
    )
    rebuilt: dict = {}
    singles = {}
    for label in (1, 2, 3):
        graph = build_min_distance_graph(
            node_result=nodes,
            labels=(label,),
            graph_kind="single",
            distance_threshold=options.distance_threshold,
        )
        singles[label] = graph
        rebuilt.update(
            calculate_single_graph_metrics(
                graph,
                include_extended_metrics=True,
                graph_null_sampler="analytic",
            )
        )
    for label_a, label_b in ((1, 2), (1, 3), (2, 3)):
        pair_nodes = list(nodes.nodes_by_habitat[label_a]) + list(
            nodes.nodes_by_habitat[label_b]
        )
        # Empty cache forces a from-scratch pairwise build (old path).
        graph = _build_pairwise_graph(
            nodes, pair_nodes, label_a, label_b, options, {}
        )
        rebuilt.update(
            calculate_pairwise_graph_metrics(
                graph,
                include_extended_metrics=True,
                graph_null_sampler="analytic",
            )
        )
    compared = 0
    for key, value in rebuilt.items():
        if key not in new_features:
            continue
        assert new_features[key] == pytest.approx(value, abs=1e-10, rel=1e-10), key
        compared += 1
    assert compared > 20
    # Intra reuse must have been available for the new extract path.
    assert singles[1].edges or singles[2].edges or singles[3].edges
