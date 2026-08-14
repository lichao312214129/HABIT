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

from habit.kernels.habitat_graph import (
    HabitatGraphFeatureOptions,
    build_adjacency_graph,
    build_centroid_distance_graph,
    build_min_distance_graph,
    extract_graph_features,
    extract_graph_features_for_labels,
    extract_habitat_nodes,
    pair_count,
)


def _extract(label_array: np.ndarray, **overrides) -> dict:
    """Run the kernel with explicit options (test defaults disable erosion)."""
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

    result = extract_habitat_nodes(label_array=label_array, connectivity="face")

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
    node_result = extract_habitat_nodes(label_array=label_array, connectivity="face")

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
    node_result = extract_habitat_nodes(label_array=label_array, connectivity="face")
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
    node_result = extract_habitat_nodes(label_array=label_array, connectivity="face")
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
        distance_threshold=3.0,
        subdivide_region_voxels=0,
        include_extended_metrics=False,
        pairwise_include_intra_edges=False,
    )
    features = extract_graph_features(label_array, options=options)
    assert features["pair_h1_h2_n_edges"] == 1.0


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
    nodes_nine = extract_habitat_nodes(label_array=labels_nine, connectivity="face")
    nodes_ten = extract_habitat_nodes(label_array=labels_ten, connectivity="face")

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
    node_result = extract_habitat_nodes(label_array=label_array, connectivity="face")

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
def test_subdivision_splits_large_region_into_multiple_nodes() -> None:
    """A large blob should become several grid-block nodes when subdivision is on."""
    label_array: np.ndarray = np.ones((10, 10), dtype=np.int32)

    without_split = extract_habitat_nodes(label_array=label_array)
    assert len(without_split.nodes_by_habitat[1]) == 1

    with_split = extract_habitat_nodes(
        label_array=label_array,
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
    assert options.block_size == 5
    assert options.distance_threshold == 5.0
    assert options.pairwise_include_intra_edges is True
    assert options.edge_method == "adjacency"
    assert options.adjacency_connectivity == "corner"
    assert options.connectivity == "full"
    assert options.adjacency_min_voxels == 10


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
def test_extended_graph_metrics_are_present_by_default() -> None:
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
    """Extended metrics should be omitted when include_extended_metrics is false."""
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
        include_extended_metrics=False,
    )

    assert "single_h1_global_efficiency" not in features
    assert "pair_h1_h2_small_world_sigma" not in features


@pytest.mark.unit
def test_betweenness_max_norm_scales_with_graph_size() -> None:
    """Normalized betweenness max should divide by the theoretical maximum."""
    label_array: np.ndarray = np.array(
        [
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    features = _extract(
        label_array,
        distance_threshold=5.0,
        subdivide_region_voxels=2,
        erosion_radius=0,
        include_extended_metrics=True,
    )

    n_nodes = features["single_h1_n_nodes"]
    if n_nodes >= 3 and features["single_h1_betweenness_max"] > 0:
        scale = (n_nodes - 1.0) * (n_nodes - 2.0) / 2.0
        assert features["single_h1_betweenness_max_norm"] == (
            features["single_h1_betweenness_max"] / scale
        )


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
        distance_threshold=3.0, erosion_radius=0, subdivide_region_voxels=0
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
    default_options = HabitatGraphFeatureOptions(subdivide_region_voxels=0)
    assert default_options.connectivity == "full"
    default_feats = extract_graph_features(label_array, options=default_options)
    face_feats = extract_graph_features(
        label_array,
        options=HabitatGraphFeatureOptions(
            subdivide_region_voxels=0, connectivity="face"
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
        adjacency_min_voxels=1, subdivide_region_voxels=0
    )
    assert default_options.adjacency_connectivity == "corner"
    default_feats = extract_graph_features(label_array, options=default_options)
    face_feats = extract_graph_features(
        label_array,
        options=HabitatGraphFeatureOptions(
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

    face = extract_habitat_nodes(label_array=label_array, connectivity="face")
    full = extract_habitat_nodes(label_array=label_array, connectivity="full")

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

    default_options = HabitatGraphFeatureOptions()
    assert default_options.erosion_radius == 0
    assert default_options.edge_method == "adjacency"
    assert default_options.adjacency_min_voxels == 10

    default_feats = extract_graph_features(label_array, options=default_options)
    eroded_feats = extract_graph_features(
        label_array,
        options=HabitatGraphFeatureOptions(erosion_radius=1),
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
        label_array=label_array, connectivity="face", erosion_radius=0
    )
    eroded = extract_habitat_nodes(
        label_array=label_array, connectivity="face", erosion_radius=1
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

    kept = extract_habitat_nodes(label_array=label_array, min_region_voxels=2)
    assert len(kept.nodes_by_habitat[1]) == 1
    assert kept.nodes_by_habitat[1][0].voxel_count == 4


@pytest.mark.unit
def test_block_min_coverage_filters_partial_blocks() -> None:
    """Low-coverage subdivision blocks are discarded; high coverage keeps them."""
    # 6x6 ones: with block_size=4, corner blocks are only partly covered.
    label_array: np.ndarray = np.ones((6, 6), dtype=np.int32)

    strict = extract_habitat_nodes(
        label_array=label_array,
        subdivide_region_voxels=1,
        block_size=4,
        block_min_coverage=0.9,
    )
    loose = extract_habitat_nodes(
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
    node_result = extract_habitat_nodes(label_array=label_array, connectivity="face")

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

    node_result = extract_habitat_nodes(label_array=label_array, connectivity="face")
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
    assert not any(key.startswith("single_") for key in features)
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
