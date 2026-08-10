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
    extract_graph_features,
    extract_habitat_nodes,
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
def test_default_options_enable_erosion_and_subdivision() -> None:
    """Default kernel options should erode and subdivide, per the config schema."""
    options = HabitatGraphFeatureOptions()

    assert options.erosion_radius == 1
    assert options.subdivide_region_voxels == 1000
    assert options.block_size == 5
    assert options.distance_threshold == 5.0
    assert options.pairwise_include_intra_edges is True


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
