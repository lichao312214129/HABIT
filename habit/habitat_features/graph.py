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
"""Graph-topology habitat features (region graphs + NetworkX metrics)."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

import numpy as np

from habit.contracts.habitat import HabitatMap
from habit.contracts.subject import Subject
from habit.contracts.table import FeatureTable
from habit.habitat_features._base import single_subject_table
from habit.habitat_features.registry import HabitatFeatureExtractorRegistry
from habit.kernels.habitat_graph import (
    HabitatGraphFeatureOptions,
    extract_graph_features,
)
from habit.spec.specs import Spec

__all__ = ["GraphHabitatFeatures"]


def _validate_graph_options(
    *,
    edge_method: str,
    distance_threshold: float,
    adjacency_connectivity: str,
    adjacency_min_voxels: int,
    edge_weight: str,
    min_region_voxels: int,
    connectivity: str,
    erosion_radius: int,
    node_method: str,
    subdivide_region_voxels: int,
    block_size: int,
    block_min_coverage: float,
    extended_min_nodes: int,
    small_world_nrand: int,
    small_world_niter: int,
    rich_club_q: int,
    graph_null_sampler: str,
) -> None:
    """Reject invalid graph-construction options before numerical execution."""
    allowed = {
        "edge_method": {"centroid_distance", "adjacency", "min_distance"},
        "adjacency_connectivity": {"face", "edge", "corner"},
        "edge_weight": {"none", "distance", "inverse_distance", "contact_voxels"},
        "connectivity": {"face", "full"},
        "node_method": {"uniform_grid", "component"},
        "graph_null_sampler": {"analytic", "config", "rewire"},
    }
    values = {
        "edge_method": edge_method,
        "adjacency_connectivity": adjacency_connectivity,
        "edge_weight": edge_weight,
        "connectivity": connectivity,
        "node_method": node_method,
        "graph_null_sampler": graph_null_sampler,
    }
    for name, value in values.items():
        if value not in allowed[name]:
            raise ValueError(f"{name} must be one of {sorted(allowed[name])}; got {value!r}.")
    non_negative = {
        "distance_threshold": distance_threshold,
        "erosion_radius": erosion_radius,
        "subdivide_region_voxels": subdivide_region_voxels,
    }
    positive = {
        "adjacency_min_voxels": adjacency_min_voxels,
        "min_region_voxels": min_region_voxels,
        "block_size": block_size,
        "extended_min_nodes": extended_min_nodes,
        "small_world_nrand": small_world_nrand,
        "small_world_niter": small_world_niter,
        "rich_club_q": rich_club_q,
    }
    for name, value in non_negative.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"{name} must be a non-negative number; got {value!r}.")
    for name, value in positive.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    if extended_min_nodes < 3:
        raise ValueError(
            f"extended_min_nodes must be an integer of at least 3; got {extended_min_nodes!r}."
        )
    if (
        isinstance(block_min_coverage, bool)
        or not isinstance(block_min_coverage, (int, float))
        or not 0 <= block_min_coverage <= 1
    ):
        raise ValueError(
            "block_min_coverage must be a number in [0, 1]; "
            f"got {block_min_coverage!r}."
        )


@HabitatFeatureExtractorRegistry.register("graph")
class GraphHabitatFeatures:
    """
    Graph-topology features of one subject's habitat map.

    Default nodes are per-cell subregion centroids on a global VOI
    lattice (``node_method='uniform_grid'``, ``block_size=8`` voxels,
    not mm): each kept cube can contribute several nodes.
    Default edges connect cubes whose closest voxels are within
    ``distance_threshold`` (``edge_method='min_distance'``, default 5).
    There is no morphological erosion (``erosion_radius=0``). Pass
    ``node_method='component'`` for connected-component nodes, and
    ``edge_method='adjacency'`` for contact-voxel edges (default
    ``adjacency_min_voxels=10``, ``adjacency_connectivity='corner'``).
    ``centroid_distance`` connects centroids within
    ``distance_threshold``. NetworkX-derived topology
    metrics are reported per habitat (``single_h*`` columns) and per habitat
    pair (``pair_h*_h*`` columns), covering degree/edge counts, density,
    components, modularity, clustering, path length, betweenness, assortativity,
    nearest-neighbor ratio, and -- by default -- efficiency, small-world sigma,
    rich-club, and node-distribution summaries. Size-dependent features carry
    VOI-normalized companions (``*_norm`` / ``*_per_habitat_volume``).

    The numeric definitions live in the L0 kernels
    :mod:`habit.kernels.habitat_graph` and are identical to the established
    implementation this family was migrated from. Like
    :class:`~habit.habitat_features.ith.IthHabitatFeatures`, columns are
    emitted for every id in the map's ``habitat_ids`` (absent habitats yield
    zero-valued empty-graph metrics), so cohort tables have stable columns.
    """

    def __init__(
        self,
        include_single_habitat_graph: bool = True,
        include_pairwise_habitat_graph: bool = True,
        edge_method: Literal[
            "centroid_distance", "adjacency", "min_distance"
        ] = "min_distance",
        distance_threshold: float = 5.0,
        adjacency_connectivity: Literal["face", "edge", "corner"] = "corner",
        adjacency_min_voxels: int = 10,
        edge_weight: Literal[
            "none", "distance", "inverse_distance", "contact_voxels"
        ] = "none",
        min_region_voxels: int = 1,
        connectivity: Literal["face", "full"] = "full",
        erosion_radius: int = 0,
        node_method: Literal["uniform_grid", "component"] = "uniform_grid",
        subdivide_region_voxels: int = 1000,
        block_size: int = 8,
        block_min_coverage: float = 0.2,
        pairwise_include_intra_edges: bool = True,
        include_extended_metrics: bool = True,
        extended_min_nodes: int = 10,
        small_world_nrand: int = 100,
        small_world_niter: int = 100,
        rich_club_q: int = 100,
        graph_null_sampler: Literal["analytic", "config", "rewire"] = "analytic",
        graph_null_device: str = "auto",
    ) -> None:
        _validate_graph_options(
            edge_method=edge_method,
            distance_threshold=distance_threshold,
            adjacency_connectivity=adjacency_connectivity,
            adjacency_min_voxels=adjacency_min_voxels,
            edge_weight=edge_weight,
            min_region_voxels=min_region_voxels,
            connectivity=connectivity,
            erosion_radius=erosion_radius,
            node_method=node_method,
            subdivide_region_voxels=subdivide_region_voxels,
            block_size=block_size,
            block_min_coverage=block_min_coverage,
            extended_min_nodes=extended_min_nodes,
            small_world_nrand=small_world_nrand,
            small_world_niter=small_world_niter,
            rich_club_q=rich_club_q,
            graph_null_sampler=graph_null_sampler,
        )
        self._options = HabitatGraphFeatureOptions(
            include_single_habitat_graph=include_single_habitat_graph,
            include_pairwise_habitat_graph=include_pairwise_habitat_graph,
            edge_method=edge_method,
            distance_threshold=distance_threshold,
            adjacency_connectivity=adjacency_connectivity,
            adjacency_min_voxels=adjacency_min_voxels,
            edge_weight=edge_weight,
            min_region_voxels=min_region_voxels,
            connectivity=connectivity,
            erosion_radius=erosion_radius,
            node_method=node_method,
            subdivide_region_voxels=subdivide_region_voxels,
            block_size=block_size,
            block_min_coverage=block_min_coverage,
            pairwise_include_intra_edges=pairwise_include_intra_edges,
            include_extended_metrics=include_extended_metrics,
            extended_min_nodes=extended_min_nodes,
            small_world_nrand=small_world_nrand,
            small_world_niter=small_world_niter,
            rich_club_q=rich_club_q,
            graph_null_sampler=graph_null_sampler,
            graph_null_device=graph_null_device,
        )
        for field in HabitatGraphFeatureOptions.__dataclass_fields__:
            setattr(self, field, getattr(self._options, field))

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        params: Dict[str, Any] = {
            field: getattr(self._options, field)
            for field in HabitatGraphFeatureOptions.__dataclass_fields__
        }
        return Spec(name="graph", params=params)

    def __call__(self, subject: Subject, habitat_map: HabitatMap) -> FeatureTable:
        """
        Compute the graph-topology feature family for one subject.

        Args:
            subject: Owning subject (labels suffice; intensities unused).
            habitat_map: Habitat labels for that subject.

        Returns:
            One-row table of graph-topology features keyed by subject id.
        """
        labels = np.asarray(habitat_map.label_array)
        features = extract_graph_features(
            labels,
            options=self._options,
            expected_labels=habitat_map.habitat_ids,
        )
        return single_subject_table(
            subject_id=subject.subject_id,
            features=features,
            habitat_map=habitat_map,
            spec=self.spec,
        )

