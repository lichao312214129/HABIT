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
from pydantic import BaseModel, ConfigDict, Field

from habit.contracts.habitat import HabitatMap
from habit.contracts.subject import Subject
from habit.contracts.table import FeatureTable
from habit.domain.habitat_features._base import single_subject_table
from habit.domain.habitat_features.registry import HabitatFeatureExtractorRegistry
from habit.kernels.habitat_graph import (
    HabitatGraphFeatureOptions,
    extract_graph_features,
)
from habit.spec.specs import Spec

__all__ = ["GraphHabitatFeatures", "GraphHabitatFeaturesParams"]


class GraphHabitatFeaturesParams(BaseModel):
    """Constructor parameters for :class:`GraphHabitatFeatures`."""

    model_config = ConfigDict(extra="forbid")

    #: Compute within-habitat region graphs for each habitat label.
    include_single_habitat_graph: bool = True
    #: Compute pairwise inter-habitat region graphs.
    include_pairwise_habitat_graph: bool = True
    #: Rule used to identify graph edges. Default ``"adjacency"``: connect
    #: regions that share a boundary.
    edge_method: Literal["centroid_distance", "adjacency"] = "adjacency"
    #: Centroid distance threshold in pixel units.
    distance_threshold: float = Field(default=5.0, ge=0.0)
    #: Neighbor definition for the ``"adjacency"`` edge method. Default
    #: ``"corner"`` = 8-connectivity in 2D / 26-connectivity in 3D.
    #: ``"face"`` = 4/6-connectivity; ``"edge"`` = 8/18-connectivity.
    adjacency_connectivity: Literal["face", "edge", "corner"] = "corner"
    #: Minimum number of adjacent voxel pairs required to create an edge when
    #: ``edge_method`` is ``"adjacency"``. Default ``10``: an edge exists only
    #: when two regions are adjacent and the contact voxel count is >= 10.
    adjacency_min_voxels: int = Field(default=10, ge=1)
    #: Optional edge weight source.
    edge_weight: Literal["none", "distance", "inverse_distance", "contact_voxels"] = (
        "none"
    )
    #: Minimum connected-region size retained as a graph node.
    min_region_voxels: int = Field(default=1, ge=1)
    #: Connected-component neighborhood rule. Default ``"full"`` =
    #: 8-connectivity in 2D / 26-connectivity in 3D. Pass ``"face"`` for
    #: 4/6-connectivity.
    connectivity: Literal["face", "full"] = "full"
    #: Binary erosion iterations applied before component labeling. Default
    #: ``0`` (off): adjacency and contact are measured on the habitat labels
    #: as drawn. Pass a positive value to shrink each habitat before edges.
    erosion_radius: int = Field(default=0, ge=0)
    #: Split connected components larger than this voxel count into grid blocks
    #: so large habitats do not collapse into a single node. Set 0 to disable.
    subdivide_region_voxels: int = Field(default=1000, ge=0)
    #: Grid block edge length (voxels) used when subdividing. Keep it close to
    #: ``distance_threshold`` so adjacent blocks connect into a lattice.
    block_size: int = Field(default=5, ge=1)
    #: Minimum covered fraction of a block volume to keep it as a node.
    block_min_coverage: float = Field(default=0.5, ge=0.0, le=1.0)
    #: Add same-habitat proximity edges to pairwise graphs so whole-graph
    #: metrics (modularity, assortativity, betweenness) reflect real tissue
    #: organization; interface metrics still use inter-class edges only.
    pairwise_include_intra_edges: bool = True
    #: Compute extended graph metrics: global/local efficiency, small-world
    #: sigma, rich-club coefficient, and node-level distribution summaries.
    include_extended_metrics: bool = True
    #: Minimum node count in the analysis subgraph required to compute
    #: small-world sigma; smaller graphs return 0 for that metric.
    extended_min_nodes: int = Field(default=10, ge=3)


@HabitatFeatureExtractorRegistry.register("graph")
class GraphHabitatFeatures:
    """
    Graph-topology features of one subject's habitat map.

    Each connected habitat region (optionally eroded and subdivided into grid
    blocks) becomes a graph node. By default there is no morphological
    erosion (``erosion_radius=0``): adjacency is measured on the habitat
    labels as drawn. With ``edge_method='adjacency'`` an edge exists when
    two regions are adjacent (default ``adjacency_connectivity='corner'``,
    8-connected in 2D / 26-connected in 3D) and the number of contact
    (shared-boundary) voxels is at least ``adjacency_min_voxels``
    (default ``10``). Components use ``connectivity='full'`` (same 8/26
    neighborhood). Pass ``face`` on both fields for 4/6-connectivity.
    Pass a positive ``erosion_radius`` to shrink habitats before edges. The
    alternative ``centroid_distance`` method connects centroids
    within ``distance_threshold``. NetworkX-derived topology
    metrics are reported per habitat (``single_h*`` columns) and per habitat
    pair (``pair_h*_h*`` columns), covering degree/edge counts, density,
    components, modularity, clustering, path length, betweenness, assortativity,
    nearest-neighbor ratio, and -- optionally -- efficiency, small-world sigma,
    rich-club, and node-distribution summaries. Size-dependent features carry
    VOI-normalized companions (``*_norm`` / ``*_per_habitat_volume``).

    The numeric definitions live in the L0 kernels
    :mod:`habit.kernels.habitat_graph` and are identical to the established
    implementation this family was migrated from. Like
    :class:`~habit.domain.habitat_features.ith.IthHabitatFeatures`, columns are
    emitted for every id in the map's ``habitat_ids`` (absent habitats yield
    zero-valued empty-graph metrics), so cohort tables have stable columns.
    """

    def __init__(
        self,
        include_single_habitat_graph: bool = True,
        include_pairwise_habitat_graph: bool = True,
        edge_method: Literal["centroid_distance", "adjacency"] = "adjacency",
        distance_threshold: float = 5.0,
        adjacency_connectivity: Literal["face", "edge", "corner"] = "corner",
        adjacency_min_voxels: int = 10,
        edge_weight: Literal[
            "none", "distance", "inverse_distance", "contact_voxels"
        ] = "none",
        min_region_voxels: int = 1,
        connectivity: Literal["face", "full"] = "full",
        erosion_radius: int = 0,
        subdivide_region_voxels: int = 1000,
        block_size: int = 5,
        block_min_coverage: float = 0.5,
        pairwise_include_intra_edges: bool = True,
        include_extended_metrics: bool = True,
        extended_min_nodes: int = 10,
    ) -> None:
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
            subdivide_region_voxels=subdivide_region_voxels,
            block_size=block_size,
            block_min_coverage=block_min_coverage,
            pairwise_include_intra_edges=pairwise_include_intra_edges,
            include_extended_metrics=include_extended_metrics,
            extended_min_nodes=extended_min_nodes,
        )

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


HabitatFeatureExtractorRegistry.register_params_model(
    "graph", GraphHabitatFeaturesParams
)
