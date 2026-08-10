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
"""L0 kernels for graph-topology habitat features.

A habitat map is turned into region graphs -- one node per connected habitat
region (optionally subdivided into grid blocks), edges by centroid proximity
or voxel adjacency -- and NetworkX-derived topology metrics are computed per
habitat (``single_h*``) and per habitat pair (``pair_h*_h*``). Everything here
is pure: arrays in, numbers out, no IO, no state, no configuration files.
"""

from __future__ import annotations

from habit.kernels.habitat_graph.edges import (
    build_adjacency_graph,
    build_centroid_distance_graph,
    iter_cross_label_nodes,
    iter_label_pairs,
)
from habit.kernels.habitat_graph.features import (
    HabitatGraphFeatureOptions,
    extract_graph_features,
    extract_graph_features_for_labels,
    pair_count,
)
from habit.kernels.habitat_graph.models import (
    EdgeMethod,
    EdgeWeightMode,
    GraphKind,
    HabitatGraph,
    HabitatGraphEdge,
    HabitatGraphNode,
    HabitatNodeExtractionResult,
)
from habit.kernels.habitat_graph.nodes import extract_habitat_nodes

__all__ = [
    "EdgeMethod",
    "EdgeWeightMode",
    "GraphKind",
    "HabitatGraph",
    "HabitatGraphEdge",
    "HabitatGraphNode",
    "HabitatNodeExtractionResult",
    "HabitatGraphFeatureOptions",
    "extract_habitat_nodes",
    "build_centroid_distance_graph",
    "build_adjacency_graph",
    "iter_label_pairs",
    "iter_cross_label_nodes",
    "extract_graph_features",
    "extract_graph_features_for_labels",
    "pair_count",
]
