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
"""Data models shared by the habitat graph kernels.

These frozen dataclasses are the lightweight intermediate representation of a
habitat graph: connected-region nodes, typed edges, and the assembled graph
that the metric functions consume before converting to NetworkX.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

__all__ = [
    "EdgeMethod",
    "EdgeWeightMode",
    "GraphKind",
    "HabitatGraphNode",
    "HabitatGraphEdge",
    "HabitatGraph",
    "HabitatNodeExtractionResult",
]


#: Edge identification strategy: centroid proximity or voxel adjacency.
EdgeMethod = Literal["centroid_distance", "adjacency"]
#: Optional edge weight source for the built graph.
EdgeWeightMode = Literal["none", "distance", "inverse_distance", "contact_voxels"]
#: Graph scope: one habitat label, or one pair of habitat labels.
GraphKind = Literal["single", "pairwise"]


@dataclass(frozen=True)
class HabitatGraphNode:
    """A connected habitat region represented as one graph node."""

    node_id: str
    habitat_label: int
    component_id: int
    centroid: np.ndarray
    voxel_count: int
    bbox: Tuple[int, ...]


@dataclass(frozen=True)
class HabitatGraphEdge:
    """A graph edge with optional distance/contact evidence for weighting."""

    source: str
    target: str
    edge_type: str
    distance: Optional[float]
    contact_voxels: Optional[int]
    weight: float


@dataclass(frozen=True)
class HabitatGraph:
    """A lightweight graph object used before conversion to NetworkX for metrics."""

    graph_kind: GraphKind
    labels: Tuple[int, ...]
    nodes: Dict[str, HabitatGraphNode]
    edges: List[HabitatGraphEdge]


@dataclass(frozen=True)
class HabitatNodeExtractionResult:
    """Connected-component node extraction result plus component maps."""

    label_array: np.ndarray
    nodes_by_habitat: Dict[int, List[HabitatGraphNode]]
    component_maps: Dict[int, np.ndarray]
