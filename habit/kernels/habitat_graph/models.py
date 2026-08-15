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
    "label_token",
    "single_feature_prefix",
    "pair_feature_prefix",
    "EdgeMethod",
    "EdgeWeightMode",
    "GraphKind",
    "NodeMethod",
    "HabitatGraphNode",
    "HabitatGraphEdge",
    "HabitatGraph",
    "HabitatNodeExtractionResult",
]


def label_token(label: int) -> str:
    """Return ``h{k}`` for one habitat class."""
    return f"h{int(label)}"


def single_feature_prefix(label: int) -> str:
    """Return ``single_h{k}`` for one habitat class."""
    return f"single_{label_token(label)}"


def pair_feature_prefix(label_a: int, label_b: int) -> str:
    """Return ``pair_h1_h2`` for one unordered habitat pair."""
    return f"pair_{label_token(label_a)}_{label_token(label_b)}"


#: Edge identification strategy: centroid proximity, voxel adjacency, or
#: closest-voxel (minimum) Euclidean distance between regions.
EdgeMethod = Literal["centroid_distance", "adjacency", "min_distance"]
#: How voxels become graph nodes: a global cube lattice with one node
#: per in-cell subregion centroid, or connected components (optionally
#: split when they exceed a size).
NodeMethod = Literal["uniform_grid", "component"]
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
    """Node extraction result plus component maps and optional lattice."""

    label_array: np.ndarray
    nodes_by_habitat: Dict[int, List[HabitatGraphNode]]
    component_maps: Dict[int, np.ndarray]
    #: Inclusive voxel-index origin of the global lattice (``uniform_grid``).
    grid_origin: Optional[Tuple[int, ...]] = None
    #: Cube edge length in voxels of that lattice (``uniform_grid``).
    grid_block_size: Optional[int] = None
