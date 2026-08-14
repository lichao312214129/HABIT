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
"""Tests for the graph habitat-feature domain extractor."""

from __future__ import annotations

import numpy as np
import pytest

from habit.domain.habitat_features import (
    GraphHabitatFeatures,
    GraphHabitatFeaturesParams,
    HabitatFeatureExtractorRegistry,
)
from habit.domain.protocols import HabitatFeatureExtractor
from habit.kernels.habitat_graph import (
    HabitatGraphFeatureOptions,
    extract_graph_features,
)

from .conftest import make_habitat_map, make_subject

#: Options shared by the domain tests: deterministic, no erosion/subdivision.
_KERNEL_OVERRIDES = {
    "distance_threshold": 3.0,
    "erosion_radius": 0,
    "subdivide_region_voxels": 0,
}


@pytest.mark.unit
def test_graph_extractor_satisfies_protocol() -> None:
    """The graph family structurally satisfies the extractor protocol."""
    assert isinstance(GraphHabitatFeatures(), HabitatFeatureExtractor)


@pytest.mark.unit
def test_graph_extractor_matches_kernel_values() -> None:
    """The extractor is a thin contract wrapper over the L0 kernels."""
    subject = make_subject("P1")
    habitat_map = make_habitat_map("P1")
    extractor = GraphHabitatFeatures(**_KERNEL_OVERRIDES)
    table = extractor(subject, habitat_map)

    expected = extract_graph_features(
        np.asarray(habitat_map.label_array),
        options=HabitatGraphFeatureOptions(**_KERNEL_OVERRIDES),
        expected_labels=habitat_map.habitat_ids,
    )
    assert table.id_columns == ("subject",)
    assert table.frame["subject"].tolist() == ["P1"]
    assert table.feature_columns == tuple(expected.keys())
    row = table.frame.iloc[0]
    for key, value in expected.items():
        assert row[key] == pytest.approx(value)
    assert table.provenance.produced_by == "habitat_feature_extractor.graph"


@pytest.mark.unit
def test_graph_extractor_columns_cover_model_habitat_ids() -> None:
    """Columns exist for every model habitat id, even when one is absent."""
    subject = make_subject("P1")
    habitat_map = make_habitat_map("P1")
    # Drop habitat 2 from the label array; the model still declares (1, 2).
    labels = np.asarray(habitat_map.label_array).copy()
    labels[labels == 2] = 0
    habitat_map = type(habitat_map)(
        subject_id=habitat_map.subject_id,
        label_array=labels,
        geometry=habitat_map.geometry,
        model_id=habitat_map.model_id,
        habitat_ids=habitat_map.habitat_ids,
        provenance=habitat_map.provenance,
    )

    table = GraphHabitatFeatures(**_KERNEL_OVERRIDES)(subject, habitat_map)

    assert "single_h2_n_nodes" in table.feature_columns
    assert table.frame["single_h2_n_nodes"].iloc[0] == 0.0
    assert "pair_h1_h2_n_edges" in table.feature_columns
    # The present-habitat count reflects the actual map contents.
    assert table.frame["graph_num_habitats"].iloc[0] == 1.0


@pytest.mark.unit
def test_graph_spec_records_all_options() -> None:
    """The spec captures every extraction parameter for provenance."""
    extractor = GraphHabitatFeatures(distance_threshold=8.0, edge_method="adjacency")
    assert extractor.spec.name == "graph"
    assert extractor.spec.params["distance_threshold"] == 8.0
    assert extractor.spec.params["edge_method"] == "adjacency"
    assert extractor.spec.params["erosion_radius"] == 0
    min_dist = GraphHabitatFeatures(edge_method="min_distance", distance_threshold=3.0)
    assert min_dist.spec.params["edge_method"] == "min_distance"
    assert min_dist.spec.params["distance_threshold"] == 3.0


@pytest.mark.unit
def test_graph_registry_round_trip() -> None:
    """The registry creates the extractor and exposes its params model."""
    extractor = HabitatFeatureExtractorRegistry.create("graph", block_size=7)
    assert isinstance(extractor, GraphHabitatFeatures)
    assert extractor.spec.params["block_size"] == 7
    assert HabitatFeatureExtractorRegistry.params_model("graph") is (
        GraphHabitatFeaturesParams
    )


@pytest.mark.unit
def test_graph_params_model_defaults_match_kernel() -> None:
    """The pydantic params model mirrors the kernel option defaults."""
    params = GraphHabitatFeaturesParams()
    kernel = HabitatGraphFeatureOptions()
    assert params.erosion_radius == kernel.erosion_radius == 0
    assert params.subdivide_region_voxels == kernel.subdivide_region_voxels == 1000
    assert params.block_size == kernel.block_size == 5
    assert params.distance_threshold == kernel.distance_threshold == 5.0
    assert params.edge_method == kernel.edge_method == "adjacency"
    assert params.adjacency_connectivity == kernel.adjacency_connectivity == "corner"
    assert params.connectivity == kernel.connectivity == "full"
    assert params.adjacency_min_voxels == kernel.adjacency_min_voxels == 10
    assert params.pairwise_include_intra_edges is (
        kernel.pairwise_include_intra_edges
    )
    accepted = GraphHabitatFeaturesParams(edge_method="min_distance")
    assert accepted.edge_method == "min_distance"


@pytest.mark.unit
def test_graph_spec_fingerprint_changes_with_params() -> None:
    """Changing extraction params must change the provenance fingerprint."""
    baseline = GraphHabitatFeatures(**_KERNEL_OVERRIDES)
    altered = GraphHabitatFeatures(**{**_KERNEL_OVERRIDES, "distance_threshold": 9.0})

    assert baseline.spec.fingerprint() != altered.spec.fingerprint()
    assert baseline.spec.fingerprint() == GraphHabitatFeatures(
        **_KERNEL_OVERRIDES
    ).spec.fingerprint()


@pytest.mark.unit
def test_graph_feature_table_is_one_row_with_subject_id() -> None:
    """The domain extractor returns a one-row FeatureTable keyed by subject."""
    subject = make_subject("P_graph")
    habitat_map = make_habitat_map("P_graph")
    table = GraphHabitatFeatures(**_KERNEL_OVERRIDES)(subject, habitat_map)

    assert table.frame.shape[0] == 1
    assert "subject" in table.id_columns
    assert table.frame["subject"].iloc[0] == "P_graph"
    assert len(table.feature_columns) > 0
    assert set(table.feature_columns).isdisjoint(set(table.id_columns))
