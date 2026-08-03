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
"""Tests for the built-in habitat feature extractors."""

from __future__ import annotations

import numpy as np
import pytest

from habit.api.exceptions import HABITAPIError
from habit.domain.habitat_features import (
    HabitatFeatureExtractorRegistry,
    HabitatVolumeFeatures,
    IthHabitatFeatures,
    MsiHabitatFeatures,
)
from habit.domain.protocols import HabitatFeatureExtractor
from habit.kernels.habitat_metrics import (
    habitat_volume_fractions,
    ith_score,
    msi_features_from_matrix,
    spatial_interaction_matrix,
)

from .conftest import make_habitat_map, make_subject


@pytest.mark.unit
def test_extractors_satisfy_protocol() -> None:
    """Every built-in family structurally satisfies the protocol."""
    assert isinstance(MsiHabitatFeatures(), HabitatFeatureExtractor)
    assert isinstance(IthHabitatFeatures(), HabitatFeatureExtractor)
    assert isinstance(HabitatVolumeFeatures(), HabitatFeatureExtractor)


@pytest.mark.unit
def test_msi_features_match_kernel_values() -> None:
    """The extractor is a thin contract wrapper over the L0 kernels."""
    subject = make_subject("P1")
    habitat_map = make_habitat_map("P1")
    table = MsiHabitatFeatures()(subject, habitat_map)
    labels = np.asarray(habitat_map.label_array)
    expected = msi_features_from_matrix(spatial_interaction_matrix(labels, 3))
    assert table.id_columns == ("subject",)
    assert table.frame["subject"].tolist() == ["P1"]
    assert table.feature_columns == tuple(expected.keys())
    row = table.frame.iloc[0]
    for key, value in expected.items():
        assert row[key] == pytest.approx(value)
    assert "habitat_feature_extractor.msi" == table.provenance.produced_by


@pytest.mark.unit
def test_msi_columns_come_from_model_ids_not_present_labels() -> None:
    """A subject missing a habitat still yields the model's full columns."""
    subject = make_subject("P1")
    habitat_map = make_habitat_map("P1")
    habitat_map.label_array[np.asarray(habitat_map.label_array) == 2] = 1
    full = MsiHabitatFeatures()(subject, make_habitat_map("P1"))
    reduced = MsiHabitatFeatures()(subject, habitat_map)
    assert full.feature_columns == reduced.feature_columns


@pytest.mark.unit
def test_msi_rejects_maps_without_habitat_ids() -> None:
    """A map with no declared habitat ids cannot define the matrix size."""
    subject = make_subject("P1")
    habitat_map = make_habitat_map("P1")
    object.__setattr__(habitat_map, "habitat_ids", ())
    with pytest.raises(HABITAPIError):
        MsiHabitatFeatures()(subject, habitat_map)


@pytest.mark.unit
def test_ith_features_match_kernel_and_cover_all_model_habitats() -> None:
    """The ITH score matches the kernel; absent habitats report zeros."""
    subject = make_subject("P1")
    habitat_map = make_habitat_map("P1")
    labels = np.asarray(habitat_map.label_array)
    table = IthHabitatFeatures()(subject, habitat_map)
    row = table.frame.iloc[0]
    assert row["ith_score"] == pytest.approx(ith_score(labels))
    assert row["num_habitats"] == 2.0
    assert row["total_area"] == float(np.count_nonzero(labels))
    assert row["habitat_1_regions"] == 1.0
    assert row["habitat_2_regions"] == 1.0
    # An absent habitat (id 3 declared by a richer model) yields zeros.
    object.__setattr__(habitat_map, "habitat_ids", (1, 2, 3))
    table = IthHabitatFeatures()(subject, habitat_map)
    row = table.frame.iloc[0]
    assert row["habitat_3_regions"] == 0.0
    assert row["habitat_3_largest_area"] == 0.0
    assert row["habitat_3_area_ratio"] == 0.0


@pytest.mark.unit
def test_volume_features_counts_and_fractions() -> None:
    """Counts are voxel counts; fractions come from the shared kernel."""
    subject = make_subject("P1")
    habitat_map = make_habitat_map("P1")
    labels = np.asarray(habitat_map.label_array)
    table = HabitatVolumeFeatures()(subject, habitat_map)
    row = table.frame.iloc[0]
    fractions = habitat_volume_fractions(labels, (1, 2))
    assert row["habitat_1_voxel_count"] == float(np.count_nonzero(labels == 1))
    assert row["habitat_2_voxel_count"] == float(np.count_nonzero(labels == 2))
    assert row["habitat_1_volume_fraction"] == pytest.approx(fractions[1])
    assert row["habitat_2_volume_fraction"] == pytest.approx(fractions[2])
    assert row["habitat_1_volume_fraction"] + row["habitat_2_volume_fraction"] == (
        pytest.approx(1.0)
    )


@pytest.mark.unit
def test_feature_tables_join_across_families() -> None:
    """Tables from different families join on the subject id column."""
    subject = make_subject("P1")
    habitat_map = make_habitat_map("P1")
    msi = MsiHabitatFeatures()(subject, habitat_map)
    volume = HabitatVolumeFeatures()(subject, habitat_map)
    joined = msi.join(volume)
    assert joined.frame.shape[0] == 1
    assert set(joined.feature_columns) == set(msi.feature_columns) | set(
        volume.feature_columns
    )


@pytest.mark.unit
def test_habitat_feature_registry() -> None:
    """All three built-in families construct through their registry."""
    assert set(HabitatFeatureExtractorRegistry.available()) == {
        "ith",
        "msi",
        "volume",
    }
    assert isinstance(
        HabitatFeatureExtractorRegistry.create("msi"), MsiHabitatFeatures
    )
    assert isinstance(
        HabitatFeatureExtractorRegistry.create("ith"), IthHabitatFeatures
    )
    assert isinstance(
        HabitatFeatureExtractorRegistry.create("volume"), HabitatVolumeFeatures
    )
