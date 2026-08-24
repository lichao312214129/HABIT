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
from habit.contracts import ArrayImageRef, Geometry, Subject
from habit.domain.habitat_features import (
    EachHabitatRadiomicsFeatures,
    HabitatFeatureExtractorRegistry,
    HabitatVolumeFeatures,
    IthHabitatFeatures,
    MsiHabitatFeatures,
    NonRadiomicsHabitatFeatures,
    TraditionalRadiomicsHabitatFeatures,
    WholeHabitatRadiomicsFeatures,
)
from habit.domain.protocols import HabitatFeatureExtractor
from habit.kernels.habitat_metrics import (
    habitat_region_stats,
    habitat_volume_fractions,
    ith_score,
    msi_features_from_matrix,
    spatial_interaction_matrix,
)

from .conftest import make_habitat_map, make_subject, provenance

#: Minimal PyRadiomics configuration keeping the radiomics tests fast.
_MEAN_ONLY_PARAMS = {
    "imageType": {"Original": {}},
    "featureClass": {"firstorder": ["Mean"]},
    "setting": {"binWidth": 25},
}


def _aligned_subject(subject_id: str = "P1") -> Subject:
    """
    Build a subject whose 4x4x4 image aligns with ``make_habitat_map``.

    Habitat 1's block holds intensity 5 and habitat 2's block intensity 10,
    so every radiomics family has a hand-computable expected mean.
    """
    geometry = Geometry.from_array((4, 4, 4))
    image = np.zeros((4, 4, 4), dtype=np.float64)
    image[0:2, 0:2, 0:2] = 5.0
    image[2:4, 0:2, 0:2] = 10.0
    return Subject(
        subject_id=subject_id,
        images={"T1": ArrayImageRef(array=image, geometry=geometry)},
        masks={},
    )


@pytest.mark.unit
def test_extractors_satisfy_protocol() -> None:
    """Every built-in family structurally satisfies the protocol."""
    assert isinstance(MsiHabitatFeatures(), HabitatFeatureExtractor)
    assert isinstance(IthHabitatFeatures(), HabitatFeatureExtractor)
    assert isinstance(HabitatVolumeFeatures(), HabitatFeatureExtractor)
    assert isinstance(NonRadiomicsHabitatFeatures(), HabitatFeatureExtractor)
    assert isinstance(
        TraditionalRadiomicsHabitatFeatures(), HabitatFeatureExtractor
    )
    assert isinstance(WholeHabitatRadiomicsFeatures(), HabitatFeatureExtractor)
    assert isinstance(EachHabitatRadiomicsFeatures(), HabitatFeatureExtractor)


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
    table = IthHabitatFeatures(include_auxiliary=True)(subject, habitat_map)
    row = table.frame.iloc[0]
    assert row["ith_score"] == pytest.approx(ith_score(labels))
    assert row["ith_num_habitats"] == 2.0
    assert row["ith_total_area"] == float(np.count_nonzero(labels))
    assert row["habitat_1_regions"] == 1.0
    assert row["habitat_2_regions"] == 1.0
    # An absent habitat (id 3 declared by a richer model) yields zeros.
    object.__setattr__(habitat_map, "habitat_ids", (1, 2, 3))
    table = IthHabitatFeatures(include_auxiliary=True)(subject, habitat_map)
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
    """All built-in families construct through their registry."""
    assert set(HabitatFeatureExtractorRegistry.available()) == {
        "each_habitat",
        "graph",
        "ith_score",
        "msi",
        "non_radiomics",
        "traditional",
        "volume",
        "whole_habitat",
    }
    assert isinstance(
        HabitatFeatureExtractorRegistry.create("msi"), MsiHabitatFeatures
    )
    assert isinstance(
        HabitatFeatureExtractorRegistry.create("ith_score"), IthHabitatFeatures
    )
    assert isinstance(
        HabitatFeatureExtractorRegistry.create("volume"), HabitatVolumeFeatures
    )
    assert isinstance(
        HabitatFeatureExtractorRegistry.create("non_radiomics"),
        NonRadiomicsHabitatFeatures,
    )
    assert isinstance(
        HabitatFeatureExtractorRegistry.create("traditional"),
        TraditionalRadiomicsHabitatFeatures,
    )
    assert isinstance(
        HabitatFeatureExtractorRegistry.create("whole_habitat"),
        WholeHabitatRadiomicsFeatures,
    )
    assert isinstance(
        HabitatFeatureExtractorRegistry.create("each_habitat"),
        EachHabitatRadiomicsFeatures,
    )


@pytest.mark.unit
def test_non_radiomics_features_match_kernels() -> None:
    """Region counts and volume ratios come straight from the L0 kernels."""
    subject = make_subject("P1")
    habitat_map = make_habitat_map("P1")
    labels = np.asarray(habitat_map.label_array)
    table = NonRadiomicsHabitatFeatures()(subject, habitat_map)
    row = table.frame.iloc[0]
    region_stats = habitat_region_stats(labels)
    fractions = habitat_volume_fractions(labels, (1, 2))
    assert row["num_habitats"] == float(len(region_stats))
    assert row["1_num_regions"] == float(region_stats[1][0])
    assert row["2_num_regions"] == float(region_stats[2][0])
    assert row["1_volume_ratio"] == pytest.approx(fractions[1])
    assert row["2_volume_ratio"] == pytest.approx(fractions[2])
    # A habitat the model declares but the subject lacks yields zeros.
    object.__setattr__(habitat_map, "habitat_ids", (1, 2, 3))
    row = NonRadiomicsHabitatFeatures()(subject, habitat_map).frame.iloc[0]
    assert row["3_num_regions"] == 0.0
    assert row["3_volume_ratio"] == 0.0


@pytest.mark.unit
def test_non_radiomics_counts_disconnected_regions() -> None:
    """Splitting one habitat into two islands doubles its region count."""
    subject = make_subject("P1")
    habitat_map = make_habitat_map("P1")
    labels = np.asarray(habitat_map.label_array)
    labels[0, 0, 0] = 0  # detach one voxel from habitat 1's block... 
    labels[3, 3, 3] = 1  # ...and place an isolated habitat-1 voxel elsewhere
    row = NonRadiomicsHabitatFeatures()(subject, habitat_map).frame.iloc[0]
    assert row["1_num_regions"] == float(habitat_region_stats(labels)[1][0])
    assert row["1_num_regions"] == 2.0


@pytest.mark.unit
def test_traditional_radiomics_masked_mean_and_column_naming() -> None:
    """Columns are {feature}_of_{modality}; the mean is the masked mean."""
    subject = _aligned_subject()
    habitat_map = make_habitat_map("P1")
    table = TraditionalRadiomicsHabitatFeatures(params=_MEAN_ONLY_PARAMS)(
        subject, habitat_map
    )
    labels = np.asarray(habitat_map.label_array)
    image = np.asarray(subject.image("T1").load())
    assert table.feature_columns == ("original_firstorder_Mean_of_T1",)
    assert table.frame.iloc[0]["original_firstorder_Mean_of_T1"] == pytest.approx(
        image[labels > 0].mean()
    )


@pytest.mark.unit
def test_traditional_radiomics_modality_resolution_and_exclusivity() -> None:
    """Unknown modalities and dual params sources are explicit errors."""
    subject = _aligned_subject()
    habitat_map = make_habitat_map("P1")
    with pytest.raises(HABITAPIError):
        TraditionalRadiomicsHabitatFeatures(
            params=_MEAN_ONLY_PARAMS, modalities=["T2"]
        )(subject, habitat_map)
    with pytest.raises(HABITAPIError):
        TraditionalRadiomicsHabitatFeatures(
            params_file="params.yaml", params=_MEAN_ONLY_PARAMS
        )(subject, habitat_map)


@pytest.mark.unit
def test_whole_habitat_radiomics_uses_the_label_image() -> None:
    """The habitat map itself is the image: bare PyRadiomics column names."""
    subject = _aligned_subject()
    habitat_map = make_habitat_map("P1")
    table = WholeHabitatRadiomicsFeatures(params=_MEAN_ONLY_PARAMS)(
        subject, habitat_map
    )
    labels = np.asarray(habitat_map.label_array)
    assert table.feature_columns == ("original_firstorder_Mean",)
    assert table.frame.iloc[0]["original_firstorder_Mean"] == pytest.approx(
        labels[labels > 0].mean()
    )


@pytest.mark.unit
def test_each_habitat_radiomics_per_habitat_columns() -> None:
    """Per-habitat means land in habitat_{id}_*_of_{modality} columns."""
    subject = _aligned_subject()
    habitat_map = make_habitat_map("P1")
    table = EachHabitatRadiomicsFeatures(params=_MEAN_ONLY_PARAMS)(
        subject, habitat_map
    )
    row = table.frame.iloc[0]
    assert row["has_habitat_1"] == 1.0
    assert row["has_habitat_2"] == 1.0
    assert row["habitat_1_original_firstorder_Mean_of_T1"] == pytest.approx(5.0)
    assert row["habitat_2_original_firstorder_Mean_of_T1"] == pytest.approx(10.0)


@pytest.mark.unit
def test_each_habitat_matches_execute_per_habitat_bin() -> None:
    """each_habitat keeps per-habitat binWidth (execute science), not union bin."""
    from habit.domain.habitat_features._radiomics import (
        build_pyradiomics_extractor,
        execute_radiomics,
        harmonize_mask_geometry,
        sitk_image_from_contract,
    )

    subject = _aligned_subject()
    habitat_map = make_habitat_map("P1")
    params = {
        "imageType": {"Original": {}},
        "featureClass": {
            "firstorder": ["Mean", "Energy"],
            "glcm": ["Autocorrelation", "Id"],
        },
        "setting": {"binWidth": 25, "voxelArrayShift": 0},
    }
    table = EachHabitatRadiomicsFeatures(params=params)(subject, habitat_map)
    row = table.frame.iloc[0]
    assert row["has_habitat_1"] == 1.0
    assert row["has_habitat_2"] == 1.0

    extractor = build_pyradiomics_extractor(None, params, owner="test")
    volume = subject.image("T1")
    image_sitk = sitk_image_from_contract(volume.load(), volume.geometry)
    mask_sitk = sitk_image_from_contract(
        np.asarray(habitat_map.label_array), habitat_map.geometry
    )
    harmonize_mask_geometry(image_sitk, mask_sitk)
    for habitat_id in (1, 2):
        executed = execute_radiomics(extractor, image_sitk, mask_sitk, habitat_id)
        for feature in (
            "original_firstorder_Mean",
            "original_firstorder_Energy",
            "original_glcm_Id",
            "original_glcm_Autocorrelation",
        ):
            col = f"habitat_{habitat_id}_{feature}_of_T1"
            assert col in table.feature_columns
            assert float(row[col]) == pytest.approx(
                float(executed[feature]), rel=1e-8, abs=1e-8
            )


@pytest.mark.unit
def test_each_habitat_absent_habitats_are_nan_not_zero() -> None:
    """A declared-but-absent habitat reports NaN and a 0 presence flag."""
    subject = _aligned_subject()
    habitat_map = make_habitat_map("P1")
    object.__setattr__(habitat_map, "habitat_ids", (1, 2, 3))
    table = EachHabitatRadiomicsFeatures(params=_MEAN_ONLY_PARAMS)(
        subject, habitat_map
    )
    row = table.frame.iloc[0]
    assert row["has_habitat_3"] == 0.0
    assert np.isnan(row["habitat_3_original_firstorder_Mean_of_T1"])
    # The column layout is canonical: a map containing only habitat 1 but
    # declaring the same model ids yields the very same columns and order.
    reduced = make_habitat_map("P1")
    reduced.label_array[np.asarray(reduced.label_array) == 2] = 1
    object.__setattr__(reduced, "habitat_ids", (1, 2, 3))
    reduced_table = EachHabitatRadiomicsFeatures(params=_MEAN_ONLY_PARAMS)(
        subject, reduced
    )
    assert reduced_table.feature_columns == table.feature_columns
    assert reduced_table.frame.iloc[0]["has_habitat_2"] == 0.0
    assert np.isnan(reduced_table.frame.iloc[0]["habitat_2_original_firstorder_Mean_of_T1"])


@pytest.mark.unit
def test_ith_and_non_radiomics_tables_join_without_collisions() -> None:
    """The two families sharing v0.1 summary names join cleanly in v1."""
    subject = make_subject("P1")
    habitat_map = make_habitat_map("P1")
    joined = IthHabitatFeatures(include_auxiliary=True)(subject, habitat_map).join(
        NonRadiomicsHabitatFeatures()(subject, habitat_map)
    )
    row = joined.frame.iloc[0]
    assert row["ith_num_habitats"] == row["num_habitats"] == 2.0
