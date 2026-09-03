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
"""Contract tests for feature composition trees and statistical extractors.

Trees evaluate nested specs (combiner nodes over extractor leaves) into the
existing level contracts, so the pipeline sees no difference between a
single extractor and a composed tree. These tests pin column naming,
aliasing, numeric agreement with pandas reductions, field binding, and the
failure modes that guard against silent misconfiguration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from habit.contracts import Subject
from habit.pipeline import SubjectPipeline
from habit.supervoxel import SupervoxelizerRegistry
from habit.supervoxel import (
    MeanSupervoxelFeatures,
    PercentileSupervoxelFeatures,
    StdSupervoxelFeatures,
)
from habit._feature_trees import (
    build_habitat_extractor,
    build_supervoxel_extractor,
    build_voxel_extractor,
)
from habit.exceptions import HABITAPIError
from habit.spec.specs import Spec

from tests.domain.conftest import make_subject, provenance


@pytest.fixture
def two_modality_subject() -> Subject:
    """Two-modality synthetic subject on a small grid."""
    return make_subject("P1", shape=(6, 6, 6), modalities=("T1", "T2"))


def _kmeans():
    """Return a tiny deterministic supervoxelizer."""
    return SupervoxelizerRegistry.create("kmeans", n_supervoxels=4, n_init=2)


class TestVoxelTrees:
    """Voxel-level trees merge child fields column-wise."""

    def test_concat_two_modalities(self, two_modality_subject: Subject) -> None:
        tree = build_voxel_extractor(
            Spec("concat", {"children": [
                {"name": "raw", "params": {"modality": "T1"}},
                {"name": "raw", "params": {"modality": "T2"}},
            ]})
        )
        field = tree(two_modality_subject)
        assert list(field.feature_names) == ["T1", "T2"]
        assert field.values.shape[1] == 2

    def test_leaf_alias_renames_column(self, two_modality_subject: Subject) -> None:
        leaf = build_voxel_extractor(Spec("raw", {"modality": "T1", "as_": "coarse"}))
        assert list(leaf(two_modality_subject).feature_names) == ["coarse"]

    def test_nested_combiner_with_alias(self, two_modality_subject: Subject) -> None:
        tree = build_voxel_extractor(
            Spec("concat", {"children": [
                {"name": "raw", "params": {"modality": "T1"}},
                {"name": "ratio", "params": {"as_": "t1_over_t2", "children": [
                    {"name": "raw", "params": {"modality": "T1"}},
                    {"name": "raw", "params": {"modality": "T2"}},
                ]}},
            ]})
        )
        field = tree(two_modality_subject)
        assert list(field.feature_names) == ["T1", "t1_over_t2"]

    def test_weighted_concat_scales_child(self, two_modality_subject: Subject) -> None:
        tree = build_voxel_extractor(
            Spec("weighted_concat", {"weights": {"T1": 2.0}, "children": [
                {"name": "raw", "params": {"modality": "T1"}},
                {"name": "raw", "params": {"modality": "T2"}},
            ]})
        )
        reference = build_voxel_extractor(
            Spec("concat", {"children": [
                {"name": "raw", "params": {"modality": "T1"}},
                {"name": "raw", "params": {"modality": "T2"}},
            ]})
        )
        scaled = tree(two_modality_subject).values
        plain = reference(two_modality_subject).values
        np.testing.assert_allclose(scaled[:, 0], 2.0 * plain[:, 0])
        np.testing.assert_allclose(scaled[:, 1], plain[:, 1])

    def test_duplicate_columns_fail(self, two_modality_subject: Subject) -> None:
        tree = build_voxel_extractor(
            Spec("concat", {"children": [
                {"name": "raw", "params": {"modality": "T1"}},
                {"name": "raw", "params": {"modality": "T1"}},
            ]})
        )
        with pytest.raises(HABITAPIError, match="duplicate"):
            tree(two_modality_subject)

    def test_alias_on_multi_column_node_fails(
        self, two_modality_subject: Subject
    ) -> None:
        tree = build_voxel_extractor(
            Spec("concat", {"as_": "oops", "children": [
                {"name": "raw", "params": {"modality": "T1"}},
                {"name": "raw", "params": {"modality": "T2"}},
            ]})
        )
        with pytest.raises(HABITAPIError, match="single output column"):
            tree(two_modality_subject)

    def test_roi_param_overrides_leaf_roi(self) -> None:
        tree = build_voxel_extractor(
            Spec("concat", {"roi": "tumor", "children": [
                {"name": "raw", "params": {"modality": "T1"}},
            ]})
        )
        assert getattr(tree, "_roi") == "tumor"
        assert tree._children[0].roi == "tumor"

    def test_combiner_without_children_fails(self) -> None:
        # ``ratio`` is a pure combiner (not also a legacy leaf like
        # ``concat``), so it must refuse to build without children.
        with pytest.raises(HABITAPIError, match="children"):
            build_voxel_extractor(Spec("ratio", {}))

    def test_unknown_component_fails(self) -> None:
        from habit.exceptions import ComponentNotFoundError

        with pytest.raises(ComponentNotFoundError, match="Unknown"):
            build_voxel_extractor(Spec("no_such_extractor", {}))


class TestSupervoxelStatistics:
    """mean / std / percentile reduce the voxel signal within each region."""

    def _units(self, subject: Subject, extractor) -> pd.DataFrame:
        """Run a pipeline with the given supervoxel extractor; return features."""
        voxel = build_voxel_extractor(Spec("raw", {"modalities": ["T1", "T2"]}))
        pipeline = SubjectPipeline(
            voxel, _kmeans(), None, supervoxel_feature_extractor=extractor
        )
        units = pipeline.units(subject)
        return units, voxel(subject)

    def test_mean_matches_groupby(self, two_modality_subject: Subject) -> None:
        units, field = self._units(
            two_modality_subject, MeanSupervoxelFeatures(modality="T1")
        )
        assert list(units.features.columns) == ["T1"]
        labels = np.asarray(units.label_array)[tuple(field.voxel_index.T)]
        frame = pd.DataFrame(field.values, columns=list(field.feature_names))
        frame["sv"] = labels
        reference = frame[frame["sv"] > 0].groupby("sv").mean()["T1"]
        np.testing.assert_allclose(
            reference.to_numpy(), units.features["T1"].to_numpy()
        )

    def test_std_and_percentile_naming(self, two_modality_subject: Subject) -> None:
        units, field = self._units(
            two_modality_subject,
            build_supervoxel_extractor(
                Spec("concat", {"children": [
                    {"name": "std", "params": {"modality": "T1"}},
                    {"name": "percentile", "params": {"modality": "T2", "q": 90}},
                ]})
            ),
        )
        assert list(units.features.columns) == ["std-T1", "p90-T2"]
        labels = np.asarray(units.label_array)[tuple(field.voxel_index.T)]
        frame = pd.DataFrame(field.values, columns=list(field.feature_names))
        frame["sv"] = labels
        grouped = frame[frame["sv"] > 0].groupby("sv")
        np.testing.assert_allclose(
            grouped.std()["T1"].to_numpy(), units.features["std-T1"].to_numpy()
        )
        np.testing.assert_allclose(
            grouped.quantile(0.9)["T2"].to_numpy(),
            units.features["p90-T2"].to_numpy(),
        )

    def test_original_source_suffix(self, two_modality_subject: Subject) -> None:
        units, _ = self._units(
            two_modality_subject,
            MeanSupervoxelFeatures(modality="T1", source="original"),
        )
        assert list(units.features.columns) == ["T1-original"]

    def test_alias_renames(self, two_modality_subject: Subject) -> None:
        units, _ = self._units(
            two_modality_subject,
            MeanSupervoxelFeatures(modality="T1", as_="coarse"),
        )
        assert list(units.features.columns) == ["coarse"]

    def test_missing_modality_column_fails(
        self, two_modality_subject: Subject
    ) -> None:
        with pytest.raises(HABITAPIError, match="no feature column"):
            self._units(
                two_modality_subject, MeanSupervoxelFeatures(modality="FLAIR")
            )

    def test_statistics_require_bound_fields_without_pipeline(
        self, two_modality_subject: Subject
    ) -> None:
        """std/percentile cannot be derived from attached means: fail loudly."""
        voxel = build_voxel_extractor(Spec("raw", {"modalities": ["T1", "T2"]}))
        field = voxel(two_modality_subject)
        partition = _kmeans()(field)
        with pytest.raises(HABITAPIError, match="bind_fields"):
            StdSupervoxelFeatures(modality="T1")(two_modality_subject, partition)
        # mean, by contrast, falls back to the attached means with identical
        # numbers by construction.
        described = MeanSupervoxelFeatures(modality="T1")(
            two_modality_subject, partition
        )
        assert list(described.features.columns) == ["T1"]

    def test_spec_folds_non_default_params(self) -> None:
        assert MeanSupervoxelFeatures().spec.params == {"source": "working"}
        assert PercentileSupervoxelFeatures(modality="T1", q=75).spec.params == {
            "source": "working",
            "modality": "T1",
            "q": 75.0,
        }


class TestHabitatTrees:
    """Habitat-level trees merge one-row-per-subject tables."""

    def _habitat_map(self, subject: Subject):
        """Two-habitat map aligned with the subject's grid."""
        from habit.contracts import HabitatMap

        volume = subject.image("T1")
        mask = subject.mask()
        labels = np.zeros(tuple(int(v) for v in volume.geometry.shape), dtype=np.int32)
        voxels = np.argwhere(np.asarray(mask.load()) > 0)
        for position, (z, y, x) in enumerate(voxels):
            labels[z, y, x] = 1 if position < len(voxels) // 2 else 2
        return HabitatMap(
            subject_id=subject.subject_id,
            label_array=labels,
            geometry=volume.geometry,
            model_id="tree-test",
            habitat_ids=(1, 2),
            provenance=provenance(),
        )

    def test_concat_volume_and_traditional(
        self, two_modality_subject: Subject
    ) -> None:
        habitat_map = self._habitat_map(two_modality_subject)
        tree = build_habitat_extractor(
            Spec("concat", {"children": [
                {"name": "volume", "params": {}},
                {"name": "traditional", "params": {"modality": "T1"}},
            ]})
        )
        table = tree(two_modality_subject, habitat_map)
        assert list(table.frame["subject"]) == [two_modality_subject.subject_id]
        columns = list(table.feature_columns)
        assert any("volume_fraction" in column for column in columns)
        assert any(column.endswith("_of_T1") for column in columns)

    def test_traditional_alias_suffix(self, two_modality_subject: Subject) -> None:
        habitat_map = self._habitat_map(two_modality_subject)
        leaf = build_habitat_extractor(
            Spec("traditional", {"modality": "T1", "as_": "coarse"})
        )
        table = leaf(two_modality_subject, habitat_map)
        assert all(
            column.endswith("_of_coarse") for column in table.feature_columns
        )
