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
"""Contract tests for the strict feature-tree expression parser.

The parser is the compact spelling of a spec tree; these tests pin the
grammar rules (quoted modalities, explicit kwargs, nested children), the
strictness guarantees (bare tokens are rejected), and the dual-form
invariant (string spelling and structured spelling fingerprint equally).
"""

from __future__ import annotations

import pytest

from habit.exceptions import HABITAPIError
from habit.spec import HabitatSpec, Spec, coerce_spec, parse_feature_expression
from habit.spec.legacy import LegacyConfigAdapter


class TestLeafForms:
    """Leaves take quoted modalities and explicit key=value parameters."""

    def test_single_modality_maps_to_modality_param(self) -> None:
        spec = parse_feature_expression('raw("T1")')
        assert spec.name == "raw"
        assert spec.params == {"modality": "T1"}

    def test_multiple_modalities_map_to_list(self) -> None:
        spec = parse_feature_expression('raw("T1", "T2")')
        assert spec.params == {"modalities": ["T1", "T2"]}

    def test_bare_name_is_a_parameterless_leaf(self) -> None:
        spec = parse_feature_expression("volume")
        assert spec.name == "volume"
        assert spec.params == {}

    def test_kwargs_numbers_booleans_lists(self) -> None:
        spec = parse_feature_expression(
            'percentile("T1", q=90, as_="hi", flags=[true, false])'
        )
        assert spec.params["q"] == 90
        assert spec.params["as_"] == "hi"
        assert spec.params["flags"] == [True, False]

    def test_float_and_negative_numbers(self) -> None:
        spec = parse_feature_expression('local_entropy("T1", radius=1.5)')
        assert spec.params["radius"] == 1.5


class TestCombinerNodes:
    """Nested calls become children; quoted strings become raw children."""

    def test_children_nest_under_params(self) -> None:
        spec = parse_feature_expression('concat(raw("T1"), raw("T2"))')
        assert spec.name == "concat"
        assert spec.params["children"] == [
            {"name": "raw", "params": {"modality": "T1"}},
            {"name": "raw", "params": {"modality": "T2"}},
        ]

    def test_quoted_strings_become_implicit_raw_children(self) -> None:
        spec = parse_feature_expression('concat("T1", raw("T2"))')
        assert spec.params["children"] == [
            {"name": "raw", "params": {"modality": "T1"}},
            {"name": "raw", "params": {"modality": "T2"}},
        ]

    def test_nested_combiner_with_alias(self) -> None:
        spec = parse_feature_expression(
            'concat(raw("T1"), ratio(raw("T1"), raw("T2"), as_="t1_over_t2"))'
        )
        ratio = spec.params["children"][1]
        assert ratio["name"] == "ratio"
        assert ratio["params"]["as_"] == "t1_over_t2"
        assert len(ratio["params"]["children"]) == 2

    def test_weights_list_keyed_by_child_source_labels(self) -> None:
        spec = parse_feature_expression(
            'weighted_concat(raw("T1", as_="coarse"), raw("T2"), weights=[2.0, 1.0])'
        )
        assert spec.params["weights"] == {"coarse": 2.0, "T2": 1.0}

    def test_weights_list_length_mismatch_fails(self) -> None:
        with pytest.raises(HABITAPIError, match="weights list"):
            parse_feature_expression(
                'weighted_concat(raw("T1"), raw("T2"), weights=[2.0])'
            )


class TestStrictness:
    """Ambiguous v0-style syntax must fail loudly, never guess."""

    @pytest.mark.parametrize(
        "expression",
        [
            "concat(raw(T1), raw(T2))",  # bare modality token
            'raw(T1)',  # bare token inside a leaf
            'concat(raw("T1"), timestamps)',  # bare parameter reference
        ],
    )
    def test_bare_identifiers_are_rejected(self, expression: str) -> None:
        with pytest.raises(HABITAPIError, match="[Bb]are"):
            parse_feature_expression(expression)

    def test_unterminated_string_fails(self) -> None:
        with pytest.raises(HABITAPIError, match="Unterminated"):
            parse_feature_expression('raw("T1)')

    def test_trailing_garbage_fails(self) -> None:
        with pytest.raises(HABITAPIError, match="Unexpected"):
            parse_feature_expression('raw("T1") extra')

    def test_duplicate_kwarg_fails(self) -> None:
        with pytest.raises(HABITAPIError, match="duplicate"):
            parse_feature_expression('raw("T1", q=1, q=2)')

    def test_empty_expression_fails(self) -> None:
        with pytest.raises(HABITAPIError, match="Empty"):
            parse_feature_expression("   ")


class TestDualForm:
    """String and structured spellings share one canonical Spec tree."""

    def test_fingerprint_equality(self) -> None:
        from_string = HabitatSpec.from_dict(
            {
                "name": "dual",
                "voxel_feature_extractor": 'concat(raw("T1"), raw("T2"))',
                "supervoxelizer": None,
                "habitat_model_fitter": {"name": "kmeans", "params": {}},
                "habitat_assigner": {"name": "nearest_centroid", "params": {}},
            }
        )
        from_structured = HabitatSpec.from_dict(
            {
                "name": "dual",
                "voxel_feature_extractor": {
                    "name": "concat",
                    "params": {
                        "children": [
                            {"name": "raw", "params": {"modality": "T1"}},
                            {"name": "raw", "params": {"modality": "T2"}},
                        ]
                    },
                },
                "supervoxelizer": None,
                "habitat_model_fitter": {"name": "kmeans", "params": {}},
                "habitat_assigner": {"name": "nearest_centroid", "params": {}},
            }
        )
        assert from_string.fingerprint() == from_structured.fingerprint()

    def test_habitat_feature_items_accept_expressions(self) -> None:
        spec = HabitatSpec.from_dict(
            {
                "name": "dual",
                "voxel_feature_extractor": {"name": "raw", "params": {"modalities": ["T1"]}},
                "supervoxelizer": None,
                "habitat_model_fitter": {"name": "kmeans", "params": {}},
                "habitat_assigner": {"name": "nearest_centroid", "params": {}},
                "habitat_features": ["volume", 'traditional("T1")'],
            }
        )
        assert spec.habitat_features[0] == Spec(name="volume", params={})
        assert spec.habitat_features[1].params == {"modality": "T1"}

    def test_coerce_spec_rejects_other_types(self) -> None:
        with pytest.raises(HABITAPIError, match="mapping or an expression"):
            coerce_spec(42)


class TestLegacyRouting:
    """Quoted expressions translate to trees; unquoted stay byte-identical."""

    def _base_payload(self) -> dict:
        return {
            "version": "0.1",
            "data_organization": {"modalities": ["T1", "T2"], "mask": "roi"},
            "feature_construction": {
                "voxel_level": {"method": "concat(raw(T1), raw(T2))", "params": {}},
            },
            "habitat_segmentation": {
                "mode": "two_step",
                "supervoxel": {"algorithm": "kmeans", "n_clusters": 5},
                "habitat": {"algorithm": "kmeans", "n_clusters": [2, 3]},
            },
            "random_state": 42,
        }

    def test_unquoted_expression_keeps_legacy_translation(self) -> None:
        translation = LegacyConfigAdapter().translate(
            self._base_payload(), workflow="habitat"
        )
        document = (
            translation.document
            if hasattr(translation, "document")
            else translation.spec
        )
        assert document["spec"]["voxel_feature_extractor"] == {
            "name": "raw",
            "params": {"modalities": ["T1", "T2"]},
        }

    def test_quoted_expression_translates_to_tree(self) -> None:
        payload = self._base_payload()
        payload["feature_construction"]["voxel_level"] = {
            "method": 'concat(raw("T1"), ratio(raw("T1"), raw("T2"), as_="r"))',
            "params": {},
        }
        translation = LegacyConfigAdapter().translate(payload, workflow="habitat")
        document = (
            translation.document
            if hasattr(translation, "document")
            else translation.spec
        )
        extractor = document["spec"]["voxel_feature_extractor"]
        assert extractor["name"] == "concat"
        children = extractor["params"]["children"]
        assert children[0] == {"name": "raw", "params": {"modality": "T1"}}
        assert children[1]["name"] == "ratio"
        assert children[1]["params"]["as_"] == "r"

    def test_quoted_supervoxel_expression_translates_to_tree(self) -> None:
        payload = self._base_payload()
        payload["feature_construction"]["supervoxel_level"] = {
            "method": 'concat(mean("T1"), std("T1"))',
            "params": {},
        }
        translation = LegacyConfigAdapter().translate(payload, workflow="habitat")
        document = (
            translation.document
            if hasattr(translation, "document")
            else translation.spec
        )
        extractor = document["spec"]["supervoxel_feature_extractor"]
        assert extractor["name"] == "concat"
        assert extractor["params"]["children"] == [
            {"name": "mean", "params": {"modality": "T1"}},
            {"name": "std", "params": {"modality": "T1"}},
        ]
