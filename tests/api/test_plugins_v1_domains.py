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
"""Contract tests for plugin introspection over the v1.0 singular domains."""

from __future__ import annotations

import pytest

from habit.api.exceptions import HABITAPIError
from habit.api.plugins import (
    get_param_schema,
    get_plugin_info,
    list_plugins,
    load_plugins,
)

#: domain -> built-in implementation names expected after a bare import.
_V1_BUILTINS = {
    "voxel_feature_extractor": {"raw"},
    "supervoxelizer": {"slic"},
    "habitat_model_fitter": {"kmeans", "gmm"},
    "habitat_assigner": {"nearest_centroid"},
    "habitat_feature_extractor": {"msi", "ith", "volume"},
}


@pytest.mark.unit
@pytest.mark.parametrize("domain", sorted(_V1_BUILTINS))
def test_v1_domain_lists_builtin_plugins(domain: str) -> None:
    """Every v1.0 domain exposes its built-ins through list_plugins."""
    names = {info.name for info in list_plugins(domain)}
    assert names == _V1_BUILTINS[domain]
    for info in list_plugins(domain):
        assert info.domain == domain
        assert info.implementation.startswith("habit.domain.")


@pytest.mark.unit
def test_get_plugin_info_and_param_schema_on_v1_domains() -> None:
    """Info and schema lookup follow the (name, domain) argument order."""
    info = get_plugin_info("slic", "supervoxelizer")
    assert info.name == "slic"
    schema = get_param_schema("slic", "supervoxelizer")
    assert schema is not None
    properties = schema.model_json_schema()["properties"]
    assert "n_supervoxels" in properties
    kmeans_schema = get_param_schema("kmeans", "habitat_model_fitter")
    assert kmeans_schema is not None
    assert "n_habitats" in kmeans_schema.model_json_schema()["properties"]
    with pytest.raises(HABITAPIError):
        get_plugin_info("watershed", "supervoxelizer")


@pytest.mark.unit
def test_v0_1_families_resolve_under_v1_singular_aliases() -> None:
    """The v1.0 singular aliases resolve to the same registries as v0.1."""
    plural = {info.name for info in list_plugins("preprocessors")}
    singular = {info.name for info in list_plugins("preprocessor")}
    assert plural == singular
    plural_models = {info.name for info in list_plugins("models")}
    singular_models = {info.name for info in list_plugins("classifier")}
    assert plural_models == singular_models
    plural_metrics = {info.name for info in list_plugins("metrics")}
    singular_metrics = {info.name for info in list_plugins("metric")}
    assert plural_metrics == singular_metrics


@pytest.mark.unit
def test_table_preprocessor_and_feature_selector_domains() -> None:
    """The remaining v1.0 domains resolve to their v0.1 registries."""
    table_preprocessor_names = {
        info.name for info in list_plugins("table_preprocessor")
    }
    assert table_preprocessor_names  # v0.1 ships built-in methods
    selector_names = {info.name for info in list_plugins("feature_selector")}
    assert selector_names  # v0.1 ships built-in selectors


@pytest.mark.unit
def test_unknown_domain_still_raises() -> None:
    """Unknown domains fail with the documented public error."""
    with pytest.raises(HABITAPIError):
        list_plugins("clustering")


@pytest.mark.unit
def test_load_plugins_scans_old_and_new_groups() -> None:
    """Plugin discovery covers the v0.1 plural and v1.0 singular groups."""
    report = load_plugins()
    assert not report.failures
