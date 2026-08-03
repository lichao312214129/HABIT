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
    "habitat_feature_extractor": {
        "msi",
        "ith_score",
        "volume",
        "non_radiomics",
        "traditional",
        "whole_habitat",
        "each_habitat",
    },
    "table_preprocessor": {
        "minmax",
        "zscore",
        "robust",
        "binning",
        "winsorize",
        "log",
        "variance_filter",
        "correlation_filter",
    },
    "feature_selector": {
        "variance",
        "correlation",
        "vif",
        "anova",
        "chi2",
        "statistical_test",
        "univariate_logistic",
        "stepwise",
        "rfecv",
        "lasso",
        "icc",
        "mrmr",
    },
    "classifier": {
        "DecisionTree",
        "KNN",
        "SVM",
        "SVC",
        "MLP",
        "LogisticRegression",
        "RandomForest",
        "GradientBoosting",
        "XGBoost",
        "AdaBoost",
        "GaussianNB",
        "MultinomialNB",
        "BernoulliNB",
        "AutoGluonTabular",
    },
    "metric": {
        "accuracy",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "f1_score",
        "auc",
        "hosmer_lemeshow_p_value",
        "spiegelhalter_z_p_value",
    },
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
def test_preprocessor_alias_resolves_to_the_v0_1_registry() -> None:
    """Image preprocessing has no v1 domain yet: the alias stays with v0.1."""
    plural = {info.name for info in list_plugins("preprocessors")}
    singular = {info.name for info in list_plugins("preprocessor")}
    assert plural == singular


@pytest.mark.unit
def test_table_ml_singular_domains_resolve_to_v1_registries() -> None:
    """classifier/metric singular domains are the v1 L3 registries, not v0.1."""
    for domain in ("classifier", "metric", "table_preprocessor", "feature_selector"):
        for info in list_plugins(domain):
            assert info.implementation.startswith("habit.domain.")
    # The v0.1 plural domains still resolve to the v0.1 factories.
    for domain in ("models", "metrics"):
        for info in list_plugins(domain):
            assert info.implementation.startswith("habit.core.")


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
