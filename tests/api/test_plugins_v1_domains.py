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

from typing import Any

import pytest

from habit.api.exceptions import HABITAPIError
from habit.utils.deprecation import HabitDeprecationWarning
from habit.api.plugins import (
    format_plugin_catalog_rst,
    get_param_schema,
    get_plugin_info,
    list_plugins,
    load_plugins,
    plugin_catalog,
)

#: Built-in habitat feature names exposed through the legacy plural domain.
_HABITAT_FEATURES_LEGACY = frozenset(
    {
        "each_habitat",
        "ith_score",
        "msi",
        "non_radiomics",
        "traditional",
        "whole_habitat",
    }
)
#: domain -> built-in implementation names expected after a bare import.
_V1_BUILTINS = {
    "voxel_feature_extractor": {
        "raw",
        "voxel_radiomics",
        "kinetic",
        "local_entropy",
        "concat",
        "expression",
    },
    "supervoxelizer": {"slic", "kmeans", "gmm"},
    "supervoxel_feature_extractor": {
        "mean_voxel_features",
        "supervoxel_radiomics",
        "mean",
        "std",
        "percentile",
    },
    "habitat_model_fitter": {"kmeans", "gmm"},
    "habitat_assigner": {"nearest_centroid"},
    "image_perturbation": {
        "gaussian_noise",
        "translation",
        "rotation",
        "rigid",
        "bspline_deform",
    },
    "combiner": {
        "concat",
        "weighted_concat",
        "average",
        "ratio",
        "difference",
        "kinetic",
        "expression",
    },
    "habitat_feature_extractor": {
        "msi",
        "ith_score",
        "volume",
        "non_radiomics",
        "traditional",
        "whole_habitat",
        "each_habitat",
        "graph",
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
        "precise_correlation_filter",
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
        "icc_precomputed",
        "mrmr",
        "univariate_cox",
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
    with pytest.raises(HABITAPIError, match="Available"):
        get_plugin_info("watershed", "supervoxelizer")


@pytest.mark.unit
def test_preprocessor_v1_domain_lists_image_steps() -> None:
    """Singular preprocessor is the v1 image-volume registry."""
    names = {info.name for info in list_plugins("preprocessor")}
    assert {
        "resample",
        "reorientation",
        "n4_correction",
        "zscore_normalization",
        "histogram_standardization",
        "adaptive_histogram_equalization",
        "registration",
    } <= names
    zscore = get_plugin_info("zscore_normalization", "preprocessor")
    assert zscore.implementation.startswith("habit.domain.")


@pytest.mark.unit
def test_preprocessor_alias_still_lists_legacy_factory() -> None:
    """Plural preprocessors keeps the v0.1 factory listing."""
    with pytest.warns(HabitDeprecationWarning, match="preprocessors"):
        plural = {info.name for info in list_plugins("preprocessors")}
    assert "resample" in plural
    assert "zscore_normalization" in plural


@pytest.mark.unit
def test_habitat_features_legacy_alias_lists_core_only() -> None:
    """Legacy habitat_features omits v1-only plugins such as volume."""
    with pytest.warns(HabitDeprecationWarning, match="habitat_features"):
        legacy = {info.name for info in list_plugins("habitat_features")}
    v1 = {info.name for info in list_plugins("habitat_feature_extractor")}
    assert _HABITAT_FEATURES_LEGACY <= legacy
    assert "volume" not in legacy
    assert "volume" in v1


@pytest.mark.unit
def test_habitat_features_legacy_alias_delegates_per_name() -> None:
    """Shared built-in names resolve to v1; legacy-only names stay on core."""
    with pytest.warns(HabitDeprecationWarning, match="habitat_features"):
        builtins = [
            info
            for info in list_plugins("habitat_features")
            if info.name in _HABITAT_FEATURES_LEGACY
        ]
    for info in builtins:
        assert info.implementation.startswith(
            ("habit.domain.", "habit.compat.engines.")
        ), info
    msi = get_plugin_info("msi", "habitat_features")
    assert msi.implementation.startswith("habit.domain.")


@pytest.mark.unit
def test_table_ml_singular_domains_resolve_to_v1_registries() -> None:
    """classifier/metric singular domains are the v1 L3 registries, not v0.1."""
    for domain in ("classifier", "metric", "table_preprocessor", "feature_selector"):
        for info in list_plugins(domain):
            assert info.implementation.startswith("habit.domain.")
    # Legacy plural domains delegate per name: v1 wins when the name exists
    # in the L3 registry, otherwise the v0.1 core factory is used.
    for domain in ("models", "metrics"):
        for info in list_plugins(domain):
            assert info.implementation.startswith(
                ("habit.domain.", "habit.compat.engines.")
            ), info


@pytest.mark.unit
def test_unknown_domain_still_raises() -> None:
    """Unknown domains fail with the documented public error."""
    with pytest.raises(HABITAPIError):
        list_plugins("clustering")


@pytest.mark.unit
def test_plugin_catalog_reads_params_model() -> None:
    """Catalog rows come from params_model, not a hand-copied table."""
    rows = plugin_catalog("table_preprocessor")
    names = {row.name for row in rows}
    assert "minmax" in names
    minmax = next(row for row in rows if row.name == "minmax")
    schema = get_param_schema("minmax", "table_preprocessor")
    assert schema is not None
    assert set(minmax.required_params) == {
        name for name, field in schema.model_fields.items() if field.is_required()
    }
    assert set(minmax.optional_params) == {
        name
        for name, field in schema.model_fields.items()
        if not field.is_required()
    }
    assert minmax.spec_example.startswith('Spec("minmax"')
    assert 'Registry.create("minmax"' in minmax.create_example
    assert minmax.params
    across = next(param for param in minmax.params if param.name == "across_features")
    assert across.required is False
    assert across.default in {"False", "false"}
    rst = format_plugin_catalog_rst("table_preprocessor")
    assert "minmax" in rst
    assert 'Spec("minmax"' in rst
    assert "across_features" in rst


@pytest.mark.unit
def test_plugin_catalog_explains_kmeans_spec_params() -> None:
    """Habitat kmeans catalog rows include allowed validation values."""
    rows = plugin_catalog("habitat_model_fitter")
    kmeans = next(row for row in rows if row.name == "kmeans")
    names = {param.name for param in kmeans.params}
    assert {"n_habitats", "validation", "min_habitats", "max_habitats"} <= names
    validation = next(param for param in kmeans.params if param.name == "validation")
    assert validation.default in {'"elbow"', "'elbow'"}
    assert "elbow" in validation.description
    rst = format_plugin_catalog_rst("habitat_model_fitter")
    assert "elbow" in rst
    assert "n_habitats" in rst


@pytest.mark.unit
def test_unknown_registry_name_lists_available() -> None:
    """Registry.create on a bad name lists the domain's registered names."""
    from habit.domain.table_preprocessing import TablePreprocessorRegistry
    from habit.exceptions import ComponentNotFoundError

    with pytest.raises(ComponentNotFoundError, match="Available") as exc_info:
        TablePreprocessorRegistry.create("not_a_real_preprocessor")
    message = str(exc_info.value)
    assert "minmax" in message
    assert "list_plugins" in message
    assert "table_preprocessor" in message


@pytest.mark.unit
def test_load_plugins_scans_old_and_new_groups() -> None:
    """Plugin discovery covers the v0.1 plural and v1.0 singular groups."""
    report = load_plugins()
    assert not report.failures


@pytest.mark.unit
def test_load_plugins_logs_nonfatal_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Broken entry points are recorded and logged without aborting discovery."""
    from habit.api import plugins

    class BrokenEntryPoint:
        """Minimal entry point whose load() always raises."""

        name = "broken"
        value = "broken_plugin:register"

        @staticmethod
        def load() -> None:
            raise RuntimeError("boom")

    logged: list[tuple[Any, ...]] = []

    def _capture_warning(msg: str, *args: Any) -> None:
        logged.append((msg, args))

    monkeypatch.setattr(plugins, "_ENTRY_POINT_GROUPS", {"models": "habit.models"})
    monkeypatch.setattr(
        plugins,
        "_entry_points_for",
        lambda group: (BrokenEntryPoint(),),
    )
    monkeypatch.setattr(plugins.logger, "warning", _capture_warning)
    plugins._LOADED_ENTRY_POINTS.clear()

    report = load_plugins()

    assert report.failures == {"models:broken": "RuntimeError: boom"}
    assert logged
    assert logged[0][0].startswith("Failed to load HABIT plugin entry point")
