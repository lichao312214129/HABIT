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
"""Thin re-exports of remaining v0.1 engines (deliberate debt).

The public API and recipes must not import ``habit.core.*`` directly once the
v1.0 refactor completes. During the transition this module is the single
compatibility facade; concrete workflow runners and registry gates live in
``habit.compat.*`` submodules so this file stays free of ``habit.core`` imports.

Deleting ``habit.core`` is blocked until every compat runner has a v1 recipe
or domain replacement and the fast/full golden gates stay green.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Type

__all__ = [
    "apply_habitat_cli_overrides_core",
    "apply_ml_mode_override_core",
    "batch_process_test_retest_files",
    "find_habitat_test_retest_mapping",
    "get_habitat_configurator_class",
    "get_legacy_feature_extractor_registry",
    "get_legacy_habitat_feature_factory",
    "get_legacy_metric_registry",
    "get_legacy_model_factory",
    "get_legacy_preprocessor_factory",
    "get_ml_pipeline_builder_class",
    "load_feature_extraction_config_from_file",
    "parse_feature_extraction_config",
    "run_dicom_sort",
    "run_feature_extraction_from_config",
    "run_habitat_analysis_from_config",
    "run_icc_analysis_from_config",
    "run_kfold_from_config",
    "run_ml_from_config",
    "run_model_comparison_from_config",
    "run_preprocess_from_config",
    "run_radiomics_from_config",
]


def get_legacy_model_factory() -> Type[Any]:
    """Return the v1 classifier registry (v0.1 ``ModelFactory`` alias surface)."""
    from habit.compat.plugin_registries import get_legacy_model_factory as _get

    return _get()


def get_legacy_metric_registry() -> Type[Any]:
    """Return the v1 metric registry (v0.1 ``MetricRegistry`` alias surface)."""
    from habit.compat.plugin_registries import get_legacy_metric_registry as _get

    return _get()


def get_legacy_preprocessor_factory() -> Type[Any]:
    """Return the v0.1 ``PreprocessorFactory`` registry class."""
    from habit.compat.plugin_registries import get_legacy_preprocessor_factory as _get

    return _get()


def get_legacy_habitat_feature_factory() -> Type[Any]:
    """Return the v0.1 ``HabitatFeatureFactory`` after optional bootstrap."""
    from habit.compat.plugin_registries import get_legacy_habitat_feature_factory as _get

    return _get()


def get_legacy_feature_extractor_registry() -> Type[Any]:
    """Return the v0.1 ``FeatureExtractorRegistry`` class."""
    from habit.compat.plugin_registries import (
        get_legacy_feature_extractor_registry as _get,
    )

    return _get()


def get_habitat_configurator_class() -> Type[Any]:
    """Return the v0.1 ``HabitatConfigurator`` class."""
    from habit.compat.estimator_bridge import get_habitat_configurator_class as _get

    return _get()


def get_ml_pipeline_builder_class() -> Type[Any]:
    """Return the v0.1 ``PipelineBuilder`` class."""
    from habit.compat.estimator_bridge import get_ml_pipeline_builder_class as _get

    return _get()


def apply_habitat_cli_overrides_core(
    config: Any,
    *,
    mode: Optional[str] = None,
    pipeline_path: Optional[str] = None,
    debug: bool = False,
    resume: bool = False,
) -> Any:
    """Apply CLI-style overrides onto a loaded habitat config."""
    if debug:
        config.debug = True
    if mode:
        config.run_mode = mode
    if pipeline_path:
        config.pipeline_path = pipeline_path
    if resume:
        config.resume = True
    return config


def apply_ml_mode_override_core(config: Any, mode: Any) -> Any:
    """Apply CLI mode override through the v0.1 ML schema."""
    from habit.schemas.workflows.ml import MLConfig

    if mode is None or config.run_mode == mode:
        return config
    return MLConfig.model_validate({**config.model_dump(), "run_mode": mode})


def run_habitat_analysis_from_config(config: Any, **kwargs: Any) -> Any:
    """Run habitat train/predict through the L1 habitat runner."""
    from habit.compat.habitat_runner import run_habitat_analysis_from_config as _run

    return _run(config, **kwargs)


def run_feature_extraction_from_config(config: Any, **kwargs: Any) -> Any:
    """Run habitat feature extraction through the L1 feature runner."""
    from habit.compat.feature_extraction_runner import (
        run_feature_extraction_from_config as _run,
    )

    return _run(config, **kwargs)


def run_radiomics_from_config(config: Any, **kwargs: Any) -> Any:
    """Run standalone radiomics through the L1 feature runner."""
    from habit.compat.feature_extraction_runner import run_radiomics_from_config as _run

    return _run(config, **kwargs)


def load_feature_extraction_config_from_file(
    config_path: Any,
) -> Tuple[Any, Dict[str, Any]]:
    """Load feature-extraction YAML including plugin sidecars."""
    from habit.compat.feature_extraction_loader import (
        load_feature_extraction_config_from_file as _load,
    )

    return _load(config_path)


def parse_feature_extraction_config(
    config: Any,
) -> Tuple[Any, Dict[str, Any]]:
    """Parse an in-memory feature-extraction config including plugins."""
    from habit.compat.feature_extraction_loader import (
        parse_feature_extraction_config as _parse,
    )

    return _parse(config)


def run_preprocess_from_config(config: Any, **kwargs: Any) -> Any:
    """Run image preprocessing through the L1 preprocess runner."""
    from habit.compat.preprocess_runner import run_preprocess_from_config as _run

    return _run(config, **kwargs)


def run_dicom_sort(config: Any, **kwargs: Any) -> Any:
    """Run DICOM sorting through the L1 adapter runner."""
    from habit.compat.dicom_sort_runner import run_dicom_sort as _run

    return _run(config, **kwargs)


def run_ml_from_config(config: Any, **kwargs: Any) -> Any:
    """Run ML hold-out train/predict through the L1 ML runner."""
    from habit.compat.ml_runner import run_ml_from_config as _run

    return _run(config, **kwargs)


def run_kfold_from_config(config: Any, **kwargs: Any) -> Any:
    """Run ML K-fold cross-validation through the L1 ML runner."""
    from habit.compat.ml_runner import run_kfold_from_config as _run

    return _run(config, **kwargs)


def run_model_comparison_from_config(config: Any, **kwargs: Any) -> Any:
    """Run model comparison through the L1 ML runner."""
    from habit.compat.ml_runner import run_model_comparison_from_config as _run

    return _run(config, **kwargs)


def run_icc_analysis_from_config(config: Any) -> Any:
    """Run ICC reliability analysis through the L1 adapter runner."""
    from habit.compat.icc_runner import run_icc_analysis_from_config as _run

    return _run(config)


def find_habitat_test_retest_mapping(*args: Any, **kwargs: Any) -> Any:
    """Delegate test-retest label mapping discovery to the L1 adapter."""
    from habit.compat.test_retest_mapper import find_habitat_mapping

    return find_habitat_mapping(*args, **kwargs)


def batch_process_test_retest_files(*args: Any, **kwargs: Any) -> Any:
    """Delegate test-retest image remapping to the L1 adapter."""
    from habit.compat.test_retest_mapper import batch_process_files

    return batch_process_files(*args, **kwargs)


def PipelineBuilder() -> Any:
    """Backward-compatible alias; prefer :func:`get_ml_pipeline_builder_class`."""
    return get_ml_pipeline_builder_class()
