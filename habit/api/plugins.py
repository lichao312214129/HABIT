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
"""Public discovery and inspection API for HABIT extension plugins."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from importlib import metadata
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple, Type, cast

from pydantic import BaseModel

from habit.exceptions import HABITAPIError
from habit.utils.deprecation import HabitDeprecationWarning, build_deprecation_message

logger = logging.getLogger("habit.api.plugins")

__all__ = [
    "PluginInfo",
    "PluginLoadReport",
    "list_plugins",
    "get_plugin_info",
    "get_param_schema",
    "load_plugins",
]

#: Plugin domain -> entry point group. The v0.1 plural domains are kept for
#: backward compatibility and resolve to the v0.1 factories; the v1.0 domains
#: are ``snake_case`` of their protocol class, singular
#: (developer/api_upgrade/08 §4), and resolve to the L3 domain registries.
#: ``preprocessor`` is the singular alias of the v0.1 image-preprocessor
#: family (the v1 architecture keeps image preprocessing in the adapters
#: layer), while ``table_preprocessor`` / ``feature_selector`` /
#: ``classifier`` / ``metric`` name the v1.0 table-ML domains.
_ENTRY_POINT_GROUPS: Mapping[str, str] = {
    # v0.1 domains (legacy, kept working).
    "preprocessors": "habit.preprocessors",
    "radiomics_backends": "habit.radiomics_backends",
    "feature_extractors": "habit.feature_extractors",
    "habitat_features": "habit.habitat_features",
    "models": "habit.models",
    "metrics": "habit.metrics",
    # v1.0 domains: snake_case(protocol name), singular.
    "voxel_feature_extractor": "habit.voxel_feature_extractor",
    "supervoxelizer": "habit.supervoxelizer",
    "supervoxel_feature_extractor": "habit.supervoxel_feature_extractor",
    "feature_preprocessing_method": "habit.feature_preprocessing_method",
    "habitat_model_fitter": "habit.habitat_model_fitter",
    "habitat_assigner": "habit.habitat_assigner",
    "habitat_feature_extractor": "habit.habitat_feature_extractor",
    "preprocessor": "habit.preprocessor",
    "table_preprocessor": "habit.table_preprocessor",
    "classifier": "habit.classifier",
    "feature_selector": "habit.feature_selector",
    "metric": "habit.metric",
}
#: v0.1 plural plugin domains that alias the v1.0 singular L3 registries.
_LEGACY_DOMAIN_ALIASES: Mapping[str, str] = {
    "models": "classifier",
    "metrics": "metric",
}
_LOADED_ENTRY_POINTS: set[Tuple[str, str, str]] = set()


@dataclass(frozen=True)
class PluginInfo:
    """Describe one component registered in a HABIT plugin domain."""

    name: str
    domain: str
    implementation: str
    params_schema: Optional[str] = None
    provider: str = "built-in"


@dataclass(frozen=True)
class PluginLoadReport:
    """Report loaded external entry points and non-fatal discovery failures."""

    loaded: Tuple[str, ...] = ()
    failures: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Expose an immutable failure snapshot to plugin-management callers."""
        object.__setattr__(self, "loaded", tuple(self.loaded))
        object.__setattr__(self, "failures", MappingProxyType(dict(self.failures)))


def _warn_legacy_plugin_domain(domain: str) -> None:
    """Emit a deprecation warning for a v0.1 plural plugin-domain alias."""
    alternative = _LEGACY_DOMAIN_ALIASES[domain]
    warnings.warn(
        build_deprecation_message(
            f"plugin domain '{domain}'",
            "1.0.0",
            alternative=f"domain='{alternative}'",
            removed_in="1.2.0",
        ),
        HabitDeprecationWarning,
        stacklevel=3,
    )


def _legacy_registry_for_domain(domain: str) -> Type[Any]:
    """Resolve a v0.1 plural ML plugin domain to its legacy core registry."""
    if domain == "models":
        from habit.core.machine_learning.models.factory import ModelFactory

        return cast(Type[Any], ModelFactory)
    if domain == "metrics":
        from habit.core.machine_learning.evaluation.metrics import MetricRegistry

        return cast(Type[Any], MetricRegistry)
    raise HABITAPIError(
        f"Domain '{domain}' is not a legacy alias. Available legacy aliases: "
        f"{sorted(_LEGACY_DOMAIN_ALIASES)}."
    )


def _registry_for_plugin_name(domain: str, name: str) -> Type[Any]:
    """Pick the registry backing one plugin name, honoring legacy aliases."""
    if domain not in _LEGACY_DOMAIN_ALIASES:
        return _registry_for_domain(domain)
    v1_registry = _registry_for_domain(_LEGACY_DOMAIN_ALIASES[domain])
    if name in v1_registry.available():
        return v1_registry
    return _legacy_registry_for_domain(domain)


def _registry_for_domain(domain: str) -> Type[Any]:
    """Resolve a public plugin domain to its HABIT registry lazily."""
    if domain in _LEGACY_DOMAIN_ALIASES:
        raise HABITAPIError(
            f"Legacy plugin domain '{domain}' must be resolved per plugin name. "
            f"Use domain='{_LEGACY_DOMAIN_ALIASES[domain]}' instead."
        )
    if domain == "preprocessors" or domain == "preprocessor":
        from habit.core.preprocessing import PreprocessorFactory

        return cast(Type[Any], PreprocessorFactory)
    if domain == "feature_extractors":
        from habit.core.habitat_analysis.clustering_features.base_extractor import (
            FeatureExtractorRegistry,
        )

        return cast(Type[Any], FeatureExtractorRegistry)
    if domain == "habitat_features":
        from habit.core.habitat_analysis.feature_registry import (
            HabitatFeatureFactory,
            bootstrap_optional_features,
        )

        bootstrap_optional_features()
        return cast(Type[Any], HabitatFeatureFactory)
    if domain == "classifier":
        from habit.domain.classification import ClassifierRegistry

        return cast(Type[Any], ClassifierRegistry)
    if domain == "metric":
        from habit.domain.evaluation import MetricRegistry

        return cast(Type[Any], MetricRegistry)
    if domain == "table_preprocessor":
        from habit.domain.table_preprocessing import TablePreprocessorRegistry

        return cast(Type[Any], TablePreprocessorRegistry)
    if domain == "feature_selector":
        from habit.domain.feature_selection import FeatureSelectorRegistry

        return cast(Type[Any], FeatureSelectorRegistry)
    if domain == "voxel_feature_extractor":
        from habit.domain.voxel_features import VoxelFeatureExtractorRegistry

        return cast(Type[Any], VoxelFeatureExtractorRegistry)
    if domain == "supervoxelizer":
        from habit.domain.supervoxel import SupervoxelizerRegistry

        return cast(Type[Any], SupervoxelizerRegistry)
    if domain == "supervoxel_feature_extractor":
        from habit.domain.supervoxel_features import (
            SupervoxelFeatureExtractorRegistry,
        )

        return cast(Type[Any], SupervoxelFeatureExtractorRegistry)
    if domain == "feature_preprocessing_method":
        from habit.domain.feature_preprocessing import (
            FeaturePreprocessingMethodRegistry,
        )

        return cast(Type[Any], FeaturePreprocessingMethodRegistry)
    if domain == "habitat_model_fitter":
        from habit.domain.habitat_model import HabitatModelFitterRegistry

        return cast(Type[Any], HabitatModelFitterRegistry)
    if domain == "habitat_assigner":
        from habit.domain.assignment import HabitatAssignerRegistry

        return cast(Type[Any], HabitatAssignerRegistry)
    if domain == "habitat_feature_extractor":
        from habit.domain.habitat_features import HabitatFeatureExtractorRegistry

        return cast(Type[Any], HabitatFeatureExtractorRegistry)
    if domain == "radiomics_backends":
        raise HABITAPIError(
            "Radiomics backends are not yet registry-backed. Use "
            "habit.radiomics.extract_features(..., backend='pyradiomics')."
        )
    raise HABITAPIError(
        f"Unknown plugin domain '{domain}'. Available domains: "
        f"{sorted(_ENTRY_POINT_GROUPS)}."
    )


def _implementation_name(payload: Any) -> str:
    """Return a stable fully qualified implementation name for metadata output."""
    module = getattr(payload, "__module__", type(payload).__module__)
    qualname = getattr(payload, "__qualname__", type(payload).__qualname__)
    return f"{module}.{qualname}"


def _schema_name(schema: Optional[Type[Any]]) -> Optional[str]:
    """Return a fully qualified Pydantic-schema name when one is registered."""
    if schema is None:
        return None
    return f"{schema.__module__}.{schema.__qualname__}"


def list_plugins(domain: Optional[str] = None) -> Tuple[PluginInfo, ...]:
    """List registered HABIT extension components without exposing core registries.

    Args:
        domain: Optional plugin domain. Omit it to enumerate all supported
            registry-backed domains.

    Returns:
        Deterministically ordered plugin metadata.
    """
    domains = (
        (domain,)
        if domain is not None
        else tuple(key for key in _ENTRY_POINT_GROUPS if key != "radiomics_backends")
    )
    infos: list[PluginInfo] = []
    for current_domain in domains:
        if current_domain in _LEGACY_DOMAIN_ALIASES:
            _warn_legacy_plugin_domain(current_domain)
            names = sorted(_legacy_registry_for_domain(current_domain).available())
        else:
            names = sorted(_registry_for_domain(current_domain).available())
        for name in names:
            registry = _registry_for_plugin_name(current_domain, name)
            payload = registry.get(name)
            if payload is None:
                continue
            schema = registry.get_params_model(name)
            infos.append(
                PluginInfo(
                    name=name,
                    domain=current_domain,
                    implementation=_implementation_name(payload),
                    params_schema=_schema_name(schema),
                )
            )
    return tuple(sorted(infos, key=lambda info: (info.domain, info.name)))


def get_plugin_info(name: str, domain: str) -> PluginInfo:
    """Return metadata for one registered component or raise a clear API error."""
    for info in list_plugins(domain):
        if info.name == name:
            return info
    raise HABITAPIError(
        f"Plugin '{name}' is not registered in domain '{domain}'. "
        f"Available: {[info.name for info in list_plugins(domain)]}."
    )


def get_param_schema(name: str, domain: str) -> Optional[Type[BaseModel]]:
    """Return the Pydantic parameter schema associated with a plugin, if any."""
    if domain in _LEGACY_DOMAIN_ALIASES:
        _warn_legacy_plugin_domain(domain)
    registry = _registry_for_plugin_name(domain, name)
    schema = registry.get_params_model(name)
    if schema is None:
        return None
    if not issubclass(schema, BaseModel):
        raise HABITAPIError(
            f"Plugin '{name}' in domain '{domain}' has a non-Pydantic parameter "
            f"schema: {_schema_name(schema)}."
        )
    return cast(Type[BaseModel], schema)


def _entry_points_for(group: str) -> Tuple[metadata.EntryPoint, ...]:
    """Return compatible entry-point selections for Python 3.10 and newer."""
    entry_points = metadata.entry_points()
    if hasattr(entry_points, "select"):
        return tuple(entry_points.select(group=group))
    return tuple(entry_points.get(group, ()))


def load_plugins(*, strict: bool = False) -> PluginLoadReport:
    """Load external HABIT plugins declared through standard Python entry points.

    An entry point may resolve to a module (whose registration decorators execute
    during import) or to a zero-argument callable that performs registration.
    Discovery is idempotent for an installed distribution entry point.

    Args:
        strict: Raise the first plugin loading error instead of returning it in
            :attr:`PluginLoadReport.failures`.

    Returns:
        Loaded entry-point identifiers and any non-fatal load errors.
    """
    loaded: list[str] = []
    failures: dict[str, str] = {}
    for domain, group in _ENTRY_POINT_GROUPS.items():
        for entry_point in _entry_points_for(group):
            identifier = (group, entry_point.name, entry_point.value)
            display_name = f"{domain}:{entry_point.name}"
            if identifier in _LOADED_ENTRY_POINTS:
                continue
            try:
                target = entry_point.load()
                if callable(target):
                    target()
            except Exception as exc:
                if strict:
                    raise
                message = f"{type(exc).__name__}: {exc}"
                failures[display_name] = message
                logger.warning(
                    "Failed to load HABIT plugin entry point %s: %s",
                    display_name,
                    message,
                )
                continue
            _LOADED_ENTRY_POINTS.add(identifier)
            loaded.append(display_name)
    return PluginLoadReport(loaded=tuple(loaded), failures=failures)
