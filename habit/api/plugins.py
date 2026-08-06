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
    "create_ml_model",
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
    "combiner": "habit.combiner",
    "image_perturbation": "habit.image_perturbation",
    "preprocessor": "habit.preprocessor",
    "table_preprocessor": "habit.table_preprocessor",
    "classifier": "habit.classifier",
    "feature_selector": "habit.feature_selector",
    "metric": "habit.metric",
}
#: v0.1 plural plugin domains that alias the v1.0 singular L3 registries.
#: Per-name resolution prefers the v1 registry when the name exists there,
#: otherwise falls back to the v0.1 core factory (see ``_registry_for_plugin_name``).
_LEGACY_DOMAIN_ALIASES: Mapping[str, str] = {
    "models": "classifier",
    "metrics": "metric",
    "preprocessors": "preprocessor",
    "habitat_features": "habitat_feature_extractor",
}
#: v1.0 domains consulted before the legacy factory when resolving the
#: one-to-many ``feature_extractors`` domain (see §11.2 in 07 doc).
_FEATURE_EXTRACTOR_V1_DOMAINS: Tuple[str, ...] = (
    "voxel_feature_extractor",
    "supervoxel_feature_extractor",
)
#: v1.0 L3 domains resolved through a lazy import table (keeps ``plugins.py``
#: free of a long ``if domain == ...`` chain for the registry-backed domains).
_V1_DOMAIN_REGISTRIES: Mapping[str, Tuple[str, str]] = {
    "classifier": ("habit.domain.classification", "ClassifierRegistry"),
    "metric": ("habit.domain.evaluation", "MetricRegistry"),
    "table_preprocessor": (
        "habit.domain.table_preprocessing",
        "TablePreprocessorRegistry",
    ),
    "feature_selector": ("habit.domain.feature_selection", "FeatureSelectorRegistry"),
    "voxel_feature_extractor": (
        "habit.domain.voxel_features",
        "VoxelFeatureExtractorRegistry",
    ),
    "supervoxelizer": ("habit.domain.supervoxel", "SupervoxelizerRegistry"),
    "supervoxel_feature_extractor": (
        "habit.domain.supervoxel_features",
        "SupervoxelFeatureExtractorRegistry",
    ),
    "feature_preprocessing_method": (
        "habit.domain.feature_preprocessing",
        "FeaturePreprocessingMethodRegistry",
    ),
    "habitat_model_fitter": (
        "habit.domain.habitat_model",
        "HabitatModelFitterRegistry",
    ),
    "habitat_assigner": ("habit.domain.assignment", "HabitatAssignerRegistry"),
    "habitat_feature_extractor": (
        "habit.domain.habitat_features",
        "HabitatFeatureExtractorRegistry",
    ),
    "combiner": ("habit.domain.combiners", "CombinerRegistry"),
    "image_perturbation": (
        "habit.domain.precision",
        "ImagePerturbationRegistry",
    ),
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


def _import_registry(module_path: str, attr_name: str) -> Type[Any]:
    """Import one L3 registry class without eager domain package side effects."""
    from importlib import import_module

    module = import_module(module_path)
    return cast(Type[Any], getattr(module, attr_name))


def _legacy_registry_for_domain(domain: str) -> Type[Any]:
    """Resolve a v0.1 plural plugin domain to its legacy registry gate."""
    from habit.compat import plugin_registries

    if domain == "models":
        return cast(Type[Any], plugin_registries.get_legacy_model_factory())
    if domain == "metrics":
        return cast(Type[Any], plugin_registries.get_legacy_metric_registry())
    if domain == "preprocessors":
        return cast(Type[Any], plugin_registries.get_legacy_preprocessor_factory())
    if domain == "habitat_features":
        return cast(Type[Any], plugin_registries.get_legacy_habitat_feature_factory())
    if domain == "feature_extractors":
        return cast(
            Type[Any], plugin_registries.get_legacy_feature_extractor_registry()
        )
    raise HABITAPIError(
        f"Domain '{domain}' is not a legacy alias. Available legacy aliases: "
        f"{sorted(_LEGACY_DOMAIN_ALIASES)}."
    )


def _feature_extractor_names() -> Tuple[str, ...]:
    """Return the merged name list for the legacy ``feature_extractors`` domain."""
    names: set[str] = set()
    for v1_domain in _FEATURE_EXTRACTOR_V1_DOMAINS:
        names.update(_registry_for_domain(v1_domain).available())
    names.update(_legacy_registry_for_domain("feature_extractors").available())
    return tuple(sorted(names))


def _registry_for_feature_extractor(name: str) -> Type[Any]:
    """
    Resolve one ``feature_extractors`` plugin name across v1 and legacy registries.

    v1 names win when present in ``voxel_feature_extractor`` or
    ``supervoxel_feature_extractor``; everything else falls back to the v0.1
    ``FeatureExtractorRegistry`` (``kinetic``, ``local_entropy``, ...).
    """
    for v1_domain in _FEATURE_EXTRACTOR_V1_DOMAINS:
        registry = _registry_for_domain(v1_domain)
        if name in registry.available():
            return registry
    return _legacy_registry_for_domain("feature_extractors")


def _registry_for_plugin_name(domain: str, name: str) -> Type[Any]:
    """Pick the registry backing one plugin name, honoring legacy aliases."""
    if domain == "feature_extractors":
        return _registry_for_feature_extractor(name)
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
    if domain in _V1_DOMAIN_REGISTRIES:
        module_path, attr_name = _V1_DOMAIN_REGISTRIES[domain]
        return _import_registry(module_path, attr_name)
    if domain == "preprocessor":
        from habit.compat.plugin_registries import get_legacy_preprocessor_factory

        return cast(Type[Any], get_legacy_preprocessor_factory())
    if domain == "feature_extractors":
        raise HABITAPIError(
            "The legacy 'feature_extractors' domain is one-to-many: resolve "
            "plugins per name via get_plugin_info(name, domain='feature_extractors') "
            "or list_plugins(domain='feature_extractors')."
        )
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
        names: list[str]
        if current_domain == "feature_extractors":
            names = list(_feature_extractor_names())
        elif current_domain in _LEGACY_DOMAIN_ALIASES:
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


def create_ml_model(model_name: str, params: Mapping[str, Any] | None = None) -> Any:
    """
    Build one ML model instance for config validation (``habit check-config``).

    Resolves the v1 ``classifier`` registry first, then falls back to the v0.1
    ``ModelFactory`` through the legacy alias routing in
    :func:`_registry_for_plugin_name`.

    Args:
        model_name: Registered model or classifier name from the YAML.
        params: Flat hyperparameter mapping from the config ``params`` block.

    Returns:
        A constructed model wrapper without running training.
    """
    registry = _registry_for_plugin_name("models", model_name)
    return registry.create(model_name, {"params": dict(params or {})})


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
