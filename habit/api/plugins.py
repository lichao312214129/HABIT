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
import inspect
from dataclasses import dataclass, field
from importlib import metadata
from types import MappingProxyType
from typing import (
    Any,
    Annotated,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
    get_args,
    get_type_hints,
    get_origin,
)

from pydantic import BaseModel

from habit.exceptions import HABITAPIError
from habit.utils.deprecation import HabitDeprecationWarning, build_deprecation_message

logger = logging.getLogger("habit.api.plugins")

__all__ = [
    "PluginInfo",
    "PluginParamInfo",
    "PluginCatalogEntry",
    "PluginLoadReport",
    "create_ml_model",
    "list_plugins",
    "get_plugin_info",
    "get_param_schema",
    "plugin_catalog",
    "format_plugin_catalog_rst",
    "load_plugins",
]

#: Plugin domain -> entry point group. The v0.1 plural domains are kept for
#: backward compatibility and resolve to the v0.1 factories; the v1.0 domains
#: are ``snake_case`` of their protocol class, singular
#: (developer/api_upgrade/08 §4), and resolve to the L3 domain registries.
#: ``preprocessor`` is the v1 image-volume domain (resample, z-score, N4,
#: …). The v0.1 plural ``preprocessors`` alias still lists the compat
#: factory. ``table_preprocessor`` / ``feature_selector`` /
#: ``classifier`` / ``metric`` name the v1.0 table-ML domains.
_ENTRY_POINT_GROUPS: Mapping[str, str] = {
    # v0.1 domains (legacy, kept working).
    "preprocessors": "habit.preprocessors",
    "radiomics_backends": "habit.radiomics_backends",
    "feature_extractors": "habit.feature_extractors",
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
    # Dataflow watershed marker (subject→cohort fan-in).
    "pooling": "habit.pooling",
}
#: v0.1 plural plugin domains that alias the v1.0 singular L3 registries.
#: Per-name resolution prefers the v1 registry when the name exists there,
#: otherwise falls back to the v0.1 core factory (see ``_registry_for_plugin_name``).
_LEGACY_DOMAIN_ALIASES: Mapping[str, str] = {
    "models": "classifier",
    "metrics": "metric",
    "preprocessors": "preprocessor",
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
    "classifier": ("habit.classification", "ClassifierRegistry"),
    "metric": ("habit.evaluation", "MetricRegistry"),
    "table_preprocessor": (
        "habit.table_preprocessing",
        "TablePreprocessorRegistry",
    ),
    "feature_selector": ("habit.feature_selection", "FeatureSelectorRegistry"),
    "voxel_feature_extractor": (
        "habit.voxel_features",
        "VoxelFeatureExtractorRegistry",
    ),
    "supervoxelizer": ("habit.supervoxel", "SupervoxelizerRegistry"),
    "supervoxel_feature_extractor": (
        "habit.supervoxel",
        "SupervoxelFeatureExtractorRegistry",
    ),
    "feature_preprocessing_method": (
        "habit.feature_preprocessing",
        "FeaturePreprocessingMethodRegistry",
    ),
    "habitat_model_fitter": (
        "habit.habitat_model",
        "HabitatModelFitterRegistry",
    ),
    "habitat_assigner": ("habit.habitat_model", "HabitatAssignerRegistry"),
    "habitat_feature_extractor": (
        "habit.habitat_features",
        "HabitatFeatureExtractorRegistry",
    ),
    "combiner": ("habit.combiners", "CombinerRegistry"),
    "image_perturbation": (
        "habit.precision",
        "ImagePerturbationRegistry",
    ),
    "pooling": ("habit.pipeline", "PoolingRegistry"),
    "preprocessor": (
        "habit.image_preprocessing",
        "PreprocessorRegistry",
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
class PluginParamInfo:
    """One ``Spec`` / ``Registry.create`` parameter from the v2 constructor.

    Rows are built from :meth:`~habit.registry.ComponentRegistry.constructor_signature`
    for built-in components. When a third-party plugin registers a legacy
    Pydantic ``params_model``, that schema is used instead.

    ``description`` prefers the Pydantic ``Field(description=...)``, then the
    registered class ``Args:`` section, then the params-model field docstring.
    ``allowed`` is a short constraint string (Literal choices, numeric bounds).
    """

    name: str
    required: bool
    annotation: str
    default: str
    allowed: str
    description: str


@dataclass(frozen=True)
class PluginCatalogEntry:
    """One catalog row derived from the constructor contract (not a hand-copied table).

    Built-in components use :meth:`~habit.registry.ComponentRegistry.constructor_signature`.
    Third-party plugins that register a legacy Pydantic ``params_model`` fall back
    to that schema for parameter metadata.
    """

    domain: str
    name: str
    purpose: str
    required_params: Tuple[str, ...]
    optional_params: Tuple[str, ...]
    spec_example: str
    create_example: str
    params: Tuple[PluginParamInfo, ...] = ()


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
        # The v0.1 plural alias remains public throughout v1.x, but its
        # implementation is now the canonical image-preprocessor registry.
        return _import_registry(
            "habit.image_preprocessing.registry",
            "PreprocessorRegistry",
        )
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
        f"Plugin {name!r} is not registered in domain {domain!r}. "
        f"Available: {[info.name for info in list_plugins(domain)]}. "
        f"Inspect with list_plugins({domain!r}) or "
        f"get_param_schema(name, {domain!r})."
    )


def get_param_schema(name: str, domain: str) -> Optional[Type[BaseModel]]:
    """Return a legacy Pydantic ``params_model`` for a plugin, if registered.

    Built-in v2 components do **not** register ``params_model``; for those use
    :meth:`~habit.registry.ComponentRegistry.constructor_signature` or
    :func:`plugin_catalog` instead. This helper remains for third-party plugins
    that still attach a Pydantic schema via ``register_params_model``.
    """
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


#: v1 domains shown on the generated plugin catalog (legacy plural aliases
#: are omitted so YAML and ``Spec`` authors see one name per component).
_CATALOG_DOMAINS: Tuple[str, ...] = (
    "voxel_feature_extractor",
    "supervoxelizer",
    "supervoxel_feature_extractor",
    "feature_preprocessing_method",
    "habitat_model_fitter",
    "habitat_assigner",
    "habitat_feature_extractor",
    "combiner",
    "image_perturbation",
    "pooling",
    "preprocessor",
    "table_preprocessor",
    "feature_selector",
    "classifier",
    "metric",
)


def _one_line_purpose(payload: Any, schema: Optional[Type[BaseModel]]) -> str:
    """Return the first docstring sentence of the class or its params model."""
    for candidate in (payload, schema):
        if candidate is None:
            continue
        doc = inspect.getdoc(candidate)
        if not doc:
            continue
        first = doc.strip().splitlines()[0].strip()
        if first:
            return first.rstrip(".")
    return "Registered plugin (see constructor_signature for arguments)"


def _param_names(schema: Optional[Type[BaseModel]]) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Split a Pydantic params model into required vs optional field names."""
    if schema is None:
        return (), ()
    required: list[str] = []
    optional: list[str] = []
    for field_name, field in schema.model_fields.items():
        if field.is_required():
            required.append(field_name)
        else:
            optional.append(field_name)
    return tuple(required), tuple(optional)


def _spec_example(name: str, required: Sequence[str]) -> str:
    """Return one ``Spec(...)`` line; required keys are shown as placeholders."""
    if not required:
        return f'Spec("{name}")'
    inner = ", ".join(f'"{key}": ...' for key in required)
    return f'Spec("{name}", {{{inner}}})'


def _create_example(name: str, required: Sequence[str]) -> str:
    """Return one ``Registry.create(...)`` line for the same component."""
    if not required:
        return f'Registry.create("{name}")'
    kwargs = ", ".join(f"{key}=..." for key in required)
    return f'Registry.create("{name}", {kwargs})'


def _google_args_map(doc: Optional[str]) -> Dict[str, str]:
    """Parse a Google-style ``Args:`` section into ``{name: description}``."""
    if not doc:
        return {}
    lines = inspect.cleandoc(doc).splitlines()
    start: Optional[int] = None
    for index, line in enumerate(lines):
        if line.strip() == "Args:":
            start = index + 1
            break
    if start is None:
        return {}
    collected: Dict[str, str] = {}
    current_name: Optional[str] = None
    current_parts: List[str] = []
    section_stops = {
        "Returns:",
        "Raises:",
        "Note:",
        "Notes:",
        "Example:",
        "Examples:",
        "See Also:",
    }

    def _flush() -> None:
        if current_name is None:
            return
        text = " ".join(part.strip() for part in current_parts if part.strip())
        if text:
            collected[current_name] = text

    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in section_stops:
            break
        indent = len(line) - len(line.lstrip(" "))
        if indent == 0:
            break
        if indent <= 4 and ":" in stripped:
            name, rest = stripped.split(":", 1)
            name = name.strip()
            if name.isidentifier() or name.endswith("_"):
                _flush()
                current_name = name
                current_parts = [rest.strip()]
                continue
        if current_name is not None:
            current_parts.append(stripped)
    _flush()
    return collected


def _format_annotation(annotation: Any) -> str:
    """Return a short, docs-safe type label for one Pydantic field."""
    if annotation is None:
        return ""
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Literal:
        return " | ".join(repr(value) for value in args)
    if origin is Annotated:
        return _format_annotation(args[0]) if args else ""
    if origin is Union:
        non_none = [item for item in args if item is not type(None)]
        if len(non_none) == 1 and type(None) in args:
            return f"{_format_annotation(non_none[0])} | None"
        return " | ".join(_format_annotation(item) for item in args)
    if origin in (list, List, Sequence, tuple, Tuple):
        inner = ", ".join(_format_annotation(item) for item in args) if args else ""
        name = "list" if origin in (list, List, Sequence) else "tuple"
        return f"{name}[{inner}]" if inner else name
    if origin in (dict, Dict, Mapping):
        return "dict"
    return getattr(annotation, "__name__", None) or str(annotation).replace("typing.", "")


def _format_default(field: Any) -> str:
    """Return a short default label; required fields are marked as such."""
    if field.is_required():
        return "(required)"
    default_factory = getattr(field, "default_factory", None)
    if default_factory is not None and field.default is None:
        # Pydantic v2 stores missing defaults as PydanticUndefined; factory
        # fields then expose default_factory instead of a concrete value.
        try:
            from pydantic_core import PydanticUndefined

            if field.default is PydanticUndefined:
                return "(factory)"
        except Exception:  # noqa: BLE001 — optional import for formatting only
            return "(factory)"
    default = field.default
    if default is None:
        return "None"
    return repr(default)


def _format_allowed(field: Any, annotation: Any) -> str:
    """Return Literal choices and numeric bounds as one constraint string."""
    parts: List[str] = []
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Literal:
        parts.append(" | ".join(repr(value) for value in args))
    elif origin is Union:
        for item in args:
            if get_origin(item) is Literal:
                parts.append(" | ".join(repr(value) for value in get_args(item)))
    metadata = getattr(field, "metadata", ()) or ()
    for item in metadata:
        ge = getattr(item, "ge", None)
        gt = getattr(item, "gt", None)
        le = getattr(item, "le", None)
        lt = getattr(item, "lt", None)
        if ge is not None:
            parts.append(f">= {ge}")
        if gt is not None:
            parts.append(f"> {gt}")
        if le is not None:
            parts.append(f"<= {le}")
        if lt is not None:
            parts.append(f"< {lt}")
    return "; ".join(parts)


def _param_infos(payload: Any, schema: Optional[Type[BaseModel]]) -> Tuple[PluginParamInfo, ...]:
    """Build per-parameter catalog rows from ``params_model`` plus class Args."""
    if schema is None:
        return ()
    docs: Dict[str, str] = {}
    docs.update(_google_args_map(inspect.getdoc(schema)))
    if payload is not None:
        for candidate in getattr(payload, "__mro__", (payload,)):
            docs.update(_google_args_map(inspect.getdoc(candidate)))
    infos: List[PluginParamInfo] = []
    for field_name, field in schema.model_fields.items():
        annotation = getattr(field, "annotation", None)
        description = (field.description or "").strip() or docs.get(field_name, "")
        infos.append(
            PluginParamInfo(
                name=field_name,
                required=bool(field.is_required()),
                annotation=_format_annotation(annotation),
                default=_format_default(field),
                allowed=_format_allowed(field, annotation),
                description=description,
            )
        )
    return tuple(infos)


def _signature_param_infos(payload: Any) -> Tuple[PluginParamInfo, ...]:
    """Derive v2 catalog fields directly from one component constructor."""
    signature = inspect.signature(payload)
    try:
        hints = get_type_hints(payload.__init__)
    except Exception:  # Third-party annotations may reference unavailable types.
        hints = {}
    docs = _google_args_map(inspect.getdoc(payload))
    infos: list[PluginParamInfo] = []
    for parameter in signature.parameters.values():
        if parameter.name == "self" or parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        annotation = hints.get(parameter.name, parameter.annotation)
        required = parameter.default is inspect.Parameter.empty
        default = "(required)" if required else repr(parameter.default)
        infos.append(
            PluginParamInfo(
                name=parameter.name,
                required=required,
                annotation=_format_annotation(annotation),
                default=default,
                allowed=_signature_allowed(annotation),
                description=docs.get(parameter.name, ""),
            )
        )
    return tuple(infos)


def _signature_allowed(annotation: Any) -> str:
    """Render ``Literal`` choices and ``Annotated`` numeric bounds."""
    origin = get_origin(annotation)
    if origin is Literal:
        return " | ".join(repr(value) for value in get_args(annotation))
    if origin is Annotated:
        base, *metadata = get_args(annotation)
        parts = [_signature_allowed(base)] if _signature_allowed(base) else []
        for item in metadata:
            for attr, operator in (("ge", ">="), ("gt", ">"), ("le", "<="), ("lt", "<")):
                value = getattr(item, attr, None)
                if value is not None:
                    parts.append(f"{operator} {value}")
        return "; ".join(parts)
    if origin is Union:
        return " | ".join(
            item_allowed
            for item in get_args(annotation)
            if (item_allowed := _signature_allowed(item))
        )
    return ""


def plugin_catalog(
    domain: Optional[str] = None,
) -> Tuple[PluginCatalogEntry, ...]:
    """
    Build a live catalog from each plugin's constructor contract.

    This is the source of truth for ``Spec("name", {{...}})`` and
    ``Registry.create("name", ...)``: names, required/optional parameters,
    and a one-line purpose come from the registered class and its
    :meth:`~habit.registry.ComponentRegistry.constructor_signature`, not from
    a hand-copied table. Third-party plugins with a legacy ``params_model``
    still publish rows from that schema when present.

    Args:
        domain: Optional v1 domain. Omit to enumerate every catalog domain.

    Returns:
        Deterministically ordered catalog rows.

    Raises:
        HABITAPIError: If ``domain`` is unknown.
    """
    if domain is not None and domain not in _CATALOG_DOMAINS:
        if domain in _ENTRY_POINT_GROUPS:
            raise HABITAPIError(
                f"plugin_catalog: {domain!r} is not a v1 catalog domain. "
                f"Use one of {list(_CATALOG_DOMAINS)} "
                f"(legacy aliases still work with list_plugins)."
            )
        raise HABITAPIError(
            f"Unknown plugin domain {domain!r}. Available domains: "
            f"{list(_CATALOG_DOMAINS)}."
        )
    domains: Iterable[str] = (domain,) if domain is not None else _CATALOG_DOMAINS
    entries: list[PluginCatalogEntry] = []
    for current_domain in domains:
        for info in list_plugins(current_domain):
            registry = _registry_for_plugin_name(current_domain, info.name)
            payload = registry.get(info.name)
            schema = get_param_schema(info.name, current_domain)
            params = (
                _param_infos(payload, schema)
                if schema is not None
                else _signature_param_infos(payload)
            )
            required = tuple(item.name for item in params if item.required)
            optional = tuple(item.name for item in params if not item.required)
            entries.append(
                PluginCatalogEntry(
                    domain=current_domain,
                    name=info.name,
                    purpose=_one_line_purpose(payload, schema),
                    required_params=required,
                    optional_params=optional,
                    spec_example=_spec_example(info.name, required),
                    create_example=_create_example(info.name, required),
                    params=params,
                )
            )
    return tuple(entries)


def format_plugin_catalog_rst(domain: Optional[str] = None) -> str:
    """
    Render :func:`plugin_catalog` as reStructuredText for the docs catalog.

    Sphinx calls this at build time so the page cannot drift from the live
    constructor signatures (or legacy third-party ``params_model`` when set).
    """
    lines: list[str] = []
    current_domain = ""
    for entry in plugin_catalog(domain):
        if entry.domain != current_domain:
            current_domain = entry.domain
            lines.extend(
                [
                    "",
                    current_domain,
                    "^" * len(current_domain),
                    "",
                ]
            )
        required = ", ".join(entry.required_params) if entry.required_params else "(none)"
        optional = ", ".join(entry.optional_params) if entry.optional_params else "(none)"
        lines.extend(
            [
                f"**{entry.name}**",
                f"  {entry.purpose}.",
                f"  Required: ``{required}``. Optional: ``{optional}``.",
                f"  ``{entry.spec_example}``",
                f"  ``{entry.create_example}``",
                "",
            ]
        )
        if entry.params:
            lines.extend(
                [
                    "  .. list-table::",
                    "     :header-rows: 1",
                    "     :widths: 18 16 24 42",
                    "",
                    "     * - Param",
                    "       - Default",
                    "       - Allowed / type",
                    "       - Meaning",
                ]
            )
            for param in entry.params:
                allowed = param.allowed or param.annotation or "—"
                meaning = param.description or "See the component docstring."
                meaning = " ".join(meaning.split())
                lines.extend(
                    [
                        f"     * - ``{param.name}``",
                        f"       - ``{param.default}``",
                        f"       - {allowed}",
                        f"       - {meaning}",
                    ]
                )
            lines.append("")
    return "\n".join(lines).strip() + "\n"


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
