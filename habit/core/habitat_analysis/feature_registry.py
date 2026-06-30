"""Registry for optional habitat feature extraction plugins.

Built-in feature types (traditional, msi, etc.) are implemented directly in
``habitat_analyzer.py``. Private plugins such as graph topology register here
and are loaded only when their package is present (v1 / HABIT-v2).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

PluginT = TypeVar("PluginT", bound="HabitatFeaturePluginBase")

BUILTIN_FEATURE_TYPES: List[str] = [
    "traditional",
    "non_radiomics",
    "whole_habitat",
    "each_habitat",
    "msi",
    "ith_score",
]

_REGISTRY: Dict[str, Type[HabitatFeaturePluginBase]] = {}
_PLUGINS_BOOTSTRAPPED: bool = False


class HabitatFeaturePluginBase:
    """Base class for registered habitat feature plugins."""

    name: str = ""
    subject_data_key: str = ""
    output_csv_name: str = ""
    progress_desc: str = ""

    def __init__(self, config: Any = None) -> None:
        self.config = config

    def extract_subject(self, habitat_path: str, logger: Any) -> Dict[str, Any]:
        """Extract features for one subject from a habitat map path."""
        raise NotImplementedError

    def export_batch(
        self,
        data: Dict[str, Dict[str, Any]],
        out_dir: str,
        logger: Any,
    ) -> Any:
        """Aggregate per-subject plugin results and write CSV output."""
        raise NotImplementedError

    def should_visualize(self) -> bool:
        """Return True when post-export visualization should run."""
        return False

    def visualize_batch(
        self,
        data: Dict[str, Dict[str, Any]],
        habitat_paths: Dict[str, str],
        out_dir: str,
        logger: Any,
        n_processes: int,
    ) -> None:
        """Optional visualization hook after CSV export."""


def register_habitat_feature(name: str) -> Callable[[Type[PluginT]], Type[PluginT]]:
    """Decorator that registers a habitat feature plugin under ``name``."""

    def decorator(cls: Type[PluginT]) -> Type[PluginT]:
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def bootstrap_optional_plugins() -> None:
    """Import optional plugin packages so they self-register (idempotent)."""
    global _PLUGINS_BOOTSTRAPPED
    if _PLUGINS_BOOTSTRAPPED:
        return
    try:
        import habit.core.habitat_analysis.habitat_features.graph_features  # noqa: F401
    except ImportError:
        pass
    _PLUGINS_BOOTSTRAPPED = True


def list_registered_plugins() -> List[str]:
    """Return names of currently registered optional plugins."""
    bootstrap_optional_plugins()
    return list(_REGISTRY.keys())


def get_all_feature_type_names() -> List[str]:
    """Return built-in plus registered optional feature type names."""
    return BUILTIN_FEATURE_TYPES + list_registered_plugins()


def get_default_feature_types() -> List[str]:
    """Default feature_types when the caller does not specify a list."""
    return list(get_all_feature_type_names())


def validate_feature_types(feature_types: List[str]) -> None:
    """Raise ValueError when unknown feature types are requested."""
    allowed = set(get_all_feature_type_names())
    unknown = [name for name in feature_types if name not in allowed]
    if unknown:
        raise ValueError(
            "Unknown feature_types: "
            f"{unknown}. Available: {sorted(allowed)}. "
            "Graph features require the private HABIT-v2 plugin package."
        )


def ensure_graph_plugin_available() -> None:
    """Raise ValueError when graph plugin is requested but not installed."""
    bootstrap_optional_plugins()
    if "graph" not in _REGISTRY:
        raise ValueError(
            "feature_types includes 'graph' but the graph feature plugin is "
            "not installed. Graph topology features are only available in "
            "the private HABIT-v2 distribution."
        )


def build_plugin(name: str, config: Optional[Any] = None) -> HabitatFeaturePluginBase:
    """Instantiate a registered plugin by name."""
    bootstrap_optional_plugins()
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown habitat feature plugin: {name!r}. "
            f"Registered plugins: {list_registered_plugins()}"
        )
    return cls(config=config)
