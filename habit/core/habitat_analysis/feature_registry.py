"""Registry for habitat feature extraction plugins.

Both built-in types (traditional, msi, etc.) and optional plugins (e.g. graph
topology in HABIT-v2) register here via ``@register_habitat_feature``.

``SubjectExtractionContext`` and ``BatchExportContext`` are typed bundles that
carry all information a plugin needs, so the orchestrator (HabitatMapAnalyzer)
never needs to know the internals of any feature type.

Adding a new feature type only requires:
    1. Subclass ``HabitatFeaturePluginBase``
    2. Decorate with ``@register_habitat_feature('my_feature')``
    3. Implement ``extract_subject()`` and ``export_batch()``
    4. No changes to HabitatMapAnalyzer are needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

from habit.core.common.registry import ClassRegistry

PluginT = TypeVar("PluginT", bound="HabitatFeaturePluginBase")

_PLUGINS_BOOTSTRAPPED: bool = False


# ---------------------------------------------------------------------------
# Context dataclasses — passed to plugins instead of individual kwargs
# ---------------------------------------------------------------------------

@dataclass
class SubjectExtractionContext:
    """All information a plugin needs to extract features for one subject.

    Attributes:
        subj: Subject identifier string.
        habitat_path: File path to the habitat map for this subject.
        image_paths: Mapping of image-name to file path (all modalities).
        mask_paths: Optional mapping of image-name to mask path.
        n_habitats: Total number of habitat labels; None if not yet known.
        logger: Logger instance (module-level or process-level).
    """

    subj: str
    habitat_path: str
    image_paths: Dict[str, str]
    mask_paths: Optional[Dict[str, str]]
    n_habitats: Optional[int]
    logger: Any


@dataclass
class BatchExportContext:
    """All information a plugin needs to aggregate results and write CSVs.

    Attributes:
        out_dir: Directory where CSV files should be written.
        n_habitats: Total number of habitat labels.
        habitat_paths: Mapping of subject-id to habitat map path.
        logger: Logger instance.
        n_processes: Number of parallel workers (for optional visualisation).
    """

    out_dir: str
    n_habitats: Optional[int]
    habitat_paths: Dict[str, str]
    logger: Any
    n_processes: int = 1


# ---------------------------------------------------------------------------
# Plugin base class
# ---------------------------------------------------------------------------

class HabitatFeaturePluginBase:
    """Base class for habitat feature extraction plugins.

    Both built-in types and optional packages (HABIT-v2) inherit from this
    class.  Register a subclass by decorating it with::

        @register_habitat_feature('my_feature')
        class MyPlugin(HabitatFeaturePluginBase):
            ...

    Class attributes that every plugin must set:
        name: Registered string name (set automatically by the decorator).
        subject_data_key: Key used to store per-subject results.
        output_csv_name: Primary CSV output filename (for logging).
        progress_desc: Short label shown in the progress bar.
    """

    name: str = ""
    subject_data_key: str = ""
    output_csv_name: str = ""
    progress_desc: str = ""

    def __init__(self, config: Any = None) -> None:
        self.config = config

    def extract_subject(self, ctx: SubjectExtractionContext) -> Dict[str, Any]:
        """Extract features for one subject.

        Args:
            ctx: Per-subject extraction context (paths, n_habitats, logger).

        Returns:
            Dict of extracted features; stored under ``subject_data_key``.
        """
        raise NotImplementedError

    def export_batch(
        self,
        data: Dict[str, Dict[str, Any]],
        ctx: BatchExportContext,
    ) -> Any:
        """Aggregate per-subject results and write CSV output.

        Args:
            data: Mapping ``{subject_id: {subject_data_key: features, ...}}``.
            ctx: Batch context (out_dir, n_habitats, logger, …).

        Returns:
            Typically a ``pd.DataFrame``; may return ``None`` on failure.
        """
        raise NotImplementedError

    def should_visualize(self) -> bool:
        """Return True to trigger ``visualize_batch()`` after ``export_batch()``."""
        return False

    def visualize_batch(
        self,
        data: Dict[str, Dict[str, Any]],
        ctx: BatchExportContext,
    ) -> None:
        """Optional visualisation hook called after ``export_batch()``.

        Args:
            data: Same data dict passed to ``export_batch()``.
            ctx: Batch context with paths, out_dir, n_processes, etc.
        """


# ---------------------------------------------------------------------------
# Registry, decorator and helpers
# ---------------------------------------------------------------------------

class HabitatFeatureRegistry(ClassRegistry[HabitatFeaturePluginBase]):
    """
    Registry of habitat feature extraction plugins.

    Uses the shared :class:`~habit.core.common.registry.ClassRegistry` contract.
    Bootstrapping (importing built-in and optional plugin modules) is driven
    explicitly by the module-level helpers below, so ``_discover`` stays a no-op.
    """

    kind = "habitat feature plugin"


def register_habitat_feature(name: str) -> Callable[[Type[PluginT]], Type[PluginT]]:
    """Decorator that registers a habitat feature plugin under ``name``.

    Args:
        name: The feature type name (e.g. ``'msi'``, ``'ith_score'``).

    Returns:
        Class decorator that registers and returns the class unchanged.
    """
    def decorator(cls: Type[PluginT]) -> Type[PluginT]:
        cls.name = name
        HabitatFeatureRegistry.register(name)(cls)
        return cls

    return decorator


def bootstrap_builtin_plugins() -> None:
    """Import built-in plugin module so all built-in types self-register.

    This is idempotent: Python's module cache prevents double-import.
    """
    import habit.core.habitat_analysis.habitat_features.builtin_plugins  # noqa: F401


def bootstrap_optional_plugins() -> None:
    """Import optional plugin packages so they self-register (idempotent).

    Built-in plugins are bootstrapped first so they are always available,
    then optional packages (e.g. HABIT-v2 graph features) are attempted.
    """
    global _PLUGINS_BOOTSTRAPPED
    if _PLUGINS_BOOTSTRAPPED:
        return
    bootstrap_builtin_plugins()
    try:
        import habit.core.habitat_analysis.habitat_features.graph_features  # noqa: F401
    except ImportError:
        pass
    _PLUGINS_BOOTSTRAPPED = True


def list_registered_plugins() -> List[str]:
    """Return names of all registered plugins (built-in + optional).

    Returns:
        List of registered feature type name strings.
    """
    bootstrap_optional_plugins()
    return HabitatFeatureRegistry.available()


def get_all_feature_type_names() -> List[str]:
    """Return all registered feature type names.

    Returns:
        List of all available feature type name strings.
    """
    return list_registered_plugins()


def get_default_feature_types() -> List[str]:
    """Return the default feature_types list (all registered types).

    Returns:
        Copy of all registered feature type names.
    """
    return list(get_all_feature_type_names())


def validate_feature_types(feature_types: List[str]) -> None:
    """Raise ValueError when unknown feature types are requested.

    Args:
        feature_types: List of feature type names to validate.

    Raises:
        ValueError: If any name is not registered.
    """
    allowed = set(get_all_feature_type_names())
    unknown = [name for name in feature_types if name not in allowed]
    if unknown:
        raise ValueError(
            f"Unknown feature_types: {unknown}. Available: {sorted(allowed)}. "
            "Graph features require the private HABIT-v2 plugin package."
        )


def ensure_graph_plugin_available() -> None:
    """Raise ValueError when the graph plugin is requested but not installed.

    Raises:
        ValueError: If the graph plugin is not registered.
    """
    bootstrap_optional_plugins()
    if HabitatFeatureRegistry.get("graph") is None:
        raise ValueError(
            "feature_types includes 'graph' but the graph feature plugin is "
            "not installed. Graph topology features are only available in "
            "the private HABIT-v2 distribution."
        )


def build_plugin(name: str, config: Optional[Any] = None) -> HabitatFeaturePluginBase:
    """Instantiate a registered plugin by name.

    Args:
        name: Registered feature type name.
        config: Optional config object passed to the plugin constructor.

    Returns:
        Instantiated plugin object.

    Raises:
        ValueError: If ``name`` is not in the registry.
    """
    bootstrap_optional_plugins()
    cls = HabitatFeatureRegistry.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown habitat feature plugin: {name!r}. "
            f"Registered plugins: {list_registered_plugins()}"
        )
    return cls(config=config)
