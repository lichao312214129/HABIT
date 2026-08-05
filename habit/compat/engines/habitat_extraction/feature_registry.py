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
"""Factory and base contract for post-segmentation habitat features.

Habitat feature extractors follow the same factory pattern as feature-table
preprocessing: concrete handlers self-register with
``@HabitatFeatureFactory.register("name")`` and callers obtain an instance by
name through :meth:`HabitatFeatureFactory.get_handler`.

``SubjectExtractionContext`` and ``BatchExportContext`` are typed bundles that
carry all information a handler needs.  Consequently,
``HabitatMapAnalyzer`` dispatches handlers through their common contract and
does not depend on concrete feature classes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

from habit.registry.base import ClassRegistry

FeatureT = TypeVar("FeatureT", bound="BaseHabitatFeature")

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

class BaseHabitatFeature(ABC):
    """Abstract handler for one post-segmentation habitat feature type.

    Both built-in features and optional packages (for example, HABIT-v2 graph
    topology features) inherit from this class. Register a subclass with::

        @HabitatFeatureFactory.register("my_feature")
        class MyFeature(BaseHabitatFeature):
            ...

    Class attributes that every handler must set:
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

    @classmethod
    def feature_name(cls) -> str:
        """Return the canonical factory name assigned by the decorator.

        Concrete handlers may override this method for an explicit declaration.
        Keeping this default preserves compatibility with third-party handlers
        registered before the factory interface introduced ``feature_name``.
        """
        return cls.name

    @abstractmethod
    def extract_subject(self, ctx: SubjectExtractionContext) -> Dict[str, Any]:
        """Extract features for one subject.

        Args:
            ctx: Per-subject extraction context (paths, n_habitats, logger).

        Returns:
            Dict of extracted features; stored under ``subject_data_key``.
        """
        raise NotImplementedError

    @abstractmethod
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

class HabitatFeatureFactory(ClassRegistry[BaseHabitatFeature]):
    """
    Factory for habitat feature extraction handlers.

    Uses the shared :class:`~habit.core.common.registry.ClassRegistry` contract.
    Built-in and optional feature modules are discovered lazily, matching
    :class:`PreprocessingMethodFactory`.
    """

    kind = "habitat feature"

    @classmethod
    def _discover(cls) -> None:
        """Import built-in handlers and optional extensions once on demand."""
        bootstrap_optional_features()

    @classmethod
    def register(cls, name: str) -> Callable[[Type[FeatureT]], Type[FeatureT]]:
        """Register a handler under ``name`` and set its class attribute.

        Overrides :meth:`ClassRegistry.register` because a plugin's registry key
        must also be exposed on the class as ``cls.name`` (the orchestrator reads
        it back), so the two never drift apart.

        Args:
            name: The feature type name (e.g. ``'msi'``, ``'ith_score'``).

        Returns:
            Class decorator that stores the class, sets ``cls.name`` and returns
            the class unchanged.
        """
        def decorator(target: Type[FeatureT]) -> Type[FeatureT]:
            target.name = name
            cls._registry[cls._normalize(name)] = target
            return target

        return decorator

    @classmethod
    def get_handler(
        cls,
        feature_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> BaseHabitatFeature:
        """Instantiate a registered habitat feature handler by name.

        Args:
            feature_name: Registered feature type name.
            *args: Positional arguments forwarded to the handler constructor.
            **kwargs: Keyword arguments forwarded to the handler constructor.

        Returns:
            Configured habitat feature handler.

        Raises:
            ValueError: If ``feature_name`` is not registered.
        """
        return cls.create(feature_name, *args, **kwargs)

    @classmethod
    def registered_feature_names(cls) -> List[str]:
        """Return sorted canonical names of all available feature handlers."""
        bootstrap_optional_features()
        return sorted(cls._registry.keys())


def bootstrap_builtin_features() -> None:
    """Import built-in handlers so every built-in type self-registers.

    This is idempotent: Python's module cache prevents double-import.
    """
    import habit.compat.engines.habitat_extraction.habitat_features.builtin_plugins  # noqa: F401


def bootstrap_optional_features() -> None:
    """Import optional feature packages so their handlers self-register.

    Built-in handlers are imported first so they are always available,
    then optional packages (e.g. HABIT-v2 graph features) are attempted.
    """
    global _PLUGINS_BOOTSTRAPPED
    if _PLUGINS_BOOTSTRAPPED:
        return
    bootstrap_builtin_features()
    try:
        import_module("habit.compat.engines.habitat_extraction.habitat_features.graph_features")
    except ImportError:
        pass
    _PLUGINS_BOOTSTRAPPED = True


def list_registered_plugins() -> List[str]:
    """Return names of all registered feature handlers (built-in + optional).

    Returns:
        List of registered feature type name strings.
    """
    return HabitatFeatureFactory.registered_feature_names()


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
    bootstrap_optional_features()
    if HabitatFeatureFactory.get("graph") is None:
        raise ValueError(
            "feature_types includes 'graph' but the graph feature plugin is "
            "not installed. Graph topology features are only available in "
            "the private HABIT-v2 distribution."
        )


def build_feature_handler(
    name: str,
    config: Optional[Any] = None,
) -> BaseHabitatFeature:
    """Instantiate a registered habitat feature handler by name.

    Args:
        name: Registered feature type name.
        config: Optional config object passed to the plugin constructor.

    Returns:
        Instantiated plugin object.

    Raises:
        ValueError: If ``name`` is not in the registry.
    """
    return HabitatFeatureFactory.get_handler(name, config=config)


# Backward-compatible aliases for external extensions built against the
# pre-factory naming. New code should use BaseHabitatFeature,
# HabitatFeatureFactory, and build_feature_handler.
HabitatFeaturePluginBase = BaseHabitatFeature
HabitatFeatureRegistry = HabitatFeatureFactory
bootstrap_builtin_plugins = bootstrap_builtin_features
bootstrap_optional_plugins = bootstrap_optional_features
build_plugin = build_feature_handler
