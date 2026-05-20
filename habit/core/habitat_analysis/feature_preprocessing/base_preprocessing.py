"""
Base classes, factory, and registration for habitat feature preprocessing.

All preprocessing methods share a DataFrame in / DataFrame out contract. Handlers
register through ``@register_preprocessing`` and are resolved by
``PreprocessingMethodFactory``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Type, Union

import pandas as pd
from pydantic import BaseModel

from .method_config_utils import normalize_method_name, resolve_method_name

_PREPROCESSING_REGISTRY: Dict[str, Type["BaseFeaturePreprocessing"]] = {}
_BUILTIN_HANDLERS_LOADED = False


@dataclass
class BaselineStats:
    """
    Initial numeric summaries captured once at the start of group-level fit.

    Per-feature zscore/minmax at transform time reuse these baseline values so
    behaviour matches the historical ``PreprocessingState`` implementation.
    """

    means: pd.Series
    stds: pd.Series
    mins: pd.Series
    maxs: pd.Series

    @classmethod
    def from_dataframe(cls, feature_df: pd.DataFrame) -> "BaselineStats":
        """
        Compute baseline statistics from a numeric feature block.

        Args:
            feature_df: Numeric feature columns only.

        Returns:
            BaselineStats instance with zero-safe std denominators.
        """
        means = feature_df.mean()
        stds = feature_df.std().replace(0, 1.0)
        mins = feature_df.min()
        maxs = feature_df.max()
        return cls(means=means, stds=stds, mins=mins, maxs=maxs)


class BaseFeaturePreprocessing(ABC):
    """
    Abstract preprocessing handler (DataFrame in, DataFrame out).

    Subclasses declare ``changes_columns=True`` when they drop or reorder
    feature columns (variance / correlation filters).
    """

    changes_columns: bool = False

    @classmethod
    @abstractmethod
    def method_name(cls) -> str:
        """Canonical registry key for this handler."""

    @abstractmethod
    def fit(
        self,
        feature_df: pd.DataFrame,
        method_config: Union[Dict[str, Any], BaseModel],
        baseline: Optional[BaselineStats] = None,
    ) -> Any:
        """
        Learn method-specific state from training data.

        Args:
            feature_df: Numeric feature block at the current pipeline step.
            method_config: YAML / pydantic configuration for this step.
            baseline: Optional cohort baseline stats from pipeline fit start.

        Returns:
            Serializable state consumed by :meth:`transform` (``None`` when unused).
        """

    @abstractmethod
    def transform(
        self,
        feature_df: pd.DataFrame,
        method_config: Union[Dict[str, Any], BaseModel],
        state: Any,
        baseline: Optional[BaselineStats] = None,
    ) -> pd.DataFrame:
        """
        Apply preprocessing to a feature block.

        Args:
            feature_df: Numeric feature block at the current pipeline step.
            method_config: YAML / pydantic configuration for this step.
            state: State returned by :meth:`fit` (may be ``None``).
            baseline: Optional cohort baseline stats from pipeline fit start.

        Returns:
            Transformed feature DataFrame (same or fewer columns).
        """


def register_preprocessing(name: str):
    """
    Decorator that registers a preprocessing handler with the factory.

    Args:
        name: Canonical method name (e.g. ``"minmax"``).

    Returns:
        Class decorator.
    """

    def decorator(cls: Type[BaseFeaturePreprocessing]) -> Type[BaseFeaturePreprocessing]:
        PreprocessingMethodFactory.register(name)(cls)
        return cls

    return decorator


class PreprocessingMethodFactory:
    """Factory for resolving and instantiating registered preprocessing handlers."""

    _registry: Dict[str, Type[BaseFeaturePreprocessing]] = _PREPROCESSING_REGISTRY

    @classmethod
    def register(cls, name: str):
        """
        Register a handler class under a canonical method name.

        Args:
            name: Registry key (stored lower-case).

        Returns:
            Class decorator.
        """

        def decorator(handler_cls: Type[BaseFeaturePreprocessing]) -> Type[BaseFeaturePreprocessing]:
            cls._registry[normalize_method_name(name)] = handler_cls
            return handler_cls

        return decorator

    @classmethod
    def _ensure_builtin_handlers_loaded(cls) -> None:
        global _BUILTIN_HANDLERS_LOADED
        if not _BUILTIN_HANDLERS_LOADED:
            from . import builtin_methods as _builtin_methods  # noqa: F401

            _BUILTIN_HANDLERS_LOADED = True

    @classmethod
    def get_handler(cls, method_name: str) -> BaseFeaturePreprocessing:
        """
        Instantiate a registered handler by method name.

        Args:
            method_name: Config ``method`` string (aliases supported).

        Returns:
            Handler instance.

        Raises:
            ValueError: When the method is not registered.
        """
        cls._ensure_builtin_handlers_loaded()
        key = normalize_method_name(method_name)
        handler_cls = cls._registry.get(key)
        if handler_cls is None:
            raise ValueError(f"Unknown preprocessing method: {method_name}")
        return handler_cls()

    @classmethod
    def get_handler_for_config(
        cls,
        method_config: Union[Dict[str, Any], BaseModel],
    ) -> BaseFeaturePreprocessing:
        """
        Resolve a handler from a preprocessing step configuration object.

        Args:
            method_config: Single step from ``PreprocessingConfig.methods``.

        Returns:
            Handler instance for the configured method.
        """
        return cls.get_handler(resolve_method_name(method_config))

    @classmethod
    def registered_method_names(cls) -> List[str]:
        """
        Return sorted canonical names of all registered handlers.

        Returns:
            List of registry keys.
        """
        cls._ensure_builtin_handlers_loaded()
        return sorted(cls._registry.keys())

    @classmethod
    def dropping_method_names(cls) -> FrozenSet[str]:
        """
        Return method names that remove or reorder feature columns.

        Returns:
            Frozen set of registry keys with ``changes_columns=True``.
        """
        cls._ensure_builtin_handlers_loaded()
        return frozenset(
            name
            for name, handler_cls in cls._registry.items()
            if handler_cls.changes_columns
        )
