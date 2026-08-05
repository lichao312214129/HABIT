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
"""
Shared registry base for HABIT factories.

Historically every subsystem defined its own ``*Factory`` / registry with
slightly different method names (``create`` vs ``create_model`` vs
``create_algorithm``; ``get_available_*`` vs ``available``; ad-hoc module-level
dicts such as ``_SELECTOR_REGISTRY`` / ``METRIC_REGISTRY``). This module provides
a single, uniform contract so that a developer learns the extension mechanism
**once** and applies it everywhere, regardless of whether the registered payload
is a *class* (instantiated on demand) or a plain *callable* (looked up and
invoked directly).

Two concrete bases build on the shared :class:`_BaseRegistry` core:

* :class:`ClassRegistry` — payload is a **class**; ``create`` instantiates it.
* :class:`CallableRegistry` — payload is a **callable** (function) plus optional
  metadata; ``get`` returns the callable, ``get_entry`` returns its metadata.

Canonical contract exposed by every subclass::

    @MyRegistry.register("my_name")          # decorator: name -> class / callable
    ...

    MyRegistry.get("my_name")                 # -> class / callable or None
    MyRegistry.available()                    # -> list[str] of registered names
    MyRegistry.register_params_model(name, m) # attach a Pydantic *Params schema
    MyRegistry.get_params_model(name)         # -> params model or None

Class registries add ``create(name, *args, **kwargs)``; callable registries add
``register(name, **metadata)`` / ``get_entry(name)`` / ``entries()``.

Notes
-----
* Each concrete subclass owns **independent** ``_registry`` / ``_params_models``
  / ``_metadata`` mappings (created in :meth:`__init_subclass__`), so
  registrations never leak between domains.
* The ``_registry`` attribute name is part of the public contract because the
  GUI reflection bridge inspects it directly; do not rename it.
* Subclasses that discover implementations lazily (by importing sibling modules
  on first access) override :meth:`_discover`. Subclasses whose registration
  keys are case-insensitive override :meth:`_normalize`.
* Class subclasses whose constructor call convention differs from
  ``cls_(**kwargs)`` (for example, models that take a single positional
  ``config`` dict) override :meth:`ClassRegistry.create` only.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar

T = TypeVar("T")


class _BaseRegistry(Generic[T]):
    """
    Shared ``name -> payload`` storage and lookup core for HABIT registries.

    Not used directly; subclass :class:`ClassRegistry` (class payloads) or
    :class:`CallableRegistry` (callable payloads) instead. This base owns the
    per-subclass storage, key normalization, lazy discovery, ``available`` and
    the Pydantic ``*Params`` schema association, all of which are identical for
    both payload kinds.

    ``kind`` is only used to produce clear error messages.
    """

    #: Human-readable component kind, used only in error messages.
    kind: str = "component"

    #: Populated per-subclass in ``__init_subclass__`` (never shared).
    _registry: Dict[str, T]
    _params_models: Dict[str, Type[Any]]
    #: Optional per-entry metadata (only populated by callable registries).
    _metadata: Dict[str, Dict[str, Any]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Give every concrete registry its own isolated storage."""
        super().__init_subclass__(**kwargs)
        cls._registry = {}
        cls._params_models = {}
        cls._metadata = {}

    # ------------------------------------------------------------------
    # Overridable hooks
    # ------------------------------------------------------------------
    @classmethod
    def _normalize(cls, name: str) -> str:
        """
        Normalize a registration key.

        The default is identity. Override (e.g. ``return name.lower()``) when
        the registry should be case-insensitive.

        Args:
            name: Raw registration / lookup key.

        Returns:
            str: Normalized key actually stored in ``_registry``.
        """
        return name

    @classmethod
    def _discover(cls) -> None:
        """
        Lazily import sibling modules so decorated components self-register.

        The default is a no-op (all implementations are imported eagerly).
        Override in subclasses that scan a package directory on first access.
        """
        return None

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    @classmethod
    def get(cls, name: str) -> Optional[T]:
        """
        Return the registered payload for ``name`` (running discovery if needed).

        Args:
            name: Registration key to look up.

        Returns:
            Optional[T]: The class / callable, or ``None`` if not registered.
        """
        key = cls._normalize(name)
        if key not in cls._registry:
            cls._discover()
        return cls._registry.get(key)

    @classmethod
    def available(cls) -> List[str]:
        """
        Return all registered names (running discovery if the registry is empty).

        Returns:
            List[str]: Registered registration keys.
        """
        if not cls._registry:
            cls._discover()
        return list(cls._registry.keys())

    # ------------------------------------------------------------------
    # Pydantic *Params schema association (used by ParamSchemaRegistry / GUI)
    # ------------------------------------------------------------------
    @classmethod
    def register_params_model(cls, name: str, params_model: Type[Any]) -> None:
        """
        Associate a Pydantic ``*Params`` schema with a registered component.

        Args:
            name: Registration key of the component.
            params_model: Pydantic model describing user-configurable params
                (used for GUI form generation and YAML validation).
        """
        cls._params_models[cls._normalize(name)] = params_model

    @classmethod
    def get_params_model(cls, name: str) -> Optional[Type[Any]]:
        """
        Return the Pydantic params schema for a component, if any.

        Args:
            name: Registration key of the component.

        Returns:
            Optional[Type[Any]]: The params model class, or ``None``.
        """
        return cls._params_models.get(cls._normalize(name))


class ClassRegistry(_BaseRegistry[Type[T]]):
    """
    Generic ``name -> class`` registry shared by HABIT class-based factories.

    Subclass it to create a domain factory::

        class PreprocessorFactory(ClassRegistry[BasePreprocessor]):
            kind = "preprocessor"
    """

    @classmethod
    def register(cls, name: str) -> Callable[[Type[T]], Type[T]]:
        """
        Return a decorator that registers a class under ``name``.

        Args:
            name: Unique registration key for the decorated class.

        Returns:
            Callable[[Type[T]], Type[T]]: Decorator that stores and returns the
            class unchanged.
        """

        def decorator(target: Type[T]) -> Type[T]:
            cls._registry[cls._normalize(name)] = target
            return target

        return decorator

    @classmethod
    def create(cls, name: str, *args: Any, **kwargs: Any) -> T:
        """
        Instantiate a registered class by name.

        Args:
            name: Registration key.
            *args: Positional arguments forwarded to the class constructor.
            **kwargs: Keyword arguments forwarded to the class constructor.

        Returns:
            T: A new instance of the registered class.

        Raises:
            ValueError: If ``name`` is not registered (after discovery).
        """
        target = cls.get(name)
        if target is None:
            raise ValueError(
                f"Unknown {cls.kind}: {name!r}. Available: {cls.available()}"
            )
        return target(*args, **kwargs)


class CallableRegistry(_BaseRegistry[T]):
    """
    Generic ``name -> callable`` registry with optional per-entry metadata.

    Use this for extension points whose payload is a plain function rather than
    a class (feature selectors, evaluation metrics). It shares the same
    ``register`` / ``get`` / ``available`` / ``*_params_model`` contract as
    :class:`ClassRegistry`; the differences are:

    * :meth:`register` accepts arbitrary keyword ``metadata`` stored alongside
      the callable and merged over :attr:`default_metadata`.
    * :meth:`get` returns the callable; :meth:`get_entry` / :meth:`entries`
      expose the metadata (which always includes ``func`` and ``display_name``).

    Subclass to create a domain registry::

        class MetricRegistry(CallableRegistry[Callable]):
            kind = "metric"
            default_metadata = {"category": "basic"}
    """

    #: Default metadata merged under every entry (overridden per subclass).
    default_metadata: Dict[str, Any] = {}

    @classmethod
    def register(
        cls, name: str, *, display_name: Optional[str] = None, **metadata: Any
    ) -> Callable[[T], T]:
        """
        Return a decorator that registers a callable under ``name``.

        Args:
            name: Unique registration key for the decorated callable.
            display_name: Human-readable label (falls back to a title-cased
                version of ``name`` when omitted).
            **metadata: Extra per-entry metadata (e.g. ``category`` for metrics,
                ``default_before_z_score`` for selectors). Merged over
                :attr:`default_metadata`.

        Returns:
            Callable[[T], T]: Decorator that stores and returns the callable
            unchanged.
        """

        def decorator(func: T) -> T:
            key = cls._normalize(name)
            entry: Dict[str, Any] = dict(cls.default_metadata)
            entry.update(metadata)
            entry["func"] = func
            entry["display_name"] = display_name or name.replace("_", " ").title()
            cls._registry[key] = func
            cls._metadata[key] = entry
            return func

        return decorator

    @classmethod
    def get_entry(cls, name: str) -> Optional[Dict[str, Any]]:
        """
        Return the full metadata entry for ``name`` (running discovery if needed).

        Args:
            name: Registration key to look up.

        Returns:
            Optional[Dict[str, Any]]: Metadata dict (including ``func`` and
            ``display_name``), or ``None`` if not registered.
        """
        key = cls._normalize(name)
        if key not in cls._metadata:
            cls._discover()
        return cls._metadata.get(key)

    @classmethod
    def entries(cls) -> Dict[str, Dict[str, Any]]:
        """
        Return a copy of all registered metadata entries.

        Returns:
            Dict[str, Dict[str, Any]]: Mapping ``name -> metadata`` (running
            discovery if the registry is empty).
        """
        if not cls._registry:
            cls._discover()
        return dict(cls._metadata)
