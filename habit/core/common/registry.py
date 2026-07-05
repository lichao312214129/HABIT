# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text. Summary:
#
#   - Non-commercial use (academic, research, education, personal) is permitted
#     provided that copyright notices are retained and HABIT usage is
#     acknowledged in publications, reports, or documentation.
#   - Commercial use requires prior written consent from the copyright holder
#     (lichao19870617@163.com) and public acknowledgment of HABIT usage in
#     product documentation or user-facing materials.
#   - Unauthorized commercial use or removal of attribution is prohibited.
#
"""
Shared registry base for HABIT class-based factories.

Historically every subsystem defined its own ``*Factory`` with slightly
different method names (``create`` vs ``create_model`` vs ``create_algorithm``;
``get_available_*`` vs ``available``). :class:`ClassRegistry` provides a single,
uniform contract so that a developer learns the extension mechanism **once** and
applies it everywhere.

Canonical contract exposed by every subclass::

    @MyFactory.register("my_name")          # decorator: name -> class
    class MyImpl(MyBase):
        ...

    MyFactory.create("my_name", **kwargs)   # instantiate by name
    MyFactory.get("my_name")                 # -> class or None
    MyFactory.available()                    # -> list[str] of registered names
    MyFactory.register_params_model(name, m) # attach a Pydantic *Params schema
    MyFactory.get_params_model(name)         # -> params model or None

Notes
-----
* Each concrete subclass owns an **independent** ``_registry`` /
  ``_params_models`` mapping (created in :meth:`__init_subclass__`), so
  registrations never leak between domains.
* The ``_registry`` attribute name is part of the public contract because the
  GUI reflection bridge inspects it directly; do not rename it.
* Subclasses that discover implementations lazily (by importing sibling modules
  on first access) override :meth:`_discover`. Subclasses whose registration
  keys are case-insensitive override :meth:`_normalize`.
* Subclasses whose constructor call convention differs from ``cls_(**kwargs)``
  (for example, models that take a single positional ``config`` dict) override
  :meth:`create` only, and inherit everything else.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar

T = TypeVar("T")


class ClassRegistry(Generic[T]):
    """
    Generic ``name -> class`` registry shared by HABIT class-based factories.

    Subclass it to create a domain factory::

        class PreprocessorFactory(ClassRegistry[BasePreprocessor]):
            kind = "preprocessor"

    ``kind`` is only used to produce clear error messages.
    """

    #: Human-readable component kind, used only in error messages.
    kind: str = "component"

    #: Populated per-subclass in ``__init_subclass__`` (never shared).
    _registry: Dict[str, Type[T]]
    _params_models: Dict[str, Type[Any]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Give every concrete factory its own isolated storage."""
        super().__init_subclass__(**kwargs)
        cls._registry = {}
        cls._params_models = {}

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
        Lazily import sibling modules so decorated classes self-register.

        The default is a no-op (all implementations are imported eagerly).
        Override in subclasses that scan a package directory on first access.
        """
        return None

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Lookup / creation
    # ------------------------------------------------------------------
    @classmethod
    def get(cls, name: str) -> Optional[Type[T]]:
        """
        Return the registered class for ``name`` (running discovery if needed).

        Args:
            name: Registration key to look up.

        Returns:
            Optional[Type[T]]: The class, or ``None`` if not registered.
        """
        key = cls._normalize(name)
        if key not in cls._registry:
            cls._discover()
        return cls._registry.get(key)

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
