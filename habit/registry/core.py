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
"""ComponentRegistry: one name-to-implementation registry per plugin domain.

This builds on the established v0.1 registry base
(:class:`~habit.core.common.registry.ClassRegistry`) and adds the two things
the v1.0 plugin model needs: a declared ``domain`` (convention:
``domain == snake_case(ProtocolName)``, singular, so anyone implementing a
protocol already knows its domain) and parameter validation against the
registered Pydantic schema at construction time.

Entry points use the group name ``habit.<domain>``, so a third-party package
registers a component by declaring e.g. ``habit.supervoxelizer`` in its
package metadata -- no HABIT-side change required.
"""

from __future__ import annotations

from importlib import metadata as importlib_metadata
from typing import Any, ClassVar, Optional, Tuple, Type, TypeVar

from pydantic import BaseModel

from habit.exceptions import ComponentNotFoundError, ConfigurationError
from habit.core.common.registry import ClassRegistry

__all__ = ["ComponentRegistry"]

T = TypeVar("T")


class ComponentRegistry(ClassRegistry[Type[T]]):
    """
    Registry for ONE component family, keyed by implementation name.

    Subclasses declare their domain::

        class SupervoxelizerRegistry(ComponentRegistry[Supervoxelizer]):
            domain = "supervoxelizer"
            kind = "supervoxelizer"

    The full surface is then::

        @SupervoxelizerRegistry.register("slic")
        class SlicSupervoxelizer: ...

        SupervoxelizerRegistry.create("slic", n_supervoxels=100)
        SupervoxelizerRegistry.available()        # -> tuple of names
        SupervoxelizerRegistry.params_model("slic")  # -> Pydantic model | None
    """

    #: Plugin domain name; ``snake_case`` of the protocol class, singular.
    #: The entry point group is ``f"habit.{domain}"``.
    domain: ClassVar[str] = "component"

    @classmethod
    def create(cls, name: str, **params: Any) -> T:
        """
        Instantiate a registered component after validating ``params``.

        When a Pydantic parameters model is registered for ``name`` (via
        :meth:`register_params_model`), the parameters are validated and
        coerced through it before construction, so a mistyped parameter fails
        at the call site with a precise message instead of deep inside an
        algorithm.

        Args:
            name: Registered implementation name.
            **params: Parameters forwarded to the component constructor.

        Returns:
            The constructed component.

        Raises:
            ComponentNotFoundError: If the name is not registered.
            ConfigurationError: If the parameters fail schema validation.
        """
        target = cls.get(name)
        if target is None:
            raise ComponentNotFoundError(
                f"Unknown {cls.kind} {name!r} in domain '{cls.domain}'. "
                f"Available: {cls.available()}"
            )
        params_model = cls.get_params_model(name)
        if params_model is not None:
            try:
                validated = params_model.model_validate(params)
            except Exception as exc:
                raise ConfigurationError(
                    f"Invalid parameters for {cls.kind} {name!r}: {exc}"
                ) from exc
            # Field values are extracted WITHOUT model_dump(): serialisation
            # would turn rich constructor objects (e.g. a HabitatModel bound
            # to an assigner) into plain dicts, while attribute access keeps
            # the validated, type-coerced Python objects intact.
            params = {
                field: getattr(validated, field)
                for field in type(validated).model_fields
            }
        return target(**params)

    @classmethod
    def available(cls) -> Tuple[str, ...]:
        """Return the registered implementation names, sorted."""
        return tuple(sorted(super().available()))

    @classmethod
    def params_model(cls, name: str) -> Optional[Type[BaseModel]]:
        """
        Return the Pydantic model describing one implementation's parameters.

        JSON Schema for a GUI or an agent is then
        ``.model_json_schema()``; keeping a single source of truth avoids a
        second, drifting schema.

        Args:
            name: Registered implementation name.

        Returns:
            The params model class, or ``None`` when none was registered.
        """
        return cls.get_params_model(name)

    @classmethod
    def entry_point_group(cls) -> str:
        """Return the entry point group third-party packages register into."""
        return f"habit.{cls.domain}"

    @classmethod
    def load_entry_points(cls) -> Tuple[str, ...]:
        """
        Load third-party components declared under ``habit.<domain>``.

        An entry point may resolve to a module (whose registration decorators
        execute during import) or to a zero-argument callable performing
        registration. Loading is idempotent per entry point and failures are
        skipped by design: a broken third-party plugin must never prevent
        built-in components from working.

        Returns:
            Names of the entry points loaded by this call.
        """
        group = cls.entry_point_group()
        entry_points = importlib_metadata.entry_points()
        if hasattr(entry_points, "select"):
            selected = entry_points.select(group=group)
        else:  # pragma: no cover - Python 3.9 fallback
            selected = entry_points.get(group, ())
        loaded = []
        for entry_point in selected:
            identifier = f"{group}:{entry_point.name}"
            if identifier in cls._metadata.get("__loaded_entry_points__", {}):
                continue
            try:
                target = entry_point.load()
                if callable(target):
                    target()
            except Exception:
                continue
            cls._metadata.setdefault("__loaded_entry_points__", {})[identifier] = True
            loaded.append(entry_point.name)
        return tuple(loaded)
