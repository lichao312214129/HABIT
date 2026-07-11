"""Shared public contracts for configuration-driven HABIT workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Generic, Mapping, Optional, Type, TypeVar, Union, cast

from habit.api.exceptions import HABITAPIError

ConfigT = TypeVar("ConfigT")
DataT = TypeVar("DataT")
ConfigInput = Union[ConfigT, Mapping[str, Any]]

__all__ = ["ConfigInput", "WorkflowResult"]


@dataclass(frozen=True)
class WorkflowResult(Generic[DataT]):
    """
    Stable return value for a configuration-driven public workflow.

    ``data`` contains the in-memory value directly produced by the workflow,
    when available.  Workflows that only create files return ``None`` there and
    expose their canonical output directory through ``artifacts["output_dir"]``.
    """

    data: Optional[DataT] = None
    output_dir: Optional[Path] = None
    artifacts: Mapping[str, Path] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize path and mapping values to immutable public snapshots."""
        normalized_output_dir = (
            Path(self.output_dir) if self.output_dir is not None else None
        )
        normalized_artifacts = {
            name: Path(path) for name, path in self.artifacts.items()
        }
        if normalized_output_dir is not None:
            normalized_artifacts.setdefault("output_dir", normalized_output_dir)

        object.__setattr__(self, "output_dir", normalized_output_dir)
        object.__setattr__(
            self,
            "artifacts",
            MappingProxyType(normalized_artifacts),
        )
        object.__setattr__(
            self,
            "metrics",
            MappingProxyType(dict(self.metrics)),
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )

    def artifact(self, name: str) -> Path:
        """
        Return a named output artifact.

        Raises:
            KeyError: If this workflow did not declare an artifact named
                ``name``.
        """
        return self.artifacts[name]


def coerce_config(
    config: ConfigInput[ConfigT],
    config_type: Type[ConfigT],
) -> ConfigT:
    """
    Accept a validated configuration instance or construct one from a mapping.

    This gives YAML and programmatically constructed dictionaries exactly the
    same Pydantic validation path at every public workflow boundary.

    Args:
        config: Validated config instance or a mapping accepted by its schema.
        config_type: Public Pydantic configuration class expected by a runner.

    Returns:
        A validated instance of ``config_type``.

    Raises:
        HABITAPIError: If ``config`` is neither the requested schema type nor a
            mapping that can be validated by that schema.
    """
    if isinstance(config, config_type):
        return config
    if not isinstance(config, Mapping):
        raise HABITAPIError(
            f"config must be a {config_type.__name__} instance or a mapping; "
            f"received {type(config).__name__}."
        )

    from_dict = getattr(config_type, "from_dict", None)
    if callable(from_dict):
        return cast(ConfigT, from_dict(dict(config)))

    model_validate = getattr(config_type, "model_validate", None)
    if callable(model_validate):
        return cast(ConfigT, model_validate(dict(config)))

    raise HABITAPIError(
        f"{config_type.__name__} does not provide a supported validation method."
    )
