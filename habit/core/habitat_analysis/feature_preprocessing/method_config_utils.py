"""
Shared helpers for reading preprocessing method configuration objects.

Configs may be plain dicts or Pydantic ``PreprocessingMethod`` models; helpers
here normalize access across handlers, the pipeline runner, and validators.
"""

from __future__ import annotations

from typing import Any, Dict, Union

from pydantic import BaseModel

_METHOD_ALIASES: Dict[str, str] = {
    "z_score": "zscore",
    "standardization": "zscore",
    "min_max": "minmax",
    "normalize": "minmax",
}


def read_method_field(
    method_config: Union[Dict[str, Any], BaseModel],
    attr: str,
    default: Any = None,
) -> Any:
    """
    Read one field from a preprocessing step config.

    Args:
        method_config: Declarative config for one preprocessing step.
        attr: Attribute name (e.g. ``"variance_threshold"``).
        default: Value when the field is missing or explicitly ``None``.

    Returns:
        Resolved field value or ``default``.
    """
    if hasattr(method_config, attr):
        try:
            value = getattr(method_config, attr)
            return value if value is not None else default
        except (AttributeError, TypeError):
            pass

    if isinstance(method_config, dict):
        return method_config.get(attr, default)
    if hasattr(method_config, "get") and callable(getattr(method_config, "get")):
        return method_config.get(attr, default)

    return default


def normalize_method_name(method_name: str) -> str:
    """
    Map legacy YAML aliases to the canonical registry key.

    Args:
        method_name: Raw ``method`` string from configuration.

    Returns:
        Canonical lowercase method name used by ``PreprocessingMethodFactory``.
    """
    key = method_name.lower()
    return _METHOD_ALIASES.get(key, key)


def resolve_method_name(method_config: Union[Dict[str, Any], BaseModel]) -> str:
    """
    Resolve the canonical preprocessing method name from a step config.

    Args:
        method_config: Single preprocessing step configuration.

    Returns:
        Canonical method name (defaults to ``minmax`` when absent).
    """
    raw = read_method_field(method_config, "method", "minmax")
    return normalize_method_name(str(raw))


def parse_winsor_limits(
    method_config: Union[Dict[str, Any], BaseModel],
    default: tuple[float, float] = (0.05, 0.05),
) -> tuple[float, float]:
    """
    Parse winsorize lower/upper quantile limits from config.

    Args:
        method_config: Preprocessing step configuration.
        default: Fallback limits when ``winsor_limits`` is absent.

    Returns:
        Tuple ``(lower_fraction, upper_fraction)`` in [0, 1].
    """
    raw = read_method_field(method_config, "winsor_limits", None)
    if raw is None:
        return default
    if isinstance(raw, list):
        return float(raw[0]), float(raw[1])
    return raw
